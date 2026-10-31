"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

DEADLINE = 145.0
START_TIME = time.time()


def _lj_energy_and_grad(positions):
    """Compute LJ energy and gradient."""
    n = len(positions)
    energy = 0.0
    grad = np.zeros_like(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20, grad
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 ** 3
            inv_r12 = inv_r6 ** 2
            e = 4.0 * (inv_r12 - inv_r6)
            energy += e
            # Gradient: dE/dr * dr/dri
            # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 4*(-12*inv_r12/r + 6*inv_r6/r)
            # dr/dri = -rij/r
            # So dE/dri = 4*(12*inv_r12 - 6*inv_r6) * rij / r^2
            #          = 4*(12*inv_r12 - 6*inv_r6) * rij * inv_r2
            coeff = 4.0 * (12.0 * inv_r12 - 6.0 * inv_r6) * inv_r2
            grad[i] -= coeff * rij
            grad[j] += coeff * rij
    return energy, grad


def _lj_energy_vec(positions):
    """Vectorized LJ energy."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4.0 * np.sum(inv_r12 - inv_r6) / 2.0


def _ensure_min_sep(positions, min_dist=0.5):
    """Ensure all atoms are at least min_dist apart by local displacement."""
    positions = positions.copy()
    for _ in range(100):
        moved = False
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                rij = positions[j] - positions[i]
                r = np.linalg.norm(rij)
                if r < min_dist:
                    if r < 1e-10:
                        direction = np.array([1.0, 0.0, 0.0])
                    else:
                        direction = rij / r
                    push = (min_dist - r) / 2.0 + 1e-6
                    positions[i] -= push * direction
                    positions[j] += push * direction
                    moved = True
        if not moved:
            break
    return positions


def _generate_fcc_points():
    """Generate FCC lattice points with NN distance = 1."""
    scale = 1.0 / math.sqrt(2.0)
    max_coord = 5
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i * scale, j * scale, k * scale])
                    points.append((p, np.dot(p, p)))
    points.sort(key=lambda x: x[1])
    return [p[0] for p in points]


def _get_core_and_outer(fcc_points):
    """
    Identify the 38-atom FCC truncated octahedron structure.
    Core: 1 center + 12 first-shell (NN) = 13
    Outer: 25 atoms from second/third shells
    
    FCC shells by r^2 (with scale=1/sqrt(2)):
    r^2=0: 1 atom (center)
    r^2=1: 12 atoms (NN, r=1/sqrt(2)*1 = 0.707... wait)
    
    Actually with scale = 1/sqrt(2), the lattice points are at (i,j,k)*scale
    where i+j+k is even. The nearest neighbor distance is sqrt(2)*scale = 1.
    
    Let me recalculate: if scale = 1/sqrt(2), then a point at (1,1,0) has 
    distance sqrt(2)*scale = sqrt(2)/sqrt(2) = 1. Good.
    
    Shells:
    r^2 = 0: (0,0,0) - 1 atom
    r^2 = 1: (1,1,0) type - 12 atoms (permutations of (±1,±1,0))
    r^2 = 2: (1,0,1) type - wait, (1,0,1) has i+j+k=2, even. Distance = sqrt(2)*scale = 1. 
    Actually (1,0,1) has r^2 = (1^2+0+1^2)*scale^2 = 2*(1/2) = 1. So same shell.
    
    Let me just list by actual distance:
    (0,0,0): r=0
    (1,1,0) and perms: r = sqrt(2)*scale = 1, count = 12
    (2,0,0) and perms: r = 2*scale = sqrt(2) ≈ 1.414, count = 6
    (2,2,0) and perms: r = sqrt(8)*scale = 2, count = 12
    (2,1,1) and perms: r = sqrt(6)*scale = sqrt(3) ≈ 1.732, count = 24
    (3,1,0) and perms: r = sqrt(10)*scale = sqrt(5) ≈ 2.236, count = 24
    (4,0,0) and perms: r = 4*scale = 2*sqrt(2) ≈ 2.828, count = 6
    (4,2,0) and perms: r = sqrt(20)*scale = sqrt(10) ≈ 3.162, count = 24
    (4,4,0) and perms: r = sqrt(32)*scale = 4, count = 12
    (4,2,2) and perms: r = sqrt(24)*scale = 2*sqrt(3) ≈ 3.464, count = 24
    
    For 38 atoms: 1 + 12 + 6 + 12 + 7 = 38
    So: center (1) + first shell (12) + second shell (6) + third shell (12) + 7 from fourth shell
    
    Or: 1 + 12 + 6 + 12 + 24 = 55, too many.
    1 + 12 + 6 + 12 = 31, need 7 more from the 24 in the next shell.
    
    Actually, the truncated octahedron for FCC38 typically has:
    - 1 center
    - 12 first neighbors (octahedral + tetrahedral directions)
    - 25 outer atoms forming the truncated octahedron surface
    
    Let me just take the 38 closest FCC points.
    """
    # Take 38 closest points from FCC lattice
    selected = fcc_points[:38]
    return np.array(selected, dtype=float)


def _make_icosahedral_structure():
    """Create an icosahedral cluster for LJ38."""
    phi = (1 + math.sqrt(5)) / 2.0
    
    # 12 vertices of icosahedron
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    # 30 edge midpoints
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_verts[i] - ico_verts[j])
            if d < 1.2:  # Nearest neighbors
                edges.append((i, j))
    
    edge_mids = []
    for i, j in edges[:15]:  # Take 15 edge midpoints
        mid = (ico_verts[i] + ico_verts[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids)
    edge_mids /= np.linalg.norm(edge_mids[0])
    
    # 20 face centers (take 10)
    # Find triangular faces
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d2 = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d3 = np.linalg.norm(ico_verts[k] - ico_verts[i])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    faces.append((i, j, k))
    
    face_centers = []
    for i, j, k in faces[:10]:
        fc = (ico_verts[i] + ico_verts[j] + ico_verts[k]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    face_centers /= np.linalg.norm(face_centers[0])
    
    # Build structure: 1 center + 12 ico + 15 edge_mids + 10 face_centers = 38
    center = np.array([[0.0, 0.0, 0.0]])
    r1 = 1.12
    r2 = 1.5
    r3 = 1.3
    
    all_points = np.vstack([
        center,
        ico_verts * r1,
        edge_mids * r2,
        face_centers * r3
    ])
    
    return all_points[:38]


def _make_truncated_octahedron_start():
    """
    Create a truncated octahedron-like structure for LJ38.
    
    The FCC truncated octahedron for 38 atoms has:
    - 1 center
    - 12 atoms at first shell (NN distance ~1.0)
    - 25 atoms at outer shells
    
    We'll use the 38 closest FCC points as the base.
    """
    fcc_points = _generate_fcc_points()
    positions = _get_core_and_outer(fcc_points)
    return positions


def _relax(positions, maxiter=200):
    """Relax positions using L-BFGS."""
    try:
        from science_chemistry import relax
        return relax(positions)
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback to scipy
    def obj(x):
        p = x.reshape(N, 3)
        e, g = _lj_energy_and_grad(p)
        return e, g.flatten()
    
    try:
        result = minimize(obj, positions.flatten(), method='L-BFGS-B', jac=True,
                         options={'maxiter': maxiter, 'gtol': 1e-8})
        return result.x.reshape(N, 3)
    except Exception:
        return positions


def _optimize_scale(positions, scale_range=(0.95, 1.15), n_steps=20):
    """Optimize the global scale factor."""
    best_scale = 1.0
    best_energy = _lj_energy_vec(positions)
    
    for scale in np.linspace(scale_range[0], scale_range[1], n_steps):
        scaled = positions * scale
        scaled = _ensure_min_sep(scaled)
        energy = _lj_energy_vec(scaled)
        if energy < best_energy:
            best_energy = energy
            best_scale = scale
    
    return positions * best_scale, best_energy


def _basin_hop(positions, n_hops=10, maxiter=100):
    """
    Simple basin hopping: perturb, relax, keep if better.
    """
    best_pos = positions.copy()
    best_energy = _lj_energy_vec(positions)
    rng = np.random.default_rng(42)
    
    for _ in range(n_hops):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        # Perturb
        perturbed = best_pos + rng.normal(0, 0.1, size=best_pos.shape)
        perturbed = _ensure_min_sep(perturbed)
        
        # Relax
        relaxed = _relax(perturbed, maxiter=maxiter)
        relaxed = _ensure_min_sep(relaxed)
        
        energy = _lj_energy_vec(relaxed)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    
    return best_pos, best_energy


def construct():
    """Main entry point: find the global minimum of LJ38."""
    global START_TIME
    START_TIME = time.time()
    
    best_positions = None
    best_energy = float('inf')
    
    # Phase 1: Start from FCC truncated octahedron
    try:
        fcc_pos = _make_truncated_octahedron_start()
        fcc_pos = _ensure_min_sep(fcc_pos)
        
        # Optimize scale
        fcc_pos, fcc_energy = _optimize_scale(fcc_pos)
        
        # Relax
        fcc_pos = _relax(fcc_pos, maxiter=300)
        fcc_pos = _ensure_min_sep(fcc_pos)
        fcc_energy = _lj_energy_vec(fcc_pos)
        
        if fcc_energy < best_energy:
            best_energy = fcc_energy
            best_positions = fcc_pos
    except Exception:
        pass
    
    # Phase 2: Start from icosahedral structure
    try:
        ico_pos = _make_icosahedral_structure()
        ico_pos = _ensure_min_sep(ico_pos)
        
        ico_pos, ico_energy = _optimize_scale(ico_pos)
        
        ico_pos = _relax(ico_pos, maxiter=300)
        ico_pos = _ensure_min_sep(ico_pos)
        ico_energy = _lj_energy_vec(ico_pos)
        
        if ico_energy < best_energy:
            best_energy = ico_energy
            best_positions = ico_pos
    except Exception:
        pass
    
    # Phase 3: Basin hopping from best so far
    if best_positions is not None:
        try:
            bh_pos, bh_energy = _basin_hop(best_positions, n_hops=20, maxiter=150)
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_pos
        except Exception:
            pass
    
    # Phase 4: More diverse starts with basin hopping
    if time.time() - START_TIME < DEADLINE - 30:
        rng = np.random.default_rng(123)
        
        for trial in range(50):
            if time.time() - START_TIME > DEADLINE - 15:
                break
            
            try:
                # Random start near FCC structure
                if best_positions is not None:
                    start = best_positions + rng.normal(0, 0.3, size=(N, 3))
                else:
                    fcc_points = _generate_fcc_points()
                    start = fcc_points[:N].copy()
                    start += rng.normal(0, 0.2, size=start.shape)
                
                start = _ensure_min_sep(start)
                
                # Try different scales
                scale = rng.uniform(0.95, 1.15)
                start = start * scale
                start = _ensure_min_sep(start)
                
                # Relax
                relaxed = _relax(start, maxiter=100)
                relaxed = _ensure_min_sep(relaxed)
                
                energy = _lj_energy_vec(relaxed)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
                    
                    # Do a quick basin hop from this new best
                    if time.time() - START_TIME < DEADLINE - 20:
                        bh_pos, bh_energy = _basin_hop(relaxed, n_hops=5, maxiter=100)
                        if bh_energy < best_energy:
                            best_energy = bh_energy
                            best_positions = bh_pos
            except Exception:
                pass
    
    # Phase 5: Final refinement
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 5:
        try:
            # Optimize scale one more time
            final_pos, final_energy = _optimize_scale(best_positions, (0.98, 1.08), 15)
            
            # Final relaxation
            final_pos = _relax(final_pos, maxiter=500)
            final_pos = _ensure_min_sep(final_pos)
            final_energy = _lj_energy_vec(final_pos)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos
        except Exception:
            pass
    
    # Fallback
    if best_positions is None:
        fcc_points = _generate_fcc_points()
        best_positions = _ensure_min_sep(np.array(fcc_points[:N], dtype=float))
    
    return best_positions

# EVOLVE-BLOCK-END
