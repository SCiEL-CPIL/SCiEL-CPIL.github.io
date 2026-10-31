import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 145.0  # Safety margin for 150s limit
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = np.dot(rij, rij)
            if r2 < 1e-12:
                return 1e20
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 * inv_r2 * inv_r2
            inv_r12 = inv_r6 * inv_r6
            energy += 4.0 * (inv_r12 - inv_r6)
    return energy

def lj_energy_vectorized(positions):
    """Vectorized LJ energy computation."""
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)  # Avoid division by zero on diagonal
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r = np.sqrt(np.dot(rij, rij))
            if r < min_dist:
                return False
    return True

def make_fcc_cluster(n_atoms=38):
    """Generate an FCC truncated octahedron cluster for LJ38.
    
    The known global minimum is a truncated octahedron from the FCC lattice.
    For 38 atoms: 1 center + 12 first shell + 25 second shell (truncated).
    """
    # FCC lattice vectors with nearest-neighbor distance = 1
    # a = 1/sqrt(2) so that |a1-a2| = 1
    a = 1.0 / math.sqrt(2.0)
    
    # Generate FCC lattice points: (i,j,k) where i+j+k is even, scaled by a
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = i * a, j * a, k * a
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2, (i, j, k)))
    
    # Sort by distance from origin
    points.sort(key=lambda p: p[3])
    
    # For LJ38 truncated octahedron:
    # Shell 0: 1 atom at origin (r=0)
    # Shell 1: 12 atoms at r = 1.0 (nearest neighbors in FCC)
    # Shell 2: 25 atoms at r ≈ sqrt(2) ≈ 1.414 (next-nearest)
    # Total: 1 + 12 + 25 = 38
    
    # Select the 38 closest points
    selected = [p[:3] for p in points[:n_atoms]]
    return np.array(selected, dtype=float)

def make_bcc_cluster(n_atoms=38):
    """Generate a BCC cluster by selecting closest lattice points to the origin."""
    # BCC lattice: corners of cubic cell + center
    # To get NN distance 1, we need to scale appropriately.
    # In BCC, NN distance is sqrt(3)/2 * a. If NN=1, a = 2/sqrt(3).
    # Points: (i, j, k) and (i+0.5, j+0.5, k+0.5) scaled by a.
    a = 2.0 / math.sqrt(3.0)
    max_coord = 4
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                # Corners
                points.append((i*a, j*a, k*a))
                # Body center
                points.append(((i+0.5)*a, (j+0.5)*a, (k+0.5)*a))
    
    # Remove duplicates (shouldn't be any if max_coord is integer)
    # Sort by distance
    points.sort(key=lambda p: p[0]**2 + p[1]**2 + p[2]**2)
    return np.array(points[:n_atoms], dtype=float)

def make_icosahedral_cluster(n_atoms=38):
    """Generate a nested icosahedral cluster."""
    phi = (1 + math.sqrt(5)) / 2.0
    r_ico = 1.0 / math.sqrt(1 + phi * phi)
    
    # 12 vertices of icosahedron
    ico_verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * r_ico
    
    points = [np.array([0.0, 0.0, 0.0])]  # Center
    points.extend([v * 1.12 for v in ico_verts])  # First shell
    
    # Need 25 more atoms.
    # Add points on the faces or edges.
    # For simplicity, add a second shell of icosahedron vertices scaled up?
    # Or just add random points in a shell?
    # Let's add 25 points on a sphere of radius 1.6
    rng = np.random.default_rng(42)
    for _ in range(n_atoms - len(points)):
        u = rng.uniform(0, 1)
        v = rng.uniform(0, 1)
        theta = 2 * math.pi * u
        z = 2 * v - 1
        r = math.sqrt(1 - z*z)
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        points.append(np.array([x, y, z]) * 1.6)
        
    return np.array(points, dtype=float)

def relax(positions, maxiter=200, gtol=1e-6):
    """Relax using L-BFGS-B."""
    def energy_func(pos_flat):
        pos = pos_flat.reshape(N, 3)
        return lj_energy(pos)
    
    # Flatten input
    x0 = positions.flatten()
    
    try:
        result = minimize(energy_func, x0, method='L-BFGS-B', 
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions):
    """High-precision polish pass with tighter tolerance."""
    if time.time() - START_TIME > DEADLINE - 10:
        return positions
    
    try:
        polished = relax(positions, maxiter=500, gtol=1e-10)
        if check_validity(polished):
            e1 = lj_energy_vectorized(positions)
            e2 = lj_energy_vectorized(polished)
            if e2 < e1:
                return polished
    except Exception:
        pass
    return positions

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    candidates = []
    
    # 1. FCC Cluster (Global Min Candidate)
    try:
        fcc = make_fcc_cluster(N)
        if check_validity(fcc):
            candidates.append(fcc)
    except Exception:
        pass
        
    # 2. Perturbed FCC (to break symmetry if needed, though not strictly necessary for global min)
    try:
        fcc_pert = make_fcc_cluster(N)
        rng = np.random.default_rng(123)
        fcc_pert += rng.normal(0, 0.1, fcc_pert.shape)
        if check_validity(fcc_pert):
            candidates.append(fcc_pert)
    except Exception:
        pass
        
    # 3. BCC Cluster
    try:
        bcc = make_bcc_cluster(N)
        if check_validity(bcc):
            candidates.append(bcc)
    except Exception:
        pass
        
    # 4. Icosahedral Cluster
    try:
        ico = make_icosahedral_cluster(N)
        if check_validity(ico):
            candidates.append(ico)
    except Exception:
        pass
        
    # Relax all candidates
    for i, cand in enumerate(candidates):
        if time.time() - START_TIME > DEADLINE - 15:
            break
            
        try:
            relaxed = relax(cand)
            if not check_validity(relaxed):
                continue
                
            energy = lj_energy_vectorized(relaxed)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    # Final high-precision polish on the best result
    if best_positions is not None:
        if time.time() - START_TIME < DEADLINE - 5:
            try:
                polished = polish(best_positions)
                e_polished = lj_energy_vectorized(polished)
                if e_polished < best_energy:
                    best_energy = e_polished
                    best_positions = polished
            except Exception:
                pass
    
    if best_positions is None:
        # Fallback to unrelaxed FCC
        try:
            best_positions = make_fcc_cluster(N)
            if not check_validity(best_positions):
                # Last resort: simple cubic grid
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            # Force validity by spreading out
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]

    return best_positions
