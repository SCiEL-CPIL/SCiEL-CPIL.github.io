import numpy as np
import time
import math
from scipy.optimize import minimize

N = 38
DEADLINE = 142.0
START_TIME = time.time()

def lj_energy(positions):
    """Compute total Lennard-Jones energy: 4 * sum(r^-12 - r^-6) over all pairs."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            r2 = float(np.dot(rij, rij))
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
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    return energy

def check_validity(positions, min_dist=0.5):
    """Check that all atoms are at least min_dist apart."""
    n = len(positions)
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    return np.all(r2 >= min_dist * min_dist)

def make_fcc_truncated_octahedron(n_atoms=38):
    """
    Generate the FCC truncated octahedron cluster for n_atoms atoms.
    FCC lattice: points (i, j, k) where i+j+k is even.
    Nearest neighbor distance in lattice units = sqrt(2).
    Scale by 2^(1/6)/sqrt(2) to reach LJ equilibrium distance.
    """
    max_coord = 6
    points = []
    
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    x, y, z = float(i), float(j), float(k)
                    r2 = x*x + y*y + z*z
                    points.append((x, y, z, r2))
    
    points.sort(key=lambda p: (p[3], abs(p[0]), abs(p[1]), abs(p[2])))
    selected = [p[:3] for p in points[:n_atoms]]
    coords = np.array(selected, dtype=float)
    # Scale so nearest-neighbor distance = 2^(1/6) (LJ minimum)
    # NN in lattice = sqrt(2), so multiply by 2^(1/6)/sqrt(2) = 2^(-1/3)
    coords *= 2.0 ** (-1.0 / 3.0)
    return coords

def _lj_energy_and_grad(pos_flat):
    """Compute LJ energy and gradient vectorized."""
    pos = pos_flat.reshape(N, 3)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(r2, 1.0)
    r2 = np.maximum(r2, 1e-12)
    inv_r2 = 1.0 / r2
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    energy = 4.0 * np.sum(inv_r12 - inv_r6) / 2.0
    inv_r14 = inv_r12 * inv_r2
    inv_r8 = inv_r6 * inv_r2
    coeff = -24.0 * inv_r14 + 12.0 * inv_r8
    grad_matrix = coeff[:, :, np.newaxis] * diff
    grad_flat = np.sum(grad_matrix, axis=1).flatten()
    return energy, grad_flat

def relax(positions, maxiter=400, gtol=1e-9):
    """Relax using L-BFGS-B with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 60, 'ftol': 1e-12})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def polish(positions, maxiter=800, gtol=1e-12):
    """High-precision final polish pass with analytic gradient."""
    x0 = positions.flatten()
    try:
        result = minimize(_lj_energy_and_grad, x0, method='L-BFGS-B',
                          jac=True,
                          options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 100, 'ftol': 1e-15})
        return result.x.reshape(N, 3)
    except Exception:
        return positions

def basin_hopping(positions, n_iter=15, step_size=0.4, maxiter=120, temperature=0.05):
    """Perform basin hopping optimization with analytic gradient."""
    x0 = positions.flatten()
    current_energy, _ = _lj_energy_and_grad(x0)
    best_energy = current_energy
    best_x = x0.copy()
    rng = np.random.default_rng(99)
    
    for i in range(n_iter):
        if time.time() - START_TIME > DEADLINE - 12:
            break
        noise = rng.normal(0, step_size, x0.shape)
        perturbed = x0 + noise
        try:
            result = minimize(_lj_energy_and_grad, perturbed, method='L-BFGS-B',
                               jac=True,
                               options={'maxiter': maxiter, 'gtol': 1e-7, 'maxls': 60})
            new_x = result.x
            new_energy, _ = _lj_energy_and_grad(new_x)
        except Exception:
            continue
        
        if new_energy < current_energy:
            x0 = new_x
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_x = new_x.copy()
        else:
            delta = new_energy - current_energy
            if delta < 0:
                accept = 1.0
            else:
                accept = math.exp(-delta / temperature)
            if rng.uniform() < accept:
                x0 = new_x
                current_energy = new_energy
    
    return best_x.reshape(N, 3), best_energy

def _make_icosahedral(n_atoms=38):
    """Build an icosahedron + inner core structure."""
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    verts = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                verts.append((sx, sy*phi, sz*phi))
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    outer_r = 2.8
    inner_r = 1.3
    outer = verts * outer_r
    inner = verts * inner_r
    ang = math.pi / 6.0
    R = np.array([[math.cos(ang), -math.sin(ang), 0],
                  [math.sin(ang), math.cos(ang), 0],
                  [0, 0, 1]])
    inner = inner @ R.T
    center = np.array([[0.0, 0.0, 0.0]])
    pts = np.vstack([center, inner, outer])
    if len(pts) >= n_atoms:
        pts = pts[:n_atoms]
    else:
        rng = np.random.default_rng(0)
        extra = rng.normal(0, 0.5, (n_atoms - len(pts), 3))
        pts = np.vstack([pts, extra])
    pts *= 2.0 ** (1.0 / 6.0) / 1.4
    return pts

def _make_bcc_fragment(n_atoms=38):
    """Build a BCC-like cubic lattice fragment."""
    side = int(math.ceil(N ** (1.0/3.0)))
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                pts.append((i - side//2, j - side//2, k - side//2))
    pts.sort(key=lambda p: p[0]*p[0] + p[1]*p[1] + p[2]*p[2])
    pts = pts[:n_atoms]
    arr = np.array(pts, dtype=float)
    arr *= 1.28 * 2.0 ** (1.0/6.0)
    return arr

def _make_fcc_with_noise(n_atoms=38, noise_scale=0.05, seed=42):
    """Generate FCC cluster with small noise to break symmetry."""
    base = make_fcc_truncated_octahedron(n_atoms)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_scale, base.shape)
    return base + noise

def _make_initial_clusters():
    """Create diverse initial structures for multi-start optimization."""
    clusters = []
    
    # Primary: FCC truncated octahedron
    try:
        fcc = make_fcc_truncated_octahedron(N)
        if check_validity(fcc):
            clusters.append(fcc)
    except Exception:
        pass
    
    # Small-noise FCC variants
    for s in (1, 2, 3, 4, 5, 6):
        try:
            c = _make_fcc_with_noise(N, noise_scale=0.04, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    
    # Larger-noise FCC variants
    for s in (7, 8, 9, 10):
        try:
            c = _make_fcc_with_noise(N, noise_scale=0.10, seed=s)
            if check_validity(c):
                clusters.append(c)
        except Exception:
            pass
    
    # BCC-like fragment
    try:
        arr = _make_bcc_fragment(N)
        if check_validity(arr):
            clusters.append(arr)
    except Exception:
        pass
    
    # Icosahedral-inspired
    try:
        ic = _make_icosahedral(N)
        if check_validity(ic):
            clusters.append(ic)
    except Exception:
        pass
    
    return clusters

def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial structures
    candidates = _make_initial_clusters()
    
    # Relax each candidate and track the best
    for cand in candidates:
        if time.time() - START_TIME > DEADLINE - 30:
            break
        try:
            relaxed = relax(cand, maxiter=300, gtol=1e-8)
            if check_validity(relaxed):
                # 1D scale optimization
                best_scale = 1.0
                best_scale_energy = lj_energy_vectorized(relaxed)
                for s in [0.95, 0.97, 0.99, 1.01, 1.03, 1.05]:
                    scaled = relaxed * s
                    if check_validity(scaled):
                        e = lj_energy_vectorized(scaled)
                        if e < best_scale_energy:
                            best_scale_energy = e
                            best_scale = s
                # Refine scale
                lo, hi = 0.90, 1.10
                if best_scale > 0.95:
                    lo = max(0.90, best_scale - 0.05)
                if best_scale < 1.05:
                    hi = min(1.10, best_scale + 0.05)
                for _ in range(20):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    e1 = lj_energy_vectorized(relaxed * m1)
                    e2 = lj_energy_vectorized(relaxed * m2)
                    if e1 < e2:
                        hi = m2
                    else:
                        lo = m1
                optimal_scale = (lo + hi) / 2.0
                if check_validity(relaxed * optimal_scale):
                    final_energy = lj_energy_vectorized(relaxed * optimal_scale)
                    if final_energy < best_scale_energy:
                        best_scale_energy = final_energy
                    relaxed = relaxed * optimal_scale
                if best_scale_energy < best_energy:
                    best_energy = best_scale_energy
                    best_positions = relaxed
        except Exception:
            continue
    
    # Basin hopping from the best structure if time permits
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 25:
        try:
            bh_result, bh_energy = basin_hopping(best_positions, n_iter=20, step_size=0.35, maxiter=100, temperature=0.05)
            if check_validity(bh_result) and bh_energy < best_energy:
                best_energy = bh_energy
                best_positions = bh_result
        except Exception:
            pass
    
    # Final high-precision polish with scale optimization
    if best_positions is not None and time.time() - START_TIME < DEADLINE - 10:
        try:
            polished = polish(best_positions, maxiter=800, gtol=1e-12)
            if check_validity(polished):
                # Scale optimization on polished structure
                lo, hi = 0.92, 1.08
                for _ in range(25):
                    m1 = lo + (hi - lo) / 3.0
                    m2 = hi - (hi - lo) / 3.0
                    e1 = lj_energy_vectorized(polished * m1)
                    e2 = lj_energy_vectorized(polished * m2)
                    if e1 < e2:
                        hi = m2
                    else:
                        lo = m1
                opt_scale = (lo + hi) / 2.0
                if check_validity(polished * opt_scale):
                    polished = polished * opt_scale
                energy = lj_energy_vectorized(polished)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = polished
        except Exception:
            pass
    
    # Final check and return
    if best_positions is None:
        # Fallback
        try:
            best_positions = make_fcc_truncated_octahedron(N)
            if not check_validity(best_positions):
                side = int(np.ceil(N ** (1/3)))
                pts = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
                pts.sort(key=lambda p: (p[0]-(side-1)/2)**2 + (p[1]-(side-1)/2)**2 + (p[2]-(side-1)/2)**2)
                best_positions = np.array(pts[:N], dtype=float) * 1.2
        except Exception:
            best_positions = np.zeros((N, 3))
            for i in range(N):
                best_positions[i] = [i*0.6, (i%3)*0.6, (i//3)%3*0.6]
    
    return best_positions
