"""Two-stage topology-forcing strategy for LJ38 global minimum."""
import numpy as np
import time
from scipy.optimize import minimize, fmin_l_bfgs_b

N = 38

# Deadline guard: 150s wall-clock, safety margin 10s
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def _elapsed():
    return time.time() - _t0

def _time_left():
    return DEADLINE - _elapsed() - SAFETY_MARGIN

def lj_energy(pos):
    """Compute total LJ energy for N atoms."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r6 = r2**(-3)
    r12 = r6**2
    energy = 4.0 * np.sum(r12 - r6)
    return energy

def lj_forces(pos):
    """Compute forces (negative gradient of energy)."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    r6 = r2**(-3)
    r12 = r6**2
    # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 4*(6*r^-7 - 12*r^-13)
    # Force on i from j: 4*(12*r^-13 - 6*r^-7) * (r_i - r_j)/r
    # In vector form: F_i = sum_j 4*(12*r_ij^-13 - 6*r_ij^-7) * (r_i - r_j)
    # = sum_j 4*(12*r6/r2 - 6*r6/r2*r6) * diff_ij ... let me be careful
    # F_i = sum_{j!=i} 4*(12/r^13 - 6/r^7) * (r_i - r_j)
    # = sum_{j!=i} 4*(12*r6^2 - 6*r6) / r^2 * diff_ij  ... no
    # Let r_ij = |r_i - r_j|, diff_ij = r_i - r_j
    # F_i = sum_j 4*(12/r_ij^13 - 6/r_ij^7) * diff_ij
    # = sum_j 4*(12*r6^2/r2 - 6*r6/r2) * diff_ij ... 
    # Actually: 1/r^13 = r6^2 / r2, 1/r^7 = r6/r2
    # So coefficient = 4*(12*r6^2/r2 - 6*r6/r2) = 4*(12*r6^2 - 6*r6)/r2
    coeff = 4.0 * (12.0 * r12 - 6.0 * r6) / r2
    np.fill_diagonal(coeff, 0.0)
    forces = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return forces

def relax(pos):
    """L-BFGS local relaxation."""
    def fun(x):
        p = x.reshape(-1, 3)
        e = lj_energy(p)
        f = -lj_forces(p).flatten()
        return e, f
    
    result = minimize(fun, pos.flatten(), jac=True, method='L-BFGS-B',
                      options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
    return result.x.reshape(-1, 3), result.fun

def _fcc_truncated_octahedron():
    """
    Generate the FCC truncated octahedron geometry for LJ38.
    
    The FCC truncated octahedron has:
    - 1 atom at center
    - 12 atoms at first shell (vertices of cuboctahedron)
    - 18 atoms at second shell (face centers of truncated octahedron)
    - 7 atoms at third shell (remaining vertices)
    Total: 1 + 12 + 18 + 7 = 38
    
    Actually, the standard FCC truncated octahedron for n=38:
    - Center: 1
    - 1st shell: 12 (cuboctahedron vertices) at distance ~1.0
    - 2nd shell: 18 at distance ~1.414 (face centers)
    - 3rd shell: 7 at distance ~1.732
    
    Let me use the known coordinates. The FCC truncated octahedron has:
    - 1 center atom
    - 12 atoms at (±1, ±1, 0) type positions (cuboctahedron)
    - 18 atoms at (±2, 0, 0), (0, ±2, 0), (0, 0, ±2) and (±1, ±1, ±1) type... 
    
    Actually let me just generate FCC lattice points within a truncated octahedron radius.
    """
    # FCC lattice: points (i,j,k) where i+j+k is even, scaled by 1/sqrt(2)
    # Truncated octahedron: |x|+|y|+|z| <= a and |x|,|y|,|z| <= b
    
    # Generate FCC points
    pts = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                if (i + j + k) % 2 == 0:
                    x, y, z = i / np.sqrt(2), j / np.sqrt(2), k / np.sqrt(2)
                    # Truncated octahedron condition: max(|x|,|y|,|z|) <= 2/sqrt(2)=sqrt(2)
                    # and |x|+|y|+|z| <= 3/sqrt(2) approximately
                    if max(abs(x), abs(y), abs(z)) <= 1.5 and abs(x)+abs(y)+abs(z) <= 2.5:
                        pts.append([x, y, z])
    
    pts = np.array(pts)
    # Sort by distance from center
    dists = np.sum(pts**2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    
    # Take first 38
    if len(pts) >= 38:
        return pts[:38].copy()
    
    # If not enough, expand
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                if (i + j + k) % 2 == 0:
                    x, y, z = i / np.sqrt(2), j / np.sqrt(2), k / np.sqrt(2)
                    if max(abs(x), abs(y), abs(z)) <= 2.0 and abs(x)+abs(y)+abs(z) <= 3.0:
                        pts = np.vstack([pts, [[x, y, z]]])
    
    dists = np.sum(pts**2, axis=1)
    order = np.argsort(dists)
    pts = pts[order]
    return pts[:38].copy()

def _topology_loss(pos, target_pos):
    """
    Smooth differentiable loss that penalizes deviations from target geometry.
    Uses pairwise distance matching (a smooth proxy for structural similarity).
    """
    # Compute all pairwise distances for pos and target
    def pairwise_dists(p):
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=-1) + 1e-12)
    
    d_pos = pairwise_dists(pos)
    d_tgt = pairwise_dists(target_pos)
    
    # Only consider upper triangle (i < j)
    i_idx, j_idx = np.triu_indices(len(pos), k=1)
    loss = np.sum((d_pos[i_idx, j_idx] - d_tgt[i_idx, j_idx])**2)
    return loss

def _topology_loss_grad(pos, target_pos):
    """Gradient of topology loss w.r.t. positions."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff**2, axis=-1) + 1e-12
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)  # avoid division by zero
    
    def pairwise_dists(p):
        d = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        return np.sqrt(np.sum(d**2, axis=-1) + 1e-12)
    
    d_tgt = pairwise_dists(target_pos)
    
    # d_loss/d_pos_i = 2 * sum_j (d_ij - d_tgt_ij) * (pos_i - pos_j) / d_ij
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            dij = r[i, j]
            dtij = d_tgt[i, j]
            coeff = 2.0 * (dij - dtij) / dij
            grad[i] += coeff * diff[i, j]
    
    return grad

def _add_noise(pos, sigma=0.3):
    """Add random noise to break symmetry."""
    return pos + np.random.normal(0, sigma, size=pos.shape)

def construct():
    global _t0
    _t0 = time.time()
    np.random.seed(42)
    
    best_pos = None
    best_energy = np.inf
    
    # Phase 1: Generate FCC truncated octahedron target
    fcc_target = _fcc_truncated_octahedron()
    
    # Phase 2: Multi-start topology-forcing + LJ relaxation
    # Start from FCC target with various perturbations
    starts = []
    
    # Start 0: FCC target itself
    starts.append(fcc_target.copy())
    
    # Start 1-5: FCC target with small noise
    for s in [0.1, 0.2, 0.3, 0.5, 0.8]:
        starts.append(_add_noise(fcc_target, sigma=s))
    
    # Start 6-10: Random perturbations of FCC
    for s in range(5):
        angle = 2 * np.pi * s / 5
        rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]])
        starts.append(fcc_target @ rot.T + np.random.normal(0, 0.2, size=fcc_target.shape))
    
    # Start 11-15: Cubic lattice with FCC bias
    side = 4
    cubic_pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                cubic_pts.append([(i - 1.5) * 1.12, (j - 1.5) * 1.12, (k - 1.5) * 1.12])
    cubic_pts.sort(key=lambda p: sum(x**2 for x in p))
    cubic_arr = np.array(cubic_pts[:N], dtype=float)
    for s in [0.0, 0.2, 0.4]:
        starts.append(cubic_arr + np.random.normal(0, s, size=cubic_arr.shape))
    
    # Start 16-20: Pure random
    for s in range(5):
        starts.append(np.random.normal(0, 2.0, size=(N, 3)))
    
    # Process each start
    for idx, start in enumerate(starts):
        if _time_left() < 2.0:
            break
        
        # Ensure minimum distance constraint
        start = np.array(start, dtype=float)
        
        # Phase 2a: Topology-forcing optimization (short, to get near FCC)
        if idx < 10:  # Only do topology forcing for FCC-based starts
            def topo_fun(x):
                p = x.reshape(-1, 3)
                return _topology_loss(p, fcc_target)
            
            def topo_grad(x):
                p = x.reshape(-1, 3)
                return _topology_loss_grad(p, fcc_target).flatten()
            
            try:
                res = minimize(topo_fun, start.flatten(), jac=topo_grad, 
                              method='L-BFGS-B', options={'maxiter': 100, 'gtol': 1e-6})
                start = res.x.reshape(-1, 3)
            except:
                pass
        
        # Phase 2b: LJ energy relaxation
        try:
            relaxed, energy = relax(start)
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
        except:
            pass
        
        # Phase 2c: Basin hopping from the relaxed structure
        if _time_left() < 5.0:
            break
            
        current_pos = start.copy()
        current_energy = energy if 'energy' in dir() else lj_energy(start)
        
        for bh_iter in range(20):
            if _time_left() < 3.0:
                break
            
            # Perturb
            sigma = 0.3 * (1.0 + 0.1 * bh_iter)
            perturbed = current_pos + np.random.normal(0, sigma, size=current_pos.shape)
            
            # Relax
            try:
                relaxed, new_energy = relax(perturbed)
            except:
                continue
            
            # Metropolis accept
            if new_energy < current_energy or np.random.random() < np.exp(-(new_energy - current_energy) / 0.5):
                current_pos = relaxed
                current_energy = new_energy
                
                if new_energy < best_energy:
                    best_energy = new_energy
                    best_pos = relaxed.copy()
        
        if _time_left() < 5.0:
            break
    
    # Final: One more relaxation of best
    if best_pos is not None and _time_left() > 2.0:
        try:
            final_pos, final_energy = relax(best_pos)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pos = final_pos
        except:
            pass
    
    if best_pos is None:
        # Fallback: cubic lattice
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        best_pos = np.array(points[:N], dtype=float) * 1.12
    
    return best_pos

# Initialize _t0
_t0 = time.time()
