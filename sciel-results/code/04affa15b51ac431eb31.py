"""LJ38 cluster global minimum - FCC truncated octahedron construction."""
import numpy as np
import time
from scipy.optimize import minimize

N = 38

def lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)  # avoid division by zero
    r6 = r2 ** 3
    r12 = r6 ** 2
    # Exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    return 4.0 * np.sum((1.0 / r12 - 1.0 / r6) * mask)

def construct():
    t_start = time.time()
    deadline = t_start + 110.0  # safety margin before 120s limit
    
    # FCC truncated octahedron for LJ38
    # Center: 1 atom
    # Inner shell: 12 atoms (cuboctahedron vertices)
    # Outer shell: 24 atoms (truncated octahedron vertices)
    
    # Cuboctahedron vertices: permutations of (±1, ±1, 0)
    inner_base = []
    for x in [1, -1]:
        for y in [1, -1]:
            inner_base.append([x, y, 0])
            inner_base.append([x, 0, y])
            inner_base.append([0, x, y])
    inner_base = np.array(inner_base, dtype=float)
    # Normalize to unit length
    inner_norms = np.linalg.norm(inner_base, axis=1, keepdims=True)
    inner_unit = inner_base / inner_norms  # 12 unit vectors
    
    # Truncated octahedron vertices: permutations of (±1, ±2, ±3)
    outer_base = []
    for a in [1, -1]:
        for b in [2, -2]:
            for c in [3, -3]:
                outer_base.append([a, b, c])
                outer_base.append([a, c, b])
                outer_base.append([b, a, c])
                outer_base.append([b, c, a])
                outer_base.append([c, a, b])
                outer_base.append([c, b, a])
    outer_base = np.array(outer_base, dtype=float)
    outer_norms = np.linalg.norm(outer_base, axis=1, keepdims=True)
    outer_unit = outer_base / outer_norms  # 24 unit vectors
    
    # Parameters: r1 (inner shell radius), r2 (outer shell radius)
    def total_energy(params):
        r1, r2 = params
        center = np.array([0.0, 0.0, 0.0])
        inner = inner_unit * r1
        outer = outer_unit * r2
        pos = np.vstack([center[None, :], inner, outer])
        # Check minimum distance constraint
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        r2_mat = np.sum(diff * diff, axis=2)
        np.fill_diagonal(r2_mat, 1e10)
        min_dist = np.sqrt(np.min(r2_mat))
        if min_dist < 0.5:
            return 1e10
        return lj_energy(pos)
    
    # Initial guess based on known LJ38 structure
    # Known energy -173.928427, let's start near expected values
    best_params = np.array([1.12, 2.24])
    best_energy = total_energy(best_params)
    
    # Try multiple starting points
    start_points = [
        [1.12, 2.24],
        [1.10, 2.20],
        [1.15, 2.30],
        [1.08, 2.16],
        [1.12, 2.15],
        [1.10, 2.30],
    ]
    
    for sp in start_points:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(total_energy, sp, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_energy:
                best_energy = res.fun
                best_params = res.x
        except:
            pass
    
    # Build final position
    r1, r2 = best_params
    center = np.array([0.0, 0.0, 0.0])
    inner = inner_unit * r1
    outer = outer_unit * r2
    pos = np.vstack([center[None, :], inner, outer])
    
    # Verify minimum distance
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2_mat = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2_mat, 1e10)
    min_dist = np.sqrt(np.min(r2_mat))
    
    if min_dist < 0.5:
        # Fallback: use the cubic lattice seed
        side = int(np.ceil(N ** (1 / 3)))
        points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
        points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
        pos = np.array(points[:N], dtype=float) * 1.12
    
    # Final polish with L-BFGS if available and time permits
    if time.time() < deadline - 10:
        try:
            from science_chemistry import relax
            pos_relaxed = relax(pos)
            e_relaxed = lj_energy(pos_relaxed)
            if e_relaxed < best_energy:
                pos = pos_relaxed
                best_energy = e_relaxed
        except:
            pass
    
    return pos
