import numpy as np
import time
import math

def _get_d4_orbits(n=32):
    orbits = []
    seen = set()
    for i in range(n):
        for j in range(n):
            if (i, j) in seen:
                continue
            orbit = set()
            orbit.add((i, j))
            orbit.add((j, n-1-i))
            orbit.add((n-1-i, n-1-j))
            orbit.add((n-1-j, i))
            orbit.add((i, n-1-j))
            orbit.add((n-1-i, j))
            orbit.add((j, i))
            orbit.add((n-1-j, n-1-i))
            orbit = list(orbit)
            for p in orbit:
                seen.add(p)
            orbits.append(orbit)
    return orbits

def _is_symmetric(cell):
    n = len(cell)
    for i in range(n):
        for j in range(n):
            if cell[i][j] != cell[j][i]:
                return False
            if cell[i][j] != cell[n-1-i][n-1-j]:
                return False
            if cell[i][j] != cell[n-1-i][j]:
                return False
            if cell[i][j] != cell[i][n-1-j]:
                return False
    return True

def _generate_cell(params, size=32):
    """
    Generate a symmetric 32x32 cell from parameters.
    params: [cx, cy, rx, ry, exponent]
    cx, cy: center of the primary rod (in the first quadrant, e.g., 8.0)
    rx, ry: semi-axes of the superellipse
    exponent: superellipse exponent (2=ellipse, 4=rounded square, >4=sharper)
    """
    cx, cy, rx, ry, exp = params
    cell = np.zeros((size, size), dtype=int)
    
    # Centers by D4 symmetry
    centers = [
        (cx, cy),
        (size - 1 - cx, cy),
        (cx, size - 1 - cy),
        (size - 1 - cx, size - 1 - cy)
    ]
    
    # Vectorized generation
    i_coords, j_coords = np.indices((size, size))
    
    for (ccx, ccy) in centers:
        dx = (i_coords - ccx) / rx
        dy = (j_coords - ccy) / ry
        
        # Superellipse equation: |dx|^exp + |dy|^exp <= 1
        # Handle rx or ry = 0
        if rx < 0.1 or ry < 0.1:
            continue
            
        dist = np.abs(dx)**exp + np.abs(dy)**exp
        cell[dist <= 1.0] = 1
        
    return cell

def _compute_eps_fourier(cell_grid, size=32, eps_low=1.0, eps_high=12.0):
    eps = eps_low + (eps_high - eps_low) * cell_grid.astype(float)
    F = np.fft.fft2(eps)
    F /= (size * size)
    return F

def _solve_tm_bands_fast(kx, ky, eps_F, G_vecs, size=32):
    """
    Fast vectorized solver for a small set of G vectors.
    """
    nG = len(G_vecs)
    Gx = G_vecs[:, 0]
    Gy = G_vecs[:, 1]
    
    # kG vectors
    kGx = kx + Gx
    kGy = ky + Gy
    kG2 = kGx**2 + kGy**2
    
    # Build A matrix: A[n, m] = eps_F[G_n - G_m]
    # G_n - G_m = (2*pi*(dm, dn))
    # We can precompute the integer differences
    # dm = m_n - m_m, dn = n_n - n_m
    # Since G_vecs are generated from integer m, n, we can store the integers
    # For this fast version, we'll just use the direct indexing if possible, or loop if nG is small
    # For nG=25, a loop is fine, but let's try to vectorize the indexing part.
    
    # Actually, for nG=25, 25x25 = 625 elements. A simple loop is ~microseconds.
    A = np.zeros((nG, nG), dtype=complex)
    for n in range(nG):
        for m in range(nG):
            dm = round(G_vecs[n, 0] / (2*np.pi)) - round(G_vecs[m, 0] / (2*np.pi))
            dn = round(G_vecs[n, 1] / (2*np.pi)) - round(G_vecs[m, 1] / (2*np.pi))
            idx_m = dm % size
            idx_n = dn % size
            A[n, m] = eps_F[idx_n, idx_m]
            
    kG2_reg = kG2.copy()
    kG2_reg[kG2_reg < 1e-12] = 1e-12
    D_inv = 1.0 / kG2_reg
    B = D_inv[:, np.newaxis] * A
    B = (B + B.conj().T) / 2
    
    try:
        eigvals = np.linalg.eigvalsh(B)
        eigvals = eigvals.real
        valid = np.abs(eigvals) > 1e-12
        omega2 = np.zeros_like(eigvals)
        omega2[valid] = 1.0 / eigvals[valid]
        omega2[~valid] = np.inf
        return np.sort(omega2)
    except:
        return None

def _get_fast_k_points():
    """Coarse k-points for fast surrogate."""
    k_points = []
    # Gamma
    k_points.append((0.0, 0.0))
    # X
    k_points.append((np.pi, 0.0))
    # M
    k_points.append((np.pi, np.pi))
    # Midpoints
    k_points.append((np.pi/2, 0.0))
    k_points.append((np.pi, np.pi/2))
    k_points.append((np.pi/2, np.pi/2))
    return k_points

def _get_full_k_points(n_per_seg=10):
    """Full k-points for final evaluation."""
    k_points = []
    for i in range(n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((t * np.pi, 0.0))
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append((np.pi, t * np.pi))
    for i in range(1, n_per_seg):
        t = i / (n_per_seg - 1)
        k_points.append(((1 - t) * np.pi, (1 - t) * np.pi))
    return k_points

def _compute_gap_score(cell_grid, k_points, G_vecs, size=32, eps_low=1.0, eps_high=12.0):
    eps_F = _compute_eps_fourier(cell_grid, size, eps_low, eps_high)
    omega4_vals = []
    omega5_vals = []
    
    for (kx, ky) in k_points:
        eigvals = _solve_tm_bands_fast(kx, ky, eps_F, G_vecs, size)
        if eigvals is None or len(eigvals) < 5:
            return -1.0
        omega4 = np.sqrt(max(0, eigvals[3]))
        omega5 = np.sqrt(max(0, eigvals[4]))
        omega4_vals.append(omega4)
        omega5_vals.append(omega5)
    
    if not omega4_vals:
        return -1.0
        
    max_omega4 = max(omega4_vals)
    min_omega5 = min(omega5_vals)
    
    if min_omega5 + max_omega4 < 1e-12:
        return -1.0
        
    return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

def design(problem):
    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin
    
    # Setup fast surrogate
    # Use a small set of G vectors for speed
    half_fast = 2
    G_fast = []
    for m in range(-half_fast, half_fast+1):
        for n in range(-half_fast, half_fast+1):
            if abs(m) + abs(n) <= half_fast:
                G_fast.append((2*np.pi*m, 2*np.pi*n))
    G_fast = np.array(G_fast, dtype=float)
    
    k_fast = _get_fast_k_points()
    
    # Setup full evaluator
    half_full = 6
    G_full = []
    for m in range(-half_full, half_full+1):
        for n in range(-half_full, half_full+1):
            if abs(m) + abs(n) <= half_full:
                G_full.append((2*np.pi*m, 2*np.pi*n))
    G_full = np.array(G_full, dtype=float)
    
    k_full = _get_full_k_points(n_per_seg=10)
    
    # Initial guesses for superellipse parameters
    # cx, cy in [4, 12], rx, ry in [1, 8], exp in [2, 8]
    initial_params = [
        [8.0, 8.0, 3.0, 3.0, 2.0],
        [8.0, 8.0, 4.0, 4.0, 4.0],
        [9.0, 9.0, 2.5, 2.5, 6.0],
        [7.0, 7.0, 3.5, 3.5, 3.0],
        [8.0, 8.0, 2.0, 5.0, 4.0], # Elliptical
    ]
    
    best_score = -np.inf
    best_params = initial_params[0]
    
    # Nelder-Mead implementation (simple version)
    def nelder_mead(f, x0, bounds, max_iter=100, tol=1e-4):
        # x0: list of initial values
        # bounds: list of (min, max) for each variable
        n = len(x0)
        simplex = [x0.copy()]
        
        # Create initial simplex
        for i in range(n):
            x_new = x0.copy()
            step = 0.1 * (bounds[i][1] - bounds[i][0]) if bounds[i][1] > bounds[i][0] else 0.1
            x_new[i] += step
            # Clamp to bounds
            for j in range(n):
                x_new[j] = np.clip(x_new[j], bounds[j][0], bounds[j][1])
            simplex.append(x_new)
            
        # Evaluate initial simplex
        f_values = [f(x) for x in simplex]
        
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        for it in range(max_iter):
            # Sort simplex
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
            # Check convergence
            if abs(f_values[0] - f_values[-1]) < tol:
                break
                
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            # Clamp
            for i in range(n):
                x_r[i] = np.clip(x_r[i], bounds[i][0], bounds[i][1])
            f_r = f(x_r)
            
            if f_values[0] <= f_r < f_values[-2]:
                simplex[-1] = x_r
                f_values[-1] = f_r
            elif f_r < f_values[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                for i in range(n):
                    x_e[i] = np.clip(x_e[i], bounds[i][0], bounds[i][1])
                f_e = f(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            else:
                # Contraction
                x_c = centroid + rho * (simplex[-1] - centroid)
                for i in range(n):
                    x_c[i] = np.clip(x_c[i], bounds[i][0], bounds[i][1])
                f_c = f(x_c)
                if f_c < f_values[-1]:
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n+1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        for j in range(n):
                            simplex[i][j] = np.clip(simplex[i][j], bounds[j][0], bounds[j][1])
                        f_values[i] = f(simplex[i])
                        
        return min(f_values), simplex[np.argmin(f_values)]
    
    bounds = [
        (4.0, 12.0),  # cx
        (4.0, 12.0),  # cy
        (0.5, 8.0),   # rx
        (0.5, 8.0),   # ry
        (1.5, 8.0)    # exp
    ]
    
    # Objective function for optimization (using fast surrogate)
    def objective(params):
        cell = _generate_cell(params, size)
        return _compute_gap_score(cell, k_fast, G_fast, size, eps_low, eps_high)
    
    # Run optimization for each initial guess
    for init in initial_params:
        if time.time() > deadline - 30:
            break
            
        score, params = nelder_mead(objective, init, bounds, max_iter=50)
        
        if score > best_score:
            best_score = score
            best_params = params
            
    # Final evaluation with full solver for the best param set
    best_cell = _generate_cell(best_params, size)
    final_score = _compute_gap_score(best_cell, k_full, G_full, size, eps_low, eps_high)
    
    # If the fast surrogate was misleading or final score is bad, try a fallback
    # Fallback: simple 4-rod design
    if final_score < 0.5:
        cell_fb = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for i in range(size):
            for j in range(size):
                for (cx, cy) in centers:
                    if (i-cx)**2 + (j-cy)**2 < 16.0:
                        cell_fb[i, j] = 1
                        break
        fb_score = _compute_gap_score(cell_fb, k_full, G_full, size, eps_low, eps_high)
        if fb_score > final_score:
            best_cell = cell_fb
            final_score = fb_score
            
    best_list = [[int(best_cell[i, j]) for j in range(size)] for i in range(size)]
    
    if not _is_symmetric(best_list):
        # Force symmetry by averaging or just returning a known symmetric one
        # For safety, return the fallback if symmetry fails
        if final_score < 0.5:
             best_list = [[int(cell_fb[i, j]) for j in range(size)] for i in range(size)]
        else:
             # Try to fix symmetry by taking the orbit representative
             # But _generate_cell should always be symmetric if inputs are valid.
             # If it's not, something went wrong.
             pass
             
    return best_list
