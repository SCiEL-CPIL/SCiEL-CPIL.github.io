import numpy as np
import time
import math
import random
from scipy.optimize import minimize

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 10s margin for safety against 200s limit

    # 1. Setup Fast Band Solver
    # Use 15x15 = 225 plane waves for the internal search loop.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 8
    for i in range(n_k + 1):
        for j in range(i, n_k + 1): # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points specifically
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)
    
    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def generate_cell_from_params(params):
        """
        Generate a symmetric 32x32 cell from parametric definition.
        params: [r1, n1, r2, n2]
        r1, r2: radii of the two superellipses (scaled to cell size)
        n1, n2: superellipse exponents
        """
        r1, n1, r2, n2 = params
        cell = np.zeros((size, size), dtype=int)
        
        # Centers of the 4 superellipses (one in each quadrant)
        # Cell is 32x32. Center of cell is (15.5, 15.5).
        # Quadrant centers: (8, 8), (23, 8), (8, 23), (23, 23)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        
        # Precompute grid coordinates relative to centers for speed?
        # Or just loop. 32x32 is small.
        y, x = np.mgrid[0:size, 0:size]
        
        # We want to fill pixels where the superellipse condition is met.
        # Superellipse: |dx|^n + |dy|^n <= r^n
        
        for cx, cy in centers:
            dx = np.abs(x - cx)
            dy = np.abs(y - cy)
            
            # Superellipse 1
            if n1 > 0:
                cond1 = (dx / r1)**n1 + (dy / r1)**n1 <= 1.0
                cell[cond1] = 1
            
            # Superellipse 2
            if n2 > 0:
                cond2 = (dx / r2)**n2 + (dy / r2)**n2 <= 1.0
                cell[cond2] = 1
                
        return cell

    def objective(params):
        cell = generate_cell_from_params(params)
        return -compute_score(cell)

    # 2. Multi-start Local Search
    best_score = -np.inf
    best_cell = None
    
    # Define bounds for parameters
    # r: 1 to 10 (pixels)
    # n: 1 to 10 (exponent)
    bounds = [(1.0, 10.0), (1.0, 10.0), (1.0, 10.0), (1.0, 10.0)]
    
    # Initial guesses
    # Start with a known decent configuration: 4 circles (n=2, r=4)
    initial_guesses = [
        [4.0, 2.0, 0.1, 2.0], # Single circle
        [4.0, 2.0, 2.0, 2.0], # Two circles
        [4.0, 4.0, 0.1, 4.0], # Square-ish
        [5.0, 3.0, 3.0, 3.0], # Mix
        [3.0, 1.0, 3.0, 1.0], # Diamond-ish
        [6.0, 2.0, 0.1, 2.0], # Larger circle
        [4.0, 8.0, 0.1, 8.0], # Octagon-ish
        [2.0, 2.0, 2.0, 2.0]  # Small circles
    ]
    
    # Add some random starts
    for _ in range(10):
        initial_guesses.append([
            random.uniform(2, 8),
            random.uniform(1, 8),
            random.uniform(2, 8),
            random.uniform(1, 8)
        ])
        
    for i, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 20.0:
            break
            
        try:
            res = minimize(objective, x0, method='Nelder-Mead', bounds=bounds, options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-4, 'disp': False})
            
            if res.fun < -best_score:
                best_score = -res.fun
                best_cell = generate_cell_from_params(res.x)
        except Exception:
            pass
            
    # 3. Refine with Pixel Search
    if best_cell is not None:
        cell = best_cell.copy()
        current_score = best_score
        
        # Simple pixel refinement
        T = 0.5
        T_min = 0.05
        cooling = 0.995
        
        while time.time() < deadline - 5.0:
            # Pick a random pixel
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            new_cell = cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            
            # Enforce symmetry
            # We will overwrite the array to be symmetric.
            # We define the value of a pixel by the "canonical" pixel in the fundamental domain.
            # Fundamental domain: 0 <= i < 16, 0 <= j < 16, i <= j.
            c_sym = np.zeros_like(new_cell)
            for ii in range(16):
                for jj in range(ii, 16):
                    val = new_cell[ii, jj]
                    c_sym[ii, jj] = val
                    c_sym[jj, ii] = val
                    c_sym[31-ii, 31-jj] = val
                    c_sym[31-jj, 31-ii] = val
                    c_sym[ii, 31-jj] = val
                    c_sym[31-jj, ii] = val
                    c_sym[31-ii, jj] = val
                    c_sym[jj, 31-ii] = val
            
            if np.array_equal(c_sym, cell):
                continue
                
            new_score = compute_score(c_sym)
            
            if new_score > current_score:
                current_score = new_score
                best_score = new_score
                best_cell = c_sym
                cell = c_sym
            else:
                delta = new_score - current_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        cell = c_sym
                        current_score = new_score
                        
            T = max(T_min, T * cooling)
    else:
        # Fallback if parametric search failed
        cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_cell = cell

    return best_cell.tolist()
