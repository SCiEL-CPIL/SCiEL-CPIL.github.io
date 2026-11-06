import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 10s margin for safety against 200s limit

    # 1. Setup Fast Band Solver for internal evaluation
    # We use a reduced set of plane waves (15x15 = 225) for the internal search loop.
    # This is faster than the evaluator's 289 (17x17) but captures the essential physics 
    # for ranking candidate designs.
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

    # 2. Constructive Design Strategy
    # The gap between TM bands 4 and 5 is driven by the coupling between plane waves 
    # separated by reciprocal lattice vectors. For a square lattice, the critical 
    # Fourier components of the dielectric function epsilon(r) are those with 
    # G = (2, 0), (0, 2), (1, 1), (1, -1), etc.
    # Specifically, the (2,0) component couples states near the X point, and 
    # (1,1) couples states near the M point.
    
    # We will construct the cell by defining a fundamental domain (1/8th of the cell) 
    # and optimizing the binary values within this domain to maximize the target 
    # Fourier coefficients.
    
    # Fundamental domain indices:
    # i in [0, 15], j in [0, 15], and i <= j.
    # This represents the triangle 0 <= i <= j <= 15.
    
    def get_fundamental_indices():
        indices = []
        for i in range(16):
            for j in range(i, 16):
                indices.append((i, j))
        return indices
    
    f_indices = get_fundamental_indices()
    n_f = len(f_indices)
    
    def construct_cell_from_fundamental(vals):
        """
        vals is a binary array of length n_f corresponding to f_indices.
        Returns a 32x32 cell.
        """
        c = np.zeros((size, size), dtype=int)
        for k, (i, j) in enumerate(f_indices):
            val = vals[k]
            # Symmetry group C4v about center (15.5, 15.5)
            # Positions:
            # 1. (i, j)
            # 2. (j, i)
            # 3. (31-i, 31-j)
            # 4. (31-j, 31-i)
            # 5. (i, 31-j)
            # 6. (31-j, i)
            # 7. (31-i, j)
            # 8. (j, 31-i)
            
            c[i, j] = val
            c[j, i] = val
            c[31-i, 31-j] = val
            c[31-j, 31-i] = val
            c[i, 31-j] = val
            c[31-j, i] = val
            c[31-i, j] = val
            c[j, 31-i] = val
        return c

    def fourier_objective(vals):
        """
        Objective function: Maximize the magnitude of the (2,0) and (2,2) Fourier 
        components of the dielectric function.
        """
        cell = construct_cell_from_fundamental(vals)
        eps_grid = eps_low + (eps_high - eps_low) * cell
        eps_F = np.fft.fft2(eps_grid)
        
        # Extract coefficients
        # G=(2,0) is at index [2, 0] and [-2, 0] (which is [30, 0])
        # G=(2,2) is at index [2, 2] and [-2, -2] (which is [30, 30])
        c20 = np.abs(eps_F[2, 0]) + np.abs(eps_F[30, 0])
        c22 = np.abs(eps_F[2, 2]) + np.abs(eps_F[30, 30])
        
        # We want to maximize these. 
        # Normalize to prevent scale issues? The gap ratio is scale invariant in a way, 
        # but the absolute gap depends on these.
        # Let's just sum them.
        return c20 + c22

    # 3. Optimization Loop
    # We use a local search on the fundamental domain (120 pixels).
    # This is much smaller than the full 32x32 (1024 pixels), making the search 
    # significantly more efficient.
    
    # Initialize with a random configuration
    current_vals = np.random.randint(0, 2, n_f)
    best_vals = current_vals.copy()
    best_obj = fourier_objective(best_vals)
    
    # Also compute the actual band gap score for the best Fourier-optimized cell
    # to ensure we are tracking the true metric.
    best_cell = construct_cell_from_fundamental(best_vals)
    best_score = compute_score(best_cell)
    
    # Local search on the fundamental domain
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    max_steps = 50000 # Safety cap
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Pick a random index in the fundamental domain
        idx = random.randint(0, n_f - 1)
        
        # Flip it
        new_vals = current_vals.copy()
        new_vals[idx] = 1 - new_vals[idx]
        
        # Calculate new objective (Fourier)
        new_obj = fourier_objective(new_vals)
        
        # Acceptance criterion for the Fourier objective
        # We use the Fourier objective as a proxy for speed, but we must 
        # occasionally verify with the full band solver to avoid getting 
        # stuck in a Fourier-optimal but band-suboptimal region.
        
        if new_obj > best_obj:
            best_obj = new_obj
            best_vals = new_vals.copy()
            current_vals = new_vals
            
            # Update best cell and score periodically or on improvement
            # To save time, we only recompute the full score every N steps 
            # or if the Fourier objective improves significantly.
            # Let's recompute every 100 steps or if obj improves by > 1%.
            if steps % 100 == 0 or new_obj > best_obj * 1.01:
                candidate_cell = construct_cell_from_fundamental(best_vals)
                candidate_score = compute_score(candidate_cell)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_cell = candidate_cell
        else:
            delta = new_obj - best_obj
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    current_vals = new_vals
            
        T = max(T_min, T * cooling)
        
        # Check time frequently
        if steps % 100 == 0 and time.time() > deadline - 10.0:
            break

    # Final verification and return
    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
