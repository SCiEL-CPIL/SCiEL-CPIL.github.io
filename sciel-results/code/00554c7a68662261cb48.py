import numpy as np
import time
import math
import random
from scipy.sparse.linalg import eigsh
from scipy.sparse import LinearOperator

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 190.0  # Leave 10s margin for safety against 200s limit

    # 1. Setup Fast Band Solver
    # We use 15x15 = 225 plane waves for the internal search loop to be fast.
    # This matches the resolution of the current seed (half_pw=7).
    # Note: The evaluator uses 289 (17x17), but 225 is sufficient for ranking.
    half_pw = 7
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    grid_side = int(np.sqrt(nG))  # 15
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed.
    # Irreducible BZ for square lattice: 0 <= kx <= ky <= pi
    k_points = []
    n_k = 8
    for i in range(n_k + 1):
        for j in range(i, n_k + 1):  # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points specifically
    k_points.append((np.pi, 0))
    k_points.append((np.pi / 2, np.pi / 2))
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
            kG2 = kG[:, 0] ** 2 + kG[:, 1] ** 2
            
            def matvec(v):
                w = kG2 * v
                u = np.fft.fft2(w.reshape(grid_side, grid_side))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
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

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        We define the value of a pixel by the "canonical" pixel in the fundamental domain.
        Fundamental domain: 0 <= i < 16, 0 <= j < 16, i <= j.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                # 1. (i, j)
                c_sym[i, j] = val
                # 2. (j, i)
                c_sym[j, i] = val
                # 3. (31-i, 31-j)
                c_sym[31 - i, 31 - j] = val
                # 4. (31-j, 31-i)
                c_sym[31 - j, 31 - i] = val
                # 5. (i, 31-j)
                c_sym[i, 31 - j] = val
                # 6. (31-j, i)
                c_sym[31 - j, i] = val
                # 7. (31-i, j)
                c_sym[31 - i, j] = val
                # 8. (j, 31-i)
                c_sym[j, 31 - i] = val
        return c_sym

    # 2. Initialize Cell
    # Start with a configuration known to be decent: 4 squares or circles.
    # We'll start with 4 squares at (8,8), (23,8), (8,23), (23,23) with size 5x5
    cell = np.zeros((size, size), dtype=int)
    centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
    for cx, cy in centers:
        cell[cx - 2:cx + 3, cy - 2:cy + 3] = 1
    cell = enforce_full_symmetry(cell)

    best_score = compute_score(cell)
    best_cell = cell.copy()

    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    steps = 0
    max_steps = 100000  # Safety cap
    
    while time.time() < deadline - 5.0 and steps < max_steps:
        steps += 1
        
        # Pick a random pixel
        i = random.randint(0, size - 1)
        j = random.randint(0, size - 1)
        
        # Flip it
        new_cell = cell.copy()
        new_cell[i, j] = 1 - new_cell[i, j]
        
        # Enforce symmetry
        new_cell = enforce_full_symmetry(new_cell)
        
        # If no change, skip
        if np.array_equal(new_cell, cell):
            continue
            
        # Calculate score
        new_score = compute_score(new_cell)
        
        # Acceptance criterion
        if new_score > best_score:
            best_score = new_score
            best_cell = new_cell.copy()
            cell = new_cell
            # Reset temperature? No, just keep cooling
        else:
            delta = new_score - best_score
            if delta < 0:
                p = math.exp(delta / T)
                if random.random() < p:
                    cell = new_cell
            
        T = max(T_min, T * cooling)
        
        # Check time frequently
        if steps % 100 == 0 and time.time() > deadline - 10.0:
            break

    # Final check: if best_cell is invalid or score is bad, try a few random restarts?
    # No, we have time. Let's just return the best found.
    
    # Ensure the returned cell is a list of lists of ints
    return best_cell.tolist()
