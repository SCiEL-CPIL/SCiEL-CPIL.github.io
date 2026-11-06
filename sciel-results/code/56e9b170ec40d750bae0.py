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
    deadline = start_time + 220.0  # Leave 10s margin for safety against 230s limit

    # 1. Setup Fast Band Solver
    # Use 11x11 = 121 plane waves for speed in the search loop
    half_pw = 5 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # Precompute k-points (coarse set for speed)
    k_points = []
    n_k = 6
    for i in range(n_k + 1):
        for j in range(i, n_k + 1): # Ensure kx <= ky
            s = i / n_k
            t = j / n_k
            k_points.append((s * np.pi, t * np.pi))
    # Add high symmetry points
    k_points.append((np.pi, 0))
    k_points.append((np.pi/2, np.pi/2))
    k_points.append((np.pi, np.pi))
    k_points = np.array(k_points)

    def compute_score_fast(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        Uses FFT-based matrix-vector product for speed.
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        # Handle zeros in eps_F (shouldn't happen for binary 0/1 with eps_low=1, eps_high=12)
        # But just in case, clip
        eps_F = np.where(np.abs(eps_F) < 1e-12, 1e-12, eps_F)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            # Pre-compute the FFT of the grid structure? No, kG2 changes.
            # But we can pre-compute the FFT of the "grid" indices? 
            # Actually, the operation is: w = kG2 * v -> FFT -> * inv_eps_F -> IFFT
            # This is a linear operator.
            
            def matvec(v):
                w = kG2 * v
                # Reshape to 2D for FFT
                u = np.fft.fft2(w.reshape(11, 11))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We need the 5th smallest eigenvalue. 
                # eigsh with which='SA' finds smallest algebraic.
                # Since we want bands 4 and 5 (indices 3 and 4), we need at least 5 eigenvalues.
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
        Fundamental domain: 0 <= i <= 15, 0 <= j <= 15, i <= j.
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
                c_sym[31-i, 31-j] = val
                # 4. (31-j, 31-i)
                c_sym[31-j, 31-i] = val
                # 5. (i, 31-j)
                c_sym[i, 31-j] = val
                # 6. (31-j, i)
                c_sym[31-j, i] = val
                # 7. (31-i, j)
                c_sym[31-i, j] = val
                # 8. (j, 31-i)
                c_sym[j, 31-i] = val
        return c_sym

    # 2. Generate Diverse Initial Candidates
    candidates = []
    
    # Candidate 1: Simple 4-rod pattern (standard PC)
    cell1 = np.zeros((size, size), dtype=int)
    # Place rods at (8,8), (23,8), (8,23), (23,23) with radius 3
    for cx, cy in [(8, 8), (23, 8), (8, 23), (23, 23)]:
        for i in range(size):
            for j in range(size):
                if (i-cx)**2 + (j-cy)**2 <= 9:
                    cell1[i, j] = 1
    candidates.append(enforce_full_symmetry(cell1))
    
    # Candidate 2: 4x4 checkerboard of 8x8 blocks
    cell2 = np.zeros((size, size), dtype=int)
    for i in range(4):
        for j in range(4):
            if (i+j) % 2 == 0:
                cell2[i*8:(i+1)*8, j*8:(j+1)*8] = 1
    candidates.append(enforce_full_symmetry(cell2))
    
    # Candidate 3: Moiré pattern (interference of two gratings)
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    x = x / size
    y = y / size
    f1 = np.sign(np.sin(2 * np.pi * 2 * x)) * np.sign(np.sin(2 * np.pi * 2 * y))
    f2 = np.sign(np.sin(2 * np.pi * 3 * x + np.pi/4)) * np.sign(np.sin(2 * np.pi * 3 * y - np.pi/4))
    val = f1 + f2
    cell3 = (val > 0).astype(int)
    candidates.append(enforce_full_symmetry(cell3))
    
    # Candidate 4: Random low-frequency pattern
    cell4 = np.random.randint(0, 2, (size, size))
    # Low-pass filter to make it smooth
    from scipy.ndimage import gaussian_filter
    cell4_smooth = gaussian_filter(cell4.astype(float), sigma=2)
    cell4 = (cell4_smooth > 0.5).astype(int)
    candidates.append(enforce_full_symmetry(cell4))
    
    # Candidate 5: 3x3 grid of rods
    cell5 = np.zeros((size, size), dtype=int)
    for i in range(3):
        for j in range(3):
            cx = 5 + i * 11
            cy = 5 + j * 11
            for ii in range(size):
                for jj in range(size):
                    if (ii-cx)**2 + (jj-cy)**2 <= 4:
                        cell5[ii, jj] = 1
    candidates.append(enforce_full_symmetry(cell5))
    
    # Candidate 6: 2x2 grid of larger rods
    cell6 = np.zeros((size, size), dtype=int)
    for i in range(2):
        for j in range(2):
            cx = 8 + i * 16
            cy = 8 + j * 16
            for ii in range(size):
                for jj in range(size):
                    if (ii-cx)**2 + (jj-cy)**2 <= 16:
                        cell6[ii, jj] = 1
    candidates.append(enforce_full_symmetry(cell6))
    
    # Candidate 7: Ring pattern
    cell7 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            r = math.sqrt((i-15.5)**2 + (j-15.5)**2)
            if 5 < r < 10:
                cell7[i, j] = 1
    candidates.append(enforce_full_symmetry(cell7))
    
    # Candidate 8: Cross pattern
    cell8 = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i % 8 < 2 or j % 8 < 2:
                cell8[i, j] = 1
    candidates.append(enforce_full_symmetry(cell8))

    # 3. Multi-Start Hill Climbing
    best_cell = None
    best_score = -np.inf
    
    # Evaluate all candidates first
    for c in candidates:
        score = compute_score_fast(c)
        if score > best_score:
            best_score = score
            best_cell = c.copy()
    
    # Run hill climbing from each of the top 3 candidates
    top_candidates = []
    for c in candidates:
        score = compute_score_fast(c)
        top_candidates.append((score, c))
    top_candidates.sort(key=lambda x: x[0], reverse=True)
    
    for start_score, start_cell in top_candidates[:3]:
        cell = start_cell.copy()
        T = 0.5
        T_min = 0.01
        cooling = 0.995
        
        steps = 0
        max_steps = 5000
        
        while time.time() < deadline - 10.0 and steps < max_steps:
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
            new_score = compute_score_fast(new_cell)
            
            # Acceptance criterion
            if new_score > start_score:
                start_score = new_score
                cell = new_cell
            else:
                delta = new_score - start_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        cell = new_cell
                else:
                    cell = new_cell
            
            T = max(T_min, T * cooling)
            
            # Check time frequently
            if steps % 100 == 0 and time.time() > deadline - 15.0:
                break
        
        # Update global best
        if start_score > best_score:
            best_score = start_score
            best_cell = cell.copy()
        
        # If time is running out, break
        if time.time() > deadline - 20.0:
            break

    # 4. Final Polish: Run a few more steps from the best cell found
    if best_cell is not None and time.time() < deadline - 15.0:
        cell = best_cell.copy()
        T = 0.1
        T_min = 0.001
        cooling = 0.99
        
        steps = 0
        max_steps = 2000
        
        while time.time() < deadline - 5.0 and steps < max_steps:
            steps += 1
            
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            new_cell = cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            new_cell = enforce_full_symmetry(new_cell)
            
            if np.array_equal(new_cell, cell):
                continue
                
            new_score = compute_score_fast(new_cell)
            
            if new_score > best_score:
                best_score = new_score
                best_cell = new_cell.copy()
                cell = new_cell
            else:
                delta = new_score - best_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        cell = new_cell
            
            T = max(T_min, T * cooling)

    # Ensure the returned cell is a list of lists of ints
    if best_cell is None:
        # Fallback
        best_cell = np.zeros((size, size), dtype=int)
        centers = [(8, 8), (23, 8), (8, 23), (23, 23)]
        for cx, cy in centers:
            best_cell[cx-2:cx+3, cy-2:cy+3] = 1
        best_cell = enforce_full_symmetry(best_cell)

    return best_cell.tolist()
