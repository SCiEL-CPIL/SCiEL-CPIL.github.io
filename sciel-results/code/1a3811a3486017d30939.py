import numpy as np
import time
import random

def design(problem):
    """
    Designs a 32x32 unit cell using Symmetry-Adapted Simulated Annealing
    on the discrete pixel space.
    """
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin

    size = problem["pixels"]
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Fundamental domain indices for C4v symmetry on 32x32
    # i in [0, 15], j in [i, 15]
    fund_indices = []
    for i in range(16):
        for j in range(i, 16):
            fund_indices.append((i, j))
    
    def expand_cell(fund_values):
        """Expand fundamental domain values to full 32x32 cell."""
        cell = np.zeros((size, size), dtype=int)
        for idx, (i, j) in enumerate(fund_indices):
            val = fund_values[idx]
            # Apply symmetries
            # 1. Identity
            cell[i, j] = val
            # 2. Reflection x->-x (j -> 31-j)
            cell[i, 31-j] = val
            # 3. Reflection y->-y (i -> 31-i)
            cell[31-i, j] = val
            # 4. Rotation 180 (i->31-i, j->31-j)
            cell[31-i, 31-j] = val
            # 5. Reflection y->x (i<->j)
            cell[j, i] = val
            # 6. Reflection y->-x (i<->31-j)
            cell[j, 31-i] = val
            # 7. Reflection x->-y (31-i <-> j)
            cell[31-i, j] = val # Already set by 3
            # 8. Rotation 90? (i->j, j->31-i) is covered by 6?
            # Let's verify the 8 symmetries of C4v on indices 0..31
            # (i, j)
            # (31-i, 31-j)
            # (j, i)
            # (31-j, 31-i)
            # (i, 31-j)
            # (31-i, j)
            # (31-j, i)
            # (j, 31-i)
            
            # My loop above:
            # cell[i, j]
            # cell[i, 31-j]
            # cell[31-i, j]
            # cell[31-i, 31-j]
            # cell[j, i]
            # cell[j, 31-i]
            # cell[31-j, i] -> Is this set?
            # cell[31-j, 31-i] -> Is this set?
            
            # Let's just assign all 8 explicitly to be safe.
            cell[i, j] = val
            cell[i, 31-j] = val
            cell[31-i, j] = val
            cell[31-i, 31-j] = val
            cell[j, i] = val
            cell[j, 31-i] = val
            cell[31-j, i] = val
            cell[31-j, 31-i] = val
            
        return cell

    def compute_fft_inv_eps(cell):
        """Compute 2D FFT of 1/epsilon."""
        eps = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps
        fft_2d = np.fft.fft2(inv_eps) / (size * size)
        return fft_2d

    def solve_bands_fast(cell, n_pw=7):
        """
        Fast TM band solver using plane-wave expansion.
        Returns gap-midgap ratio for bands 4 and 5.
        """
        fft_2d = compute_fft_inv_eps(cell)
        
        def get_coeff(m, n):
            m_idx = m % size
            n_idx = n % size
            return fft_2d[m_idx, n_idx]
        
        # K-points in irreducible BZ
        k_pts = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.25, 0.0),
            (0.25, 0.25),
            (0.5, 0.25),
            (0.125, 0.0),
            (0.125, 0.125),
        ]
        
        # Build list of plane waves within circular cutoff
        g_pairs = []
        for m in range(-n_pw, n_pw + 1):
            for n in range(-n_pw, n_pw + 1):
                if m * m + n * n <= n_pw * n_pw:
                    g_pairs.append((m, n))
        
        n_pw_actual = len(g_pairs)
        
        max_omega4 = -1e10
        min_omega5 = 1e10
        
        for kx, ky in k_pts:
            M = np.zeros((n_pw_actual, n_pw_actual), dtype=complex)
            
            for i, (mi, ni) in enumerate(g_pairs):
                for j, (mj, nj) in enumerate(g_pairs):
                    diff_m = mi - mj
                    diff_n = ni - nj
                    eps_val = get_coeff(diff_m, diff_n)
                    kx_g = kx - mj
                    ky_g = ky - nj
                    M[i, j] = eps_val * (kx_g * kx_g + ky_g * ky_g)
            
            H = (M + M.conj().T) / 2.0
            
            try:
                eigenvalues = np.linalg.eigvalsh(H)
                eigenvalues = np.sort(np.real(eigenvalues))
                if len(eigenvalues) >= 5:
                    omega4 = eigenvalues[3]
                    omega5 = eigenvalues[4]
                    if omega4 > max_omega4:
                        max_omega4 = omega4
                    if omega5 < min_omega5:
                        min_omega5 = omega5
            except:
                pass
        
        if max_omega4 <= 0 or min_omega5 <= 0 or min_omega5 <= max_omega4:
            return -1.0
        
        gap = min_omega5 - max_omega4
        midgap = (min_omega5 + max_omega4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap

    # Initialize with a random symmetric cell or a known good shape
    # Let's start with a "hole" in the center or a "dot"
    # Try a few initial seeds
    best_score = -1e10
    best_fund = None
    
    # Seed 1: Random
    random_fund = [random.randint(0, 1) for _ in fund_indices]
    cell = expand_cell(random_fund)
    score = solve_bands_fast(cell)
    if score > best_score:
        best_score = score
        best_fund = random_fund.copy()
        
    # Seed 2: Center dot
    fund2 = [0 for _ in fund_indices]
    for idx, (i, j) in enumerate(fund_indices):
        # Distance from center (15.5, 15.5)
        dist = np.sqrt((i - 15.5)**2 + (j - 15.5)**2)
        if dist < 5:
            fund2[idx] = 1
    cell = expand_cell(fund2)
    score = solve_bands_fast(cell)
    if score > best_score:
        best_score = score
        best_fund = fund2.copy()
        
    # Seed 3: Superellipse-like (from previous best logic)
    fund3 = [0 for _ in fund_indices]
    for idx, (i, j) in enumerate(fund_indices):
        # Check if inside superellipse centered at (8,8) etc?
        # The fundamental domain is top-left.
        # Centers in full cell: (8,8), (24,8), (8,24), (24,24)
        # In fund domain (0-15), only (8,8) is fully represented?
        # (8,8) is in fund.
        dx = j - 8
        dy = i - 8
        if dx*dx + dy*dy < 16: # Radius 4
            fund3[idx] = 1
    cell = expand_cell(fund3)
    score = solve_bands_fast(cell)
    if score > best_score:
        best_score = score
        best_fund = fund3.copy()

    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.995
    n_iter_max = 5000
    
    current_fund = best_fund
    current_score = best_score
    
    for it in range(n_iter_max):
        if time.time() > deadline:
            break
        if T < T_min:
            break
            
        # Pick a random index in fundamental domain
        idx = random.randint(0, len(fund_indices) - 1)
        
        # Flip the pixel
        new_fund = current_fund.copy()
        new_fund[idx] = 1 - new_fund[idx]
        
        # Expand and evaluate
        new_cell = expand_cell(new_fund)
        new_score = solve_bands_fast(new_cell)
        
        delta = new_score - current_score
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_fund = new_fund
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_fund = current_fund.copy()
                
        T *= cooling_rate
        
        # Reheat occasionally if stuck?
        if it % 500 == 0 and T < 0.1:
            T = 0.5

    # Final refinement with higher precision PW if time permits
    if time.time() < deadline - 30:
        # Try a few more random restarts or local searches around best
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            # Perturb best_fund
            perturbed = best_fund.copy()
            for _ in range(10):
                idx = random.randint(0, len(fund_indices) - 1)
                perturbed[idx] = 1 - perturbed[idx]
            cell = expand_cell(perturbed)
            score = solve_bands_fast(cell)
            if score > best_score:
                best_score = score
                best_fund = perturbed.copy()

    final_cell = expand_cell(best_fund)
    
    # Convert to nested list
    result = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(int(final_cell[i, j]))
        result.append(row)
    
    return result
