import time
import numpy as np
from numpy.linalg import eigh

def _compute_gap_fast(cell, n_k=8, pw_radius=10):
    """
    Fast surrogate evaluator using a coarse plane-wave expansion.
    Returns the gap-midgap ratio.
    """
    N = len(cell)
    inv_eps = 1.0 / (1.0 + 11.0 * np.array(cell, dtype=float))
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    dqx = Gx_diff.astype(int)
    dqy = Gy_diff.astype(int)
    inv_eps_mat = fft_inv[dqx % N, dqy % N]
    
    # Key points in the irreducible Brillouin zone (square lattice)
    # Gamma (0,0), M (0.5, 0), K (0.5, 0.5)
    k_points = [
        (0.0, 0.0), 
        (0.25, 0.0), 
        (0.5, 0.0), 
        (0.5, 0.25),
        (0.5, 0.5), 
        (0.4, 0.4),
        (0.3, 0.3),
        (0.2, 0.2)
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    
    for k in k_points:
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals = eigh(M, eigvals_only=True)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                max_w4 = max(max_w4, eigvals[3])
                min_w5 = min(min_w5, eigvals[4])
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0
    return 2 * (min_w5 - max_w4) / (min_w5 + max_w4)

def _fourier_to_cell(coeffs, N=32, offset=0.0, scale=1.0):
    """
    Generate a 32x32 binary cell from C4v-symmetric Fourier coefficients.
    coeffs: list of amplitudes for the first few modes.
    """
    x, y = np.meshgrid(np.arange(N), np.arange(N))
    x = (x + 0.5) / N - 0.5
    y = (y + 0.5) / N - 0.5
    
    profile = np.zeros((N, N))
    
    # Mode 0: Uniform background
    if len(coeffs) > 0:
        profile += coeffs[0]
        
    # Mode 1: 4-fold symmetric (x^2 - y^2) -> cos(2pi(2x)) + cos(2pi(2y))
    if len(coeffs) > 1:
        profile += coeffs[1] * (np.cos(2 * np.pi * 2 * x) + np.cos(2 * np.pi * 2 * y))
        
    # Mode 2: 4-fold symmetric (xy) -> cos(2pi(x+y)) + cos(2pi(x-y))
    if len(coeffs) > 2:
        profile += coeffs[2] * (np.cos(2 * np.pi * (x + y)) + np.cos(2 * np.pi * (x - y)))
        
    # Mode 3: 4-fold symmetric (x^4 - 6x^2y^2 + y^4) approx or next harmonic
    # Let's use 4th order: cos(4pi x) + cos(4pi y)
    if len(coeffs) > 3:
        profile += coeffs[3] * (np.cos(2 * np.pi * 4 * x) + np.cos(2 * np.pi * 4 * y))
        
    # Mode 4: 8-fold symmetric (mixed)
    if len(coeffs) > 4:
        profile += coeffs[4] * (np.cos(2 * np.pi * (2*x + 2*y)) + np.cos(2 * np.pi * (2*x - 2*y)))

    profile = profile * scale + offset
    cell = (profile > 0.5).astype(int)
    return cell.tolist()

def _random_coefficients():
    """Generate random C4v Fourier coefficients."""
    return [
        np.random.uniform(0.2, 0.6),      # Background
        np.random.uniform(-0.3, 0.3),     # 4-fold x/y
        np.random.uniform(-0.3, 0.3),     # 4-fold xy
        np.random.uniform(-0.2, 0.2),     # 4-fold higher
        np.random.uniform(-0.1, 0.1)      # 8-fold
    ]

def _mutate_coefficients(coeffs, rate=0.1, sigma=0.1):
    """Perturb coefficients."""
    new_coeffs = coeffs.copy()
    for i in range(len(new_coeffs)):
        if np.random.random() < rate:
            new_coeffs[i] += np.random.normal(0, sigma)
    return new_coeffs

def design(problem):
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin for 200s limit + overhead
    
    # Initial set of candidates
    candidates = []
    
    # 1. Geometric seeds
    # Rod
    for r in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35]:
        cell = [[0]*32 for _ in range(32)]
        for i in range(32):
            for j in range(32):
                x = (i + 0.5) / 32.0 - 0.5
                y = (j + 0.5) / 32.0 - 0.5
                if x*x + y*y < r*r:
                    cell[i][j] = 1
        candidates.append(cell)
        
    # Square
    for r in [0.1, 0.2, 0.3]:
        cell = [[0]*32 for _ in range(32)]
        for i in range(32):
            for j in range(32):
                x = (i + 0.5) / 32.0 - 0.5
                y = (j + 0.5) / 32.0 - 0.5
                if abs(x) < r and abs(y) < r:
                    cell[i][j] = 1
        candidates.append(cell)
        
    # 2. Fourier seeds
    for _ in range(10):
        coeffs = _random_coefficients()
        candidates.append(_fourier_to_cell(coeffs))
        
    # Evaluate initial set
    best_cell = None
    best_score = -10.0
    best_coeffs = None
    
    for cell in candidates:
        if time.time() > deadline - 60: break
        try:
            score = _compute_gap_fast(cell, n_k=6, pw_radius=8)
            if score > best_score:
                best_score = score
                best_cell = cell
        except:
            pass
            
    # If we didn't find a good cell yet, or if we have time, do local search on Fourier space
    # We need to map the best cell back to Fourier space? No, that's hard.
    # Instead, we run a separate optimization loop on Fourier coefficients.
    
    # Phase 2: Fourier Optimization
    best_fourier_score = -10.0
    best_fourier_cell = None
    best_fourier_coeffs = None
    
    # Try a few random starting points for Fourier search
    search_points = [_random_coefficients() for _ in range(5)]
    # Also add a "known good" structure approximated by Fourier
    # A cross shape or a ring can be approximated
    search_points.append([0.4, 0.2, 0.0, -0.1, 0.0]) # Rod-like
    search_points.append([0.4, -0.2, 0.0, 0.1, 0.0])  # Diamond-like
    search_points.append([0.6, 0.0, 0.2, 0.0, 0.0])   # Square-ish
    
    for init_coeffs in search_points:
        if time.time() > deadline - 30: break
        
        current_coeffs = init_coeffs
        current_cell = _fourier_to_cell(current_coeffs)
        try:
            current_score = _compute_gap_fast(current_cell, n_k=5, pw_radius=7)
        except:
            current_score = -10.0
            
        if current_score > best_fourier_score:
            best_fourier_score = current_score
            best_fourier_cell = current_cell
            best_fourier_coeffs = current_coeffs
            
        # Local search (Hill Climbing)
        no_improve = 0
        max_no_improve = 20
        step_size = 0.1
        
        while time.time() < deadline - 20 and no_improve < max_no_improve:
            mutated_coeffs = _mutate_coefficients(current_coeffs, rate=1.0, sigma=step_size)
            mutated_cell = _fourier_to_cell(mutated_coeffs)
            
            try:
                new_score = _compute_gap_fast(mutated_cell, n_k=5, pw_radius=7)
            except:
                new_score = -10.0
                
            if new_score > current_score:
                current_score = new_score
                current_coeffs = mutated_coeffs
                current_cell = mutated_cell
                no_improve = 0
                # Decrease step size slightly
                step_size *= 0.9
            else:
                no_improve += 1
                # Increase step size if stuck for a while? Or just reset?
                if no_improve > 5:
                    step_size *= 1.2 # Expand search
                    
            if current_score > best_fourier_score:
                best_fourier_score = current_score
                best_fourier_cell = current_cell
                best_fourier_coeffs = current_coeffs
                step_size = max(0.02, step_size * 0.8) # Refine

    # Compare Phase 1 (Geometric/Pixel) and Phase 2 (Fourier)
    if best_fourier_cell is not None and (best_cell is None or best_fourier_score > best_score):
        best_cell = best_fourier_cell
        best_score = best_fourier_score

    # Final Fallback
    if best_cell is None:
        best_cell = [[0]*32 for _ in range(32)]
        for i in range(32):
            for j in range(32):
                x = (i + 0.5) / 32.0 - 0.5
                y = (j + 0.5) / 32.0 - 0.5
                if x*x + y*y < 0.2*0.2:
                    best_cell[i][j] = 1

    return best_cell
