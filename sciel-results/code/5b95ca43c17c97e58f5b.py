import numpy as np
import time

N = 100

def compute_score_fast(h):
    """
    Compute the maximum overlap of h with shifted 1-h using FFT.
    Overlap(s) = sum_i h[i] * (1 - h[(i+s) % N]) = N/2 - autocorr(s)
    We want to minimize max Overlap(s), which is N/2 - min autocorr(s).
    """
    H = np.fft.fft(h)
    # Circular autocorrelation
    # acf[k] = sum_j h[j] * h[(j+k) % N]
    acf = np.real(np.fft.ifft(H * np.conj(H)))
    # Due to numerical precision, acf[0] should be sum(h^2).
    # We look for min acf over all shifts.
    # Note: acf[0] is the largest, so min will be at s != 0 usually.
    min_acf = np.min(acf)
    max_overlap = N / 2.0 - min_acf
    return max_overlap

def generate_block_pattern(block_lengths, levels):
    """
    Generate a sequence by repeating a pattern defined by block lengths and levels.
    block_lengths: list of ints, sum must divide N or we tile/truncate.
    levels: list of floats, same length as block_lengths.
    """
    total_len = sum(block_lengths)
    if total_len == 0:
        return np.zeros(N)
    
    # Create one period
    period = np.concatenate([np.full(l, v) for l, v in zip(block_lengths, levels)])
    
    # Tile to reach N
    repeats = N // total_len
    remainder = N % total_len
    h = np.tile(period, repeats)
    if remainder > 0:
        h = np.concatenate([h, period[:remainder]])
    
    return h

def project_to_feasible(h):
    """
    Project h to satisfy sum = N/2 and 0 <= h[i] <= 1.
    """
    h = h.copy()
    h = np.clip(h, 0, 1)
    target = N / 2.0
    diff = target - np.sum(h)
    
    if abs(diff) > 1e-12:
        # Distribute difference uniformly
        h += diff / N
        h = np.clip(h, 0, 1)
        
        # Second pass if needed
        diff2 = target - np.sum(h)
        if abs(diff2) > 1e-12:
            mask = (h > 1e-10) & (h < 1 - 1e-10)
            if np.sum(mask) > 0:
                h[mask] += diff2 / np.sum(mask)
                h = np.clip(h, 0, 1)
    
    # Final exact adjustment to ensure sum is exactly N/2 within float precision
    diff3 = target - np.sum(h)
    if abs(diff3) > 1e-14:
        h += diff3 / N
        h = np.clip(h, 0, 1)
        
    return h

def construct():
    deadline = time.time() + 140  # Leave 10s margin
    
    best_score = float('inf')
    best_h = None
    
    # 1. Uniform solution (Known optimal or near-optimal baseline)
    h_uniform = np.full(N, 0.5)
    score_uniform = compute_score_fast(h_uniform)
    best_score = score_uniform
    best_h = h_uniform.copy()
    
    # 2. Explore Piecewise Constant (Block Interleaved) Structures
    # Strategy: Try various block lengths and levels.
    # Since we suspect uniform is optimal, we check if any structure beats it.
    
    # Candidate 1: Binary Alternating (1, 0, 1, 0...)
    h_bin_alt = np.tile([1.0, 0.0], N // 2)
    h_bin_alt = project_to_feasible(h_bin_alt)
    score_bin_alt = compute_score_fast(h_bin_alt)
    if score_bin_alt < best_score:
        best_score = score_bin_alt
        best_h = h_bin_alt.copy()
        
    # Candidate 2: Binary Blocks (e.g., 10 ones, 10 zeros)
    for block_len in [2, 4, 5, 10, 20, 25]:
        if N % block_len != 0:
            continue
        # Pattern: block_len/2 ones, block_len/2 zeros
        if block_len % 2 != 0:
            continue
        half = block_len // 2
        pattern = np.concatenate([np.ones(half), np.zeros(half)])
        h_block = np.tile(pattern, N // block_len)
        h_block = project_to_feasible(h_block)
        score_block = compute_score_fast(h_block)
        if score_block < best_score:
            best_score = score_block
            best_h = h_block.copy()
            
    # Candidate 3: 3-Level Patterns
    # Example: Levels 0, 0.5, 1.0
    # Try patterns like [0, 0.5, 1, 0.5] or [0, 1, 0.5, 0.5]
    levels_3 = [0.0, 0.5, 1.0]
    # Try a few permutations of block lengths for these levels
    # We need sum = 50.
    # If we use equal blocks, average level must be 0.5.
    # Pattern [0, 1, 0.5, 0.5] avg = 0.5.
    patterns_3 = [
        [0.0, 1.0, 0.5, 0.5],
        [0.0, 0.5, 1.0, 0.5],
        [0.0, 1.0, 0.0, 1.0], # Binary
        [0.5, 0.5, 0.5, 0.5], # Uniform
    ]
    
    for pat in patterns_3:
        if len(pat) == 0: continue
        h_test = np.tile(pat, N // len(pat))
        if len(h_test) < N:
             h_test = np.concatenate([h_test, pat[:N - len(h_test)]])
        h_test = project_to_feasible(h_test)
        score_test = compute_score_fast(h_test)
        if score_test < best_score:
            best_score = score_test
            best_h = h_test.copy()
            
    # 3. Local Refinement on the best candidate found so far
    # Since the landscape is convex-ish around the optimum (uniform),
    # and uniform is likely the global min, we do a simple gradient step
    # or coordinate descent to see if we can shave off any float error or find a better local min.
    
    # Use L-BFGS-B to maximize min autocorrelation (minimize max overlap)
    # Objective: -min(acf)
    # Constraints: sum(h)=50, 0<=h<=1
    
    from scipy.optimize import minimize
    
    def objective(x):
        h_local = project_to_feasible(x)
        H = np.fft.fft(h_local)
        acf = np.real(np.fft.ifft(H * np.conj(H)))
        # We want to maximize min(acf), so minimize -min(acf)
        return -np.min(acf)
        
    # Start from best_h
    if time.time() < deadline:
        try:
            # Add a small constraint for sum to help optimizer? 
            # L-BFGS-B doesn't handle linear equality constraints well directly without penalty.
            # We rely on projection.
            result = minimize(objective, best_h, method='L-BFGS-B', 
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            h_opt = project_to_feasible(result.x)
            score_opt = compute_score_fast(h_opt)
            if score_opt < best_score:
                best_score = score_opt
                best_h = h_opt.copy()
        except Exception:
            pass
            
    # Final check: ensure sum is exactly N/2
    best_h = project_to_feasible(best_h)
    
    return best_h
