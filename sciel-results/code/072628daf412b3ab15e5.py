import numpy as np
import time

N = 400
DEADLINE = 120.0
M = 4 * N  # fine grid size
dx = 0.5 / M  # domain length is 0.5

def _compute_c3_fast(signed_step):
    """
    Fast computation of C3 = max_t |autoconv(t)| for signed step function.
    Uses vectorized fine-grid construction and FFT.
    """
    # Vectorized construction of fine grid from step function
    # Each step interval i maps to M//N fine grid points
    f_fine = np.repeat(signed_step.astype(np.float64), M // N)
    
    # Autoconvolution via FFT
    fft_f = np.fft.fft(f_fine, n=2*M)
    autoconv = np.fft.ifft(fft_f * np.conj(fft_f)).real * dx
    
    return np.max(np.abs(autoconv))

def _compute_c3(signed_step):
    """
    Compute the C3 upper bound for a signed step function.
    C3 = max_t |autoconv(t)| where autoconv is the convolution of f with itself.
    """
    return _compute_c3_fast(signed_step)

def _greedy_improve(step, c3, start_time, max_time):
    """
    Perform greedy hill climbing. Returns improved step and c3.
    """
    current_step = step.copy()
    current_c3 = c3
    no_improve = 0
    max_no_improve = 100 # Allow up to 100 full passes without improvement before stopping
    
    while (time.time() - start_time < max_time) and (no_improve < max_no_improve):
        improved = False
        # Random order of indices to break symmetry
        indices = np.random.permutation(N)
        
        for idx in indices:
            if time.time() - start_time > max_time:
                break
            
            old_val = current_step[idx]
            current_step[idx] = -old_val
            
            new_c3 = _compute_c3(current_step)
            
            if new_c3 < current_c3:
                current_c3 = new_c3
                improved = True
                no_improve = 0
            else:
                # Revert
                current_step[idx] = old_val
        
        if not improved:
            no_improve += 1

    return current_step, current_c3

def construct():
    start_time = time.time()
    np.random.seed(42)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. All ones (Seed)
    candidates.append(np.ones(N, dtype=np.int8))
    
    # 2. Alternating
    candidates.append(np.array([1, -1] * (N // 2), dtype=np.int8))
    
    # 3. Random
    candidates.append(np.random.choice([-1, 1], size=N).astype(np.int8))
    
    # 4. Block structure (e.g., 1/4, 3/4 split)
    block1 = np.ones(N // 4, dtype=np.int8)
    block2 = np.ones(3 * N // 4, dtype=np.int8) * -1
    candidates.append(np.concatenate([block1, block2]))
    
    # 5. Another block structure
    block1 = np.ones(N // 2, dtype=np.int8)
    block2 = np.ones(N // 2, dtype=np.int8) * -1
    candidates.append(np.concatenate([block1, block2]))
    
    # 6. Random block
    cuts = np.sort(np.random.choice(N, size=4, replace=False))
    signs = np.random.choice([-1, 1], size=5)
    cand = np.zeros(N, dtype=np.int8)
    prev = 0
    for i, cut in enumerate(cuts):
        cand[prev:cut] = signs[i]
        prev = cut
    cand[prev:] = signs[4]
    candidates.append(cand)

    best_step = None
    best_c3 = float('inf')
    
    # Evaluate and improve each candidate
    # Allocate roughly equal time to each, but prioritize better ones
    # Simple approach: iterate through candidates, spend a fixed chunk of time on each
    # Then use remaining time to polish the best
    
    time_per_candidate = (DEADLINE - 10.0) / len(candidates)
    
    for i, cand in enumerate(candidates):
        current_deadline = start_time + time_per_candidate
        if time.time() > current_deadline:
            continue
            
        c3 = _compute_c3(cand)
        if c3 < best_c3:
            best_c3 = c3
            best_step = cand.copy()
            
        # Run greedy improvement
        improved_step, improved_c3 = _greedy_improve(cand, c3, start_time, current_deadline)
        
        if improved_c3 < best_c3:
            best_c3 = improved_c3
            best_step = improved_step.copy()

    # If we have time, continue improving the best solution found
    if best_step is not None and time.time() - start_time < DEADLINE - 5.0:
        # Run another round of greedy improvement on the best step
        # This might find further improvements if the previous run was cut short
        final_step, final_c3 = _greedy_improve(best_step, best_c3, start_time, DEADLINE - 2.0)
        if final_c3 < best_c3:
            best_c3 = final_c3
            best_step = final_step.copy()
            
        # If still time, try a kick and re-greedy
        if time.time() - start_time < DEADLINE - 2.0:
             kick_step = best_step.copy()
             kick_idx = np.random.randint(N)
             kick_step[kick_idx] = -kick_step[kick_idx]
             kick_c3 = _compute_c3(kick_step)
             
             kicked_step, kicked_c3 = _greedy_improve(kick_step, kick_c3, start_time, DEADLINE - 1.0)
             if kicked_c3 < best_c3:
                 best_c3 = kicked_c3
                 best_step = kicked_step.copy()

    return best_step
