# EVOLVE-BLOCK-START
import numpy as np
import time
import math

N = 100

def _compute_max_overlap(h):
    """Compute the maximum overlap of h with shifted 1-h."""
    n = len(h)
    if n == 0:
        return 0.0
    
    H = np.fft.fft(h)
    autocorr = np.real(np.fft.ifft(H * np.conj(H)))
    sum_h = np.sum(h)
    # We want max overlap for k=1..n-1
    # overlap(k) = sum_h - autocorr(k)
    overlaps = sum_h - autocorr[1:]
    return np.max(overlaps)

def _sa_optimize(h_init, time_limit, seed=42):
    """Simulated Annealing to minimize max overlap."""
    n = len(h_init)
    h = h_init.copy()
    rng = np.random.RandomState(seed)
    
    current_energy = _compute_max_overlap(h)
    best_energy = current_energy
    best_h = h.copy()
    
    start_time = time.time()
    T0 = 1.0
    Tend = 1e-5
    
    # Precompute indices for speed? No, h changes.
    # But we can maintain lists of ones and zeros.
    ones = np.where(h == 1.0)[0]
    zeros = np.where(h == 0.0)[0]
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        # Linear cooling in log T? Or exponential?
        # Let's use exponential decay of T based on elapsed time fraction
        # T = T0 * (Tend/T0)^(elapsed/time_limit)
        if time_limit > 0:
            T = T0 * math.pow(Tend / T0, elapsed / time_limit)
        else:
            T = T0
            
        if T < 1e-10:
            break
            
        # Pick random i from ones, j from zeros
        if len(ones) == 0 or len(zeros) == 0:
            break
            
        i_idx = rng.randint(len(ones))
        j_idx = rng.randint(len(zeros))
        i = ones[i_idx]
        j = zeros[j_idx]
        
        # Swap
        h[i] = 0.0
        h[j] = 1.0
        
        # Update indices lists
        # Remove i from ones, add to zeros
        # Remove j from zeros, add to ones
        # This is O(N) if we do it naively. 
        # Given N=100, O(N) is fine.
        # ones[i_idx] = ones[-1]; ones = ones[:-1] ... 
        # Let's just recompute or use a set? 
        # Recomputing np.where every step is O(N log N). 
        # 100 log 100 ~ 700. 
        # If we do 10k steps/sec, that's 7M ops/sec. Fine.
        
        new_energy = _compute_max_overlap(h)
        
        delta_E = new_energy - current_energy
        accept = False
        if delta_E < 0:
            accept = True
        elif T > 0:
            # Avoid overflow in exp
            if delta_E / T < 100: 
                if rng.rand() < math.exp(-delta_E / T):
                    accept = True
        
        if accept:
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_h = h.copy()
                
            # Update lists
            # i was 1, now 0. j was 0, now 1.
            # We need to update the arrays `ones` and `zeros`
            # This is tricky with numpy arrays without O(N) copy.
            # Let's just recompute `ones` and `zeros` for simplicity and correctness.
            # It might be slower but N=100 is small.
            ones = np.where(h == 1.0)[0]
            zeros = np.where(h == 0.0)[0]
        else:
            # Revert
            h[i] = 1.0
            h[j] = 0.0
            # No need to update lists if we revert, they are still valid for h?
            # Yes, h is back to original state.
            
    return best_h, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n_ones = N // 2
    rng_global = np.random.RandomState(42)
    
    # Initial guess
    h_init = np.zeros(N)
    h_init[::2] = 1.0
    if np.sum(h_init) != n_ones:
        idx = rng_global.permutation(N)
        h_init[idx[:n_ones]] = 1.0
        
    best_h = h_init.copy()
    best_score = _compute_max_overlap(best_h)
    
    # Run SA
    sa_time = min(deadline - time.time() - 5, 100)
    if sa_time > 1:
        h_sa, score_sa = _sa_optimize(best_h, sa_time, seed=42)
        if score_sa < best_score:
            best_score = score_sa
            best_h = h_sa.copy()
            
    # Final Validation
    best_h = (best_h > 0.5).astype(float)
    current_sum = np.sum(best_h)
    if current_sum != n_ones:
        # Fix sum
        if current_sum < n_ones:
            zeros = np.where(best_h == 0)[0]
            need = n_ones - int(current_sum)
            if need > 0:
                to_add = rng_global.choice(zeros, need, replace=False)
                best_h[to_add] = 1.0
        else:
            ones = np.where(best_h == 1)[0]
            need = int(current_sum) - n_ones
            if need > 0:
                to_remove = rng_global.choice(ones, need, replace=False)
                best_h[to_remove] = 0.0
                
    return best_h
# EVOLVE-BLOCK-END
