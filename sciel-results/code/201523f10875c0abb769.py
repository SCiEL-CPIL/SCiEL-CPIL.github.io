"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses a spectral basis representation to reduce dimensionality, 
combined with Differential Evolution for global search and 
Nelder-Mead for local refinement, followed by a binary polish.
"""
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 100
_T_START = time.time()
_DEADLINE = _T_START + 145

def _linear_overlap_values(h):
    """Compute C_k = (2/N)*(N/2 - r_k) where r_k = sum_{i=k}^{N-1} h_i h_{i-k}."""
    n = len(h)
    n_fft = 1
    while n_fft < 2 * n:
        n_fft *= 2
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _project_sum(h, target_sum):
    """
    Project h onto {x : 0 <= x_i <= 1, sum(x) = target_sum}.
    Uses bisection on a global shift.
    """
    h = np.asarray(h, dtype=np.float64)
    current_sum = np.sum(h)
    if abs(current_sum - target_sum) < 1e-12:
        return np.clip(h, 0.0, 1.0)
    
    # If we just shift, we might hit bounds.
    # Bisection on shift delta: sum(clip(h + delta, 0, 1)) = target_sum
    low = -1.0
    high = 1.0
    
    # Ensure the range covers the solution
    if np.sum(np.clip(h + low, 0, 1)) > target_sum:
        low = -np.max(h) - 1.0
    if np.sum(np.clip(h + high, 0, 1)) < target_sum:
        high = 1.0 - np.min(h) + 1.0
        
    for _ in range(50):
        mid = (low + high) / 2.0
        s_mid = np.sum(np.clip(h + mid, 0.0, 1.0))
        if s_mid > target_sum:
            high = mid
        else:
            low = mid
            
    delta = (low + high) / 2.0
    h_proj = np.clip(h + delta, 0.0, 1.0)
    
    # Final precision fix
    diff = target_sum - np.sum(h_proj)
    if abs(diff) > 1e-14:
        if diff > 0:
            idx = np.argmin(h_proj)
            h_proj[idx] = min(1.0, h_proj[idx] + diff)
        else:
            idx = np.argmax(h_proj)
            h_proj[idx] = max(0.0, h_proj[idx] + diff)
            
    return h_proj

def _spectral_to_h(coeffs, K):
    """
    Convert spectral coefficients to h vector.
    coeffs = [a1, b1, ..., aK, bK]
    h_k = 0.5 + sum_{m=1}^K (a_m cos(2pi m k / N) + b_m sin(2pi m k / N))
    """
    k = np.arange(N)
    h = np.full(N, 0.5)
    for m in range(1, K + 1):
        a_m = coeffs[2 * (m - 1)]
        b_m = coeffs[2 * (m - 1) + 1]
        angle = 2 * np.pi * m * k / N
        h += a_m * np.cos(angle) + b_m * np.sin(angle)
    
    # Project to [0,1] and sum N/2
    h = _project_sum(h, N / 2.0)
    return h

def _score_from_coeffs(coeffs, K):
    h = _spectral_to_h(coeffs, K)
    return _score(h)

def _binary_polish(h, max_iters=1000):
    """
    Try to improve h by snapping to binary and performing local swaps.
    """
    h = np.array(h, dtype=np.float64)
    best_h = h.copy()
    best_score = _score(h)
    
    # 1. Snap to binary
    h_bin = np.where(h > 0.5, 1.0, 0.0)
    # Ensure exactly N/2 ones
    ones = np.sum(h_bin == 1.0)
    if ones > N / 2:
        # Remove some ones (those with lowest h value? or random?)
        idx_ones = np.where(h_bin == 1.0)[0]
        # Sort by h value ascending, remove lowest
        idx_sorted = idx_ones[np.argsort(h[idx_ones])]
        to_remove = idx_sorted[:ones - N // 2]
        h_bin[to_remove] = 0.0
    elif ones < N / 2:
        idx_zeros = np.where(h_bin == 0.0)[0]
        idx_sorted = idx_zeros[np.argsort(-h[idx_zeros])]
        to_add = idx_sorted[:N // 2 - ones]
        h_bin[to_add] = 1.0
        
    s_bin = _score(h_bin)
    if s_bin < best_score:
        best_score = s_bin
        best_h = h_bin.copy()
        
    # 2. Local Swaps (Flipping 0 to 1 and 1 to 0)
    # We want to minimize max overlap.
    # Greedy local search: try swapping pairs (i,j) where h_i=1, h_j=0.
    # If it improves score, keep it.
    
    h_cur = best_h.copy()
    s_cur = best_score
    
    for _ in range(max_iters):
        if time.time() > _DEADLINE - 5:
            break
            
        improved = False
        # Get indices of 1s and 0s
        idx_ones = np.where(h_cur == 1.0)[0]
        idx_zeros = np.where(h_cur == 0.0)[0]
        
        # Randomly sample some swaps to avoid O(N^2)
        # If N is small, we could do all, but N=100, N^2=10000, might be slow if done many times.
        # Let's try a few random swaps per iteration.
        num_swaps_to_try = 20
        for _ in range(num_swaps_to_try):
            if len(idx_ones) == 0 or len(idx_zeros) == 0:
                break
            i = idx_ones[np.random.randint(len(idx_ones))]
            j = idx_zeros[np.random.randint(len(idx_zeros))]
            
            h_test = h_cur.copy()
            h_test[i] = 0.0
            h_test[j] = 1.0
            
            s_test = _score(h_test)
            if s_test < s_cur - 1e-12:
                h_cur = h_test
                s_cur = s_test
                best_h = h_cur.copy()
                best_score = s_cur
                improved = True
                # Update indices? Or just recalculate in next loop?
                # Recalculating is safer.
                idx_ones = np.where(h_cur == 1.0)[0]
                idx_zeros = np.where(h_cur == 0.0)[0]
                break # Restart sampling after improvement
                
        if not improved:
            # Try a few more random restarts of swaps if stuck?
            # Or break if no improvement in a while.
            # Let's just break for now to save time for other candidates.
            break
            
    return best_h, best_score

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Spectral Global Optimization
    # Use a small number of Fourier modes
    K = 10
    dim = 2 * K
    
    # Bounds for coefficients: amplitude should be roughly <= 0.5 to stay in [0,1] before projection
    # But projection handles it. Let's use [-1, 1] to be safe.
    bounds = [(-1.0, 1.0)] * dim
    
    # Differential Evolution
    # maxiter is limited by time. 
    # DE can be slow. We will run it with a modest population.
    
    try:
        if time.time() < _DEADLINE - 60:
            de_result = differential_evolution(
                _score_from_coeffs, 
                bounds, 
                args=(K,),
                seed=42,
                maxiter=50,
                tol=0.01,
                popsize=15,
                polish=False
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_h = _spectral_to_h(de_result.x, K)
                
    except Exception:
        pass

    # Phase 2: Local Refinement with Nelder-Mead on Spectral Coeffs
    if best_h is not None:
        # We need initial coefficients.
        # If we have best_h, we can compute its Fourier coefficients.
        # Or we can just use the DE result if we kept it.
        # Let's re-extract coeffs from best_h for NM start.
        
        k = np.arange(N)
        coeffs_init = np.zeros(dim)
        for m in range(1, K + 1):
            angle = 2 * np.pi * m * k / N
            coeffs_init[2*(m-1)] = (2.0/N) * np.sum(best_h * np.cos(angle))
            coeffs_init[2*(m-1)+1] = (2.0/N) * np.sum(best_h * np.sin(angle))
            
        try:
            if time.time() < _DEADLINE - 30:
                nm_result = minimize(
                    _score_from_coeffs, 
                    coeffs_init, 
                    args=(K,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if nm_result.fun < best_score:
                    best_score = nm_result.fun
                    best_h = _spectral_to_h(nm_result.x, K)
        except Exception:
            pass

    # Phase 3: Binary Polish
    if best_h is not None:
        h_bin, s_bin = _binary_polish(best_h, max_iters=500)
        if s_bin < best_score:
            best_score = s_bin
            best_h = h_bin

    # Phase 4: If spectral approach failed or found nothing, fallback to random binary search
    if best_h is None or best_score > 0.5:
        for _ in range(100):
            if time.time() > _DEADLINE - 5:
                break
            h_rand = np.random.choice([0.0, 1.0], size=N)
            h_rand = _project_sum(h_rand, N/2.0)
            s_rand = _score(h_rand)
            if s_rand < best_score:
                best_score = s_rand
                best_h = h_rand.copy()
                
            # Polish this random one
            h_pol, s_pol = _binary_polish(h_rand, max_iters=100)
            if s_pol < best_score:
                best_score = s_pol
                best_h = h_pol

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum(best_h, N / 2.0)
    return best_h
