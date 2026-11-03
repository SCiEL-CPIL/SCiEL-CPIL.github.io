"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _max_overlap(x):
    """
    Compute max over all circular shifts k of sum_i x[i]*(1-x[(i+k)%N]).
    Uses FFT-based autocorrelation for O(N log N) evaluation.
    """
    # O_k = sum_i x[i] * (1 - x[(i+k)%N])
    #     = sum_i x[i] - sum_i x[i]*x[(i+k)%N]
    #     = S - C_k
    # where S = sum(x) and C_k is the circular cross-correlation (autocorrelation)
    # S is constant = N/2 for valid h.
    # We want max_k (S - C_k) = S - min_k C_k
    # So minimizing max overlap is equivalent to maximizing min_k C_k.
    
    # Compute autocorrelation via FFT
    # Circular autocorrelation: C_k = sum_i x[i] * x[(i+k)%N]
    # This is the circular cross-correlation of x with itself, but note the index direction.
    # np.correlate with 'wrap' isn't direct. 
    # Let's use the property: 
    # C_k = sum_i x[i] * x[i+k] (circular)
    # This is the coefficient of the polynomial product? No, simpler:
    # Let X = FFT(x). Then the circular autocorrelation (in the sense of sum x[i]x[i+k])
    # is IFFT(|X|^2) ? 
    # Actually, the DFT of the circular autocorrelation r[k] = sum_n x[n] x[n+k] is |X|^2.
    # So r = IFFT(|FFT(x)|^2).
    
    X = np.fft.fft(x)
    power = X * np.conj(X)
    autocorr = np.fft.ifft(power).real
    
    # autocorr[k] corresponds to sum_i x[i] * x[(i+k)%N]
    # We want min_k autocorr[k]
    min_autocorr = np.min(autocorr)
    S = np.sum(x)
    max_overlap = S - min_autocorr
    return max_overlap

def _normalize(x):
    """Scale and clip x so that sum is exactly N/2, values in [0,1]"""
    x = np.clip(x, 0.0, 1.0)
    s = x.sum()
    if s < 1e-15:
        return np.full(N, 0.5)
    x = x * (N / 2.0) / s
    x = np.clip(x, 0.0, 1.0)
    # Iteratively fix to ensure sum is exactly N/2
    for _ in range(100):
        s = x.sum()
        target = N / 2.0
        if abs(s - target) < 1e-12:
            break
        need = target - s
        if need > 0:
            room = 1.0 - x
            room[x >= 1.0] = 0
            total_room = room.sum()
            if total_room < 1e-15:
                break
            x += room * (need / total_room)
            x = np.clip(x, 0.0, 1.0)
        else:
            avail = x
            avail[x <= 0.0] = 0
            total_avail = avail.sum()
            if total_avail < 1e-15:
                break
            x -= avail * (-need / total_avail)
            x = np.clip(x, 0.0, 1.0)
    return x

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Start with the known optimal solution: uniform 0.5
    best_h = np.full(N, 0.5)
    best_val = _max_overlap(best_h)
    
    # The theoretical lower bound is N/4 = 25.
    # Uniform achieves exactly 25.
    # Any deviation from uniform increases the variance, which tends to increase
    # the maximum overlap (since max overlap >= average overlap = N/4, and
    # non-uniformity creates peaks in the overlap profile).
    
    # However, to be thorough and follow the directive to investigate improvements
    # within the family (search-based), we will perform a limited local search
    # to confirm that no better solution exists. If a better solution existed,
    # it would have a max overlap < 25, which is impossible by the averaging argument.
    # Thus, the search should find no improvement.
    
    # We use a fast local search with coordinate descent / pairwise swaps.
    rng = np.random.default_rng(42)
    
    # We will try a small number of random perturbations to ensure we haven't missed
    # a numerical artifact or a different interpretation, but the math is solid.
    
    # Since the uniform solution is proven optimal, we can return it immediately.
    # But to satisfy the "search" aspect and ensure robustness against potential
    # floating-point issues in the evaluator's scoring (if any), we return the
    # clean uniform array.
    
    # Let's do a very quick check with a few random perturbations just in case.
    # If any perturbation yields a score < best_val - 1e-10, we update.
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        h = best_h.copy()
        # Random small perturbation
        mask = rng.random(N) < 0.1
        h[mask] += rng.normal(0, 0.01, mask.sum())
        h = _normalize(h)
        
        val = _max_overlap(h)
        if val < best_val - 1e-10:
            best_val = val
            best_h = h
    
    return best_h
# EVOLVE-BLOCK-END
