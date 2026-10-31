"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np

N = 100


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _overlap_for_2block(k, a, b):
    """Compute max overlap for 2-block with first k elements = a, rest = b."""
    n = N
    if k <= 0 or k >= n:
        return float('inf')
    vals = []
    if k >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if n - k >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b)
    return max(vals)


def _smooth_overlap_2block(k, a, b, beta=50.0):
    """Smooth surrogate for max overlap using log-sum-exp."""
    n = N
    vals = []
    if k >= 2:
        vals.append(a * (1.0 - a))
    vals.append(a * (1.0 - b))
    if n - k >= 2:
        vals.append(b * (1.0 - b))
    vals.append(b)
    v = np.array(vals, dtype=np.float64)
    m = np.max(v)
    if m < 1e-300:
        return 0.0
    return m + np.log(np.sum(np.exp(beta * (v - m)))) / beta


def _optimize_2block_smooth(k, a0, b0):
    """Use scipy to optimize a for the 2-block construction with smooth objective."""
    from scipy.optimize import minimize
    
    n = N
    
    def objective(x):
        a = x[0]
        if a < 0.0 or a > 1.0:
            return 1.0
        b = (N / 2.0 - k * a) / (N - k)
        if b < -1e-6 or b > 1.0 + 1e-6:
            return 1.0
        b = np.clip(b, 0.0, 1.0)
        return _smooth_overlap_2block(k, a, b)
    
    # Try multiple starting points
    best_x = None
    best_f = float('inf')
    
    a_min = max(0.0, (N / 2.0 - N + k) / k)
    a_max = min(1.0, N / (2.0 * k))
    
    if a_min >= a_max:
        return None, float('inf')
    
    # Multiple initial guesses
    for frac in [0.1, 0.3, 0.5, 0.7, 0.9]:
        a_init = a_min + frac * (a_max - a_min)
        b_init = (N / 2.0 - k * a_init) / (N - k)
        if b_init < 0 or b_init > 1:
            continue
        try:
            res = minimize(objective, np.array([a_init]), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.success or res.fun < best_f:
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
        except Exception:
            pass
    
    if best_x is None:
        return None, float('inf')
    
    a_opt = np.clip(best_x[0], 0.0, 1.0)
    b_opt = (N / 2.0 - k * a_opt) / (N - k)
    b_opt = np.clip(b_opt, 0.0, 1.0)
    
    # Verify with exact objective
    s_exact = _overlap_for_2block(k, a_opt, b_opt)
    return (a_opt, b_opt), s_exact


def construct():
    deadline = time.time() + 100.0  # safety margin before 120s kill

    best_h = None
    best_score = float('inf')

    # Try various splits k (number of elements in first block)
    for k in range(1, N):
        # Constraint: k*a + (N-k)*b = N/2
        # b = (N/2 - k*a) / (N-k)
        # We want to minimize max(a*(1-a), a*(1-b), b*(1-b), b)
        # subject to a in [0,1], b in [0,1]
        
        # Bounds on a:
        # b >= 0: N/2 - k*a >= 0 -> a <= N/(2k)
        # b <= 1: N/2 - k*a <= N-k -> k*a >= N/2 - N + k -> a >= (N/2 - N + k)/k
        a_min = max(0.0, (N / 2.0 - N + k) / k)
        a_max = min(1.0, N / (2.0 * k))
        
        if a_min >= a_max:
            continue
        
        # The objective is piecewise. Let's sample a range of a values
        # and find the one that minimizes the max overlap.
        # Use a fine grid + local refinement
        
        # Coarse grid
        n_samples = 300
        a_vals = np.linspace(a_min + 1e-9, a_max - 1e-9, n_samples)
        
        best_local_score = float('inf')
        best_local_a = None
        
        for a_val in a_vals:
            b_val = (N / 2.0 - k * a_val) / (N - k)
            if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                continue
            b_val = np.clip(b_val, 0.0, 1.0)
            s = _overlap_for_2block(k, a_val, b_val)
            if s < best_local_score:
                best_local_score = s
                best_local_a = a_val
        
        if best_local_a is not None:
            # Refine around best_local_a with finer grid
            a_refine_min = max(a_min + 1e-9, best_local_a - (a_max - a_min) / n_samples * 3)
            a_refine_max = min(a_max - 1e-9, best_local_a + (a_max - a_min) / n_samples * 3)
            if a_refine_min < a_refine_max:
                a_refine_vals = np.linspace(a_refine_min, a_refine_max, 400)
                for a_val in a_refine_vals:
                    b_val = (N / 2.0 - k * a_val) / (N - k)
                    if b_val < -1e-12 or b_val > 1.0 + 1e-12:
                        continue
                    b_val = np.clip(b_val, 0.0, 1.0)
                    s = _overlap_for_2block(k, a_val, b_val)
                    if s < best_local_score:
                        best_local_score = s
                        best_local_a = a_val
            
            # Now use scipy for high-precision polish
            if time.time() < deadline - 5.0:
                opt_result, s_smooth = _optimize_2block_smooth(k, best_local_a, 
                    (N / 2.0 - k * best_local_a) / (N - k))
                if opt_result is not None and s_smooth < best_local_score:
                    best_local_score = s_smooth
                    best_local_a = opt_result[0]
            
            if best_local_score < best_score:
                best_score = best_local_score
                a_opt = best_local_a
                b_opt = (N / 2.0 - k * a_opt) / (N - k)
                b_opt = np.clip(b_opt, 0.0, 1.0)
                h = np.zeros(N)
                h[:k] = a_opt
                h[k:] = b_opt
                # Fix sum precisely
                s = h.sum()
                diff = N / 2.0 - s
                if abs(diff) > 1e-15:
                    # Distribute correction to the last element of first block
                    h[k - 1] += diff
                    h = np.clip(h, 0.0, 1.0)
                    # Re-fix if clipping changed sum
                    s2 = h.sum()
                    diff2 = N / 2.0 - s2
                    if abs(diff2) > 1e-15:
                        h[k - 1] += diff2
                        h = np.clip(h, 0.0, 1.0)
                best_h = h.copy()
        
        if time.time() > deadline:
            break

    if best_h is None:
        # Fallback
        A = (3.0 - math.sqrt(5.0)) / 2.0
        B = 1.0 - A
        h = np.zeros(N)
        h[:N//2] = A
        h[N//2:] = B
        best_h = h

    return best_h
