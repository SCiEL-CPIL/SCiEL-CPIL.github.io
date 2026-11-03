"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Frank-Wolfe (Conditional Gradient) method on the convex formulation
minimize t subject to C_k <= t for all k, h in [0,1]^N, sum(h) = N/2.
This exploits the global convexity of the feasible region and the linear 
dependence of overlaps on h, avoiding local minima.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    for _ in range(10):
        diff = target - np.sum(h)
        if abs(diff) < 1e-15:
            break
        h += diff / N
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact adjustment
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            diff -= add
            if abs(diff) > 1e-12:
                mask = h < 1.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            diff += sub
            if abs(diff) > 1e-12:
                mask = h > 0.0
                if np.any(mask):
                    h[mask] += diff / np.sum(mask)
                    h = np.clip(h, 0.0, 1.0)
                    
    return h


def _frank_wolfe(h_init, max_iter=5000):
    """
    Frank-Wolfe algorithm for:
    min t
    s.t. C_k(h) <= t
         0 <= h_i <= 1
         sum(h) = N/2
    """
    h = _fix_sum(h_init.copy())
    t = _score(h)
    
    # Pre-compute gradient of C_k with respect to h.
    # C_k = (2/N) * (N/2 - sum_j h_j h_{j+k})
    # dC_k/dh_i = -(2/N) * (h_{i+k} + h_{i-k}) (with boundary handling)
    
    # We maintain the dual variables (multipliers) for the constraints C_k <= t.
    # Alternatively, we can use a penalty or direct subgradient.
    # Frank-Wolfe requires a convex objective. Let's use the smooth approximation or 
    # directly minimize t via bisection on t and feasibility check?
    # No, standard FW minimizes a convex function f(x). 
    # Let f(h) = max_k C_k(h). This is convex.
    # We can apply FW directly to f(h).
    
    best_h = h.copy()
    best_val = t
    
    for it in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        # 1. Compute subgradient of f(h) = max_k C_k(h)
        cs = _linear_overlap_values(h)
        max_c = np.max(cs)
        
        # Find active constraints
        active = np.where(cs >= max_c - 1e-8)[0]
        
        # Subgradient is the average of gradients of active constraints
        # grad C_k = -(2/N) * (h_{i+k} + h_{i-k})
        grad = np.zeros(N)
        for k in active:
            # Shifted terms
            # h_{i+k}
            if k == 0:
                shift1 = h
            else:
                shift1 = np.zeros(N)
                shift1[:N-k] = h[k:]
            # h_{i-k}
            if k == 0:
                shift2 = h
            else:
                shift2 = np.zeros(N)
                shift2[k:] = h[:N-k]
                
            grad += -(2.0 / N) * (shift1 + shift2)
            
        grad /= len(active)
        
        # 2. Linear Minimization Oracle (LMO)
        # Minimize <grad, y> s.t. y in P
        # P = { y | 0 <= y_i <= 1, sum(y) = N/2 }
        # This is equivalent to assigning the N/2 units of "mass" to the indices
        # with the smallest gradient values.
        
        # Since sum is fixed, we pick the smallest N/2 values of grad.
        # However, if we pick a binary vector (0 or 1), it's a vertex.
        # The LMO for the box with sum constraint is:
        # Sort indices by grad value.
        # The optimal vertex has 1s at the indices with the smallest grad values.
        # Specifically, if N/2 is integer, we pick the bottom N/2 indices.
        
        sorted_idx = np.argsort(grad)
        y = np.zeros(N)
        y[sorted_idx[:N//2]] = 1.0
        
        # 3. Line Search
        # Minimize f(h + alpha * (y - h)) for alpha in [0, 1]
        # f is convex, so we can use backtracking or ternary search.
        # Let's use a simple grid search or ternary search for speed and robustness.
        
        h_dir = y - h
        best_alpha = 0.0
        best_f = f_val = _score(h)
        
        # Ternary search on alpha
        lo = 0.0
        hi = 1.0
        for _ in range(20):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0
            h_m1 = h + m1 * h_dir
            h_m2 = h + m2 * h_dir
            f_m1 = _score(h_m1)
            f_m2 = _score(h_m2)
            
            if f_m1 < f_m2:
                hi = m2
            else:
                lo = m1
                
        alpha = (lo + hi) / 2.0
        h_new = h + alpha * h_dir
        f_new = _score(h_new)
        
        if f_new < best_f:
            best_f = f_new
            best_alpha = alpha
            best_h = h_new.copy()
            
        h = h + best_alpha * h_dir
        
        # Check convergence
        if best_f - _score(h) < 1e-10:
            break
            
    return best_h, best_f


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)

    best_h = None
    best_score = float('inf')

    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()

    i = np.arange(N)
    
    # 1. Analytical Starts
    starts = []
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)))
    starts.append(_fix_sum(0.5 + 0.5 * (1.0 - 2.0 * np.abs((2.0 * i / N) - 1.0))))
    starts.append(np.full(N, 0.5))
    starts.append(_fix_sum(0.5 + 0.5 * (2.0 * (i / N) - 1.0)))
    
    # 2. Random Starts
    for _ in range(5):
        if time.time() - t0 > _DEADLINE - 30:
            break
        h_rand = np.random.uniform(0, 1, N)
        starts.append(_fix_sum(h_rand))

    # Run Frank-Wolfe from all starts
    for h_start in starts:
        if time.time() - t0 > _DEADLINE - 10:
            break
            
        h_res, s_res = _frank_wolfe(h_start)
        _update_best(h_res)

    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = _fix_sum(best_h)
    return best_h
