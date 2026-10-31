"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import comb

N = 100
DEADLINE_MARGIN = 15.0  # seconds to spare before 150s kill


def _score(h):
    """Compute max overlap of h with shifted (1-h), treating h as zero outside [0,2]."""
    n = len(h)
    shifted = np.zeros(n)
    shifted[:-1] = h[1:]  # shifted[x] = h[x+1] for x in [0,N-2]
    overlap = h * (1.0 - shifted)
    return float(np.max(overlap))


def _fix_sum(h, target=None):
    """Adjust h so that sum(h) == target (default N/2), preserving [0,1] bounds."""
    if target is None:
        target = N / 2.0
    h = h.copy()
    for _ in range(10):
        diff = target - h.sum()
        if abs(diff) < 1e-14:
            break
        step = diff / len(h)
        h += step
        h = np.clip(h, 0.0, 1.0)
    # Final correction on non-saturated entries
    diff = target - h.sum()
    if abs(diff) > 1e-14:
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if mask.sum() > 0:
            h[mask] += diff / mask.sum()
            h = np.clip(h, 0.0, 1.0)
    return h


def _binomial_transition_profile(k, w):
    """
    Generate a transition profile using cumulative binomial distribution.
    k = number of binomial trials (order), w = window width.
    Returns an array of length w with values in [0,1] that monotonically increases.
    """
    if w <= 1:
        return np.array([0.5])
    
    # Compute binomial CDF at each point in the window
    # We want a smooth transition from low to high values
    # Use binomial(k, p) where p is chosen so that the CDF spans [0,1] over the window
    
    # The transition goes from 0 to 1 over w steps
    # We use the binomial CDF: P(X <= j) for j = 0, 1, ..., w-1
    # where X ~ Binomial(k, 0.5)
    
    # To make it span [0,1], we normalize
    # CDF at j: sum_{i=0}^{j} C(k,i) * 0.5^k
    
    cdf_vals = np.zeros(w)
    for j in range(w):
        cdf_vals[j] = sum(comb(k, i, exact=True) for i in range(j + 1)) * (0.5 ** k)
    
    # Normalize to [0, 1]
    if cdf_vals[-1] > 0:
        cdf_vals = cdf_vals / cdf_vals[-1]
    
    return cdf_vals


def _construct_binomial_h(h_low, h_high, pos, width, k):
    """
    Construct h as a piecewise constant function with a single transition window.
    - h_low: value before the transition
    - h_high: value after the transition
    - pos: position of the transition start (0 to N-1)
    - width: width of the transition window
    - k: binomial order for the transition profile
    
    The function is periodic: positions wrap around modulo N.
    """
    h = np.full(N, h_low)
    
    # Define the transition window positions
    # The transition occupies positions [pos, pos+width-1] mod N
    profile = _binomial_transition_profile(k, width)
    
    for j in range(width):
        idx = (pos + j) % N
        # Interpolate between h_low and h_high using the binomial CDF profile
        h[idx] = h_low + (h_high - h_low) * profile[j]
    
    return h


def _optimize_binomial_transition():
    """
    Optimize piecewise-constant h with binomial-distribution transition window.
    Parameters: h_low, h_high, pos, width, k
    Use derivative-free global search (Nelder-Mead with multiple starts).
    """
    best_score = float('inf')
    best_h = None
    best_params = None
    
    def evaluate(params):
        h_low, h_high, pos, width, k = params
        h_low = np.clip(h_low, 0.0, 1.0)
        h_high = np.clip(h_high, 0.0, 1.0)
        pos = int(round(pos)) % N
        width = max(2, min(N, int(round(width))))
        k = max(2, min(20, int(round(k))))
        
        h = _construct_binomial_h(h_low, h_high, pos, width, k)
        h = np.clip(h, 0.0, 1.0)
        h = _fix_sum(h)
        s = _score(h)
        return s, h, (h_low, h_high, pos, width, k)
    
    # Phase 1: Coarse grid search
    h_low_vals = np.linspace(0.1, 0.45, 8)
    h_high_vals = np.linspace(0.55, 0.9, 8)
    pos_vals = np.arange(0, N, max(1, N // 20))
    width_vals = np.arange(4, 30, 2)
    k_vals = [2, 3, 4, 5, 6, 8, 10]
    
    for h_low in h_low_vals:
        for h_high in h_high_vals:
            for pos in pos_vals:
                for width in width_vals:
                    for k in k_vals:
                        s, h, params = evaluate([h_low, h_high, pos, width, k])
                        if s < best_score:
                            best_score = s
                            best_h = h.copy()
                            best_params = params
    
    # Phase 2: Local Nelder-Mead refinement from best grid point
    if best_params is not None:
        def obj_bin(params):
            s, h, _ = evaluate(params)
            return s
        
        x0 = np.array([float(best_params[0]), float(best_params[1]), 
                       float(best_params[2]), float(best_params[3]), float(best_params[4])])
        try:
            res = minimize(obj_bin, x0, method='Nelder-Mead',
                           options={'maxiter': 15000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                params = (float(res.x[0]), float(res.x[1]), int(round(res.x[2])) % N,
                          max(2, min(N, int(round(res.x[3])))), max(2, min(20, int(round(res.x[4])))))
                s, h, _ = evaluate(params)
                best_h = h.copy()
                best_params = params
        except Exception:
            pass
    
    # Phase 3: Multiple random starting points
    np.random.seed(42)
    for _ in range(30):
        h_low_init = np.random.uniform(0.1, 0.45)
        h_high_init = np.random.uniform(0.55, 0.9)
        pos_init = np.random.randint(0, N)
        width_init = np.random.randint(4, 30)
        k_init = np.random.randint(2, 15)
        
        x0 = np.array([h_low_init, h_high_init, float(pos_init), float(width_init), float(k_init)])
        try:
            res = minimize(obj_bin, x0, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            if res.fun < best_score:
                best_score = res.fun
                params = (float(res.x[0]), float(res.x[1]), int(round(res.x[2])) % N,
                          max(2, min(N, int(round(res.x[3])))), max(2, min(20, int(round(res.x[4])))))
                s, h, _ = evaluate(params)
                best_h = h.copy()
                best_params = params
        except Exception:
            pass
    
    if best_h is None:
        best_h = np.full(N, 0.5)
        best_score = _score(best_h)
    
    return best_score, best_h, best_params


def _optimize_direct(h0, max_iter=3000):
    """
    Direct SLSQP optimization with smooth max (log-sum-exp) objective.
    """
    beta = 500.0

    def smooth_overlap(h):
        n = len(h)
        shifted = np.zeros(n)
        shifted[:-1] = h[1:]
        overlap = h * (1.0 - shifted)
        m = np.max(overlap)
        if m < 0:
            return m
        return m + np.log(np.sum(np.exp(beta * (overlap - m)))) / beta

    def obj(h):
        h = np.clip(h, 0.0, 1.0)
        return smooth_overlap(h)

    constraints = [{'type': 'eq', 'fun': lambda h: h.sum() - N / 2.0}]

    try:
        res = minimize(obj, h0, method='SLSQP',
                       bounds=[(0.0, 1.0)] * N,
                       constraints=constraints,
                       options={'maxiter': max_iter, 'ftol': 1e-15})
        h_opt = np.clip(res.x, 0.0, 1.0)
        h_opt = _fix_sum(h_opt)
        return _score(h_opt), h_opt
    except Exception:
        return float('inf'), h0


def _local_polish(h, max_iters=3000):
    """
    Local search: try small adjustments to individual entries.
    """
    h = h.copy()
    current_score = _score(h)
    improved = True
    iters = 0
    deltas = [0.005, -0.005, 0.01, -0.01, 0.002, -0.002, 0.001, -0.001]

    while improved and iters < max_iters:
        improved = False
        iters += 1

        for i in range(N):
            for delta in deltas:
                new_val = h[i] + delta
                if new_val < 0 or new_val > 1:
                    continue
                h_trial = h.copy()
                h_trial[i] = new_val
                h_trial = _fix_sum(h_trial)
                s = _score(h_trial)
                if s < current_score - 1e-13:
                    h = h_trial
                    current_score = s
                    improved = True
                    break

    return current_score, h


def construct():
    start_time = time.time()
    deadline = start_time + (150.0 - DEADLINE_MARGIN)

    best_score = float('inf')
    best_h = None

    # Phase 1: Binomial transition optimization (new structural approach)
    if time.time() < deadline - 10.0:
        s_bin, h_bin, _ = _optimize_binomial_transition()
        if s_bin < best_score:
            best_score = s_bin
            best_h = h_bin.copy()

    # Phase 2: Direct SLSQP from best candidate
    if best_h is not None and time.time() < deadline - 25.0:
        s_direct, h_direct = _optimize_direct(best_h, max_iter=3000)
        if s_direct < best_score:
            best_score = s_direct
            best_h = h_direct.copy()

    # Phase 3: Local polish
    if best_h is not None and time.time() < deadline - 10.0:
        s_polish, h_polish = _local_polish(best_h, max_iters=3000)
        if s_polish < best_score:
            best_score = s_polish
            best_h = h_polish.copy()

    # Phase 4: Final high-precision SLSQP
    if best_h is not None and time.time() < deadline - 5.0:
        s_final, h_final = _optimize_direct(best_h, max_iter=5000)
        if s_final < best_score:
            best_score = s_final
            best_h = h_final.copy()

    # Ensure valid output
    if best_h is None:
        best_h = np.full(N, 0.5)

    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _fix_sum(best_h)

    return best_h
