import numpy as np
import time
from scipy.optimize import minimize

N = 50
DELTA = 1.0 / N

def _autoconv_all(h):
    """Compute all autoconvolution values at breakpoints."""
    n = len(h)
    num_points = 2 * n - 1
    ff_vals = np.zeros(num_points)
    
    for k in range(num_points):
        total = 0.0
        i_min = max(0, k - (n - 1))
        i_max = min(n - 1, k)
        for i in range(i_min, i_max + 1):
            j_min = max(0, k - i - 1)
            j_max = min(n - 1, k - i + 1)
            if j_min > j_max:
                continue
            js = np.arange(j_min, j_max + 1)
            lo = np.maximum(i, k - js)
            hi = np.minimum(i + 1, k - js + 1)
            overlaps = np.maximum(0, hi - lo)
            total += np.sum(h[i] * h[js] * overlaps)
        ff_vals[k] = total * DELTA
    
    return ff_vals

def _smooth_autoconv_sup(h, beta=50.0):
    """Compute smooth approximation of sup norm using log-sum-exp."""
    ff_vals = _autoconv_all(h)
    max_val = np.max(ff_vals)
    if max_val == 0:
        return 0.0
    return max_val + np.log(np.sum(np.exp((ff_vals - max_val) * beta))) / beta

def _smooth_ratio(h, beta=50.0):
    """Compute smooth ratio."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    sup_ff = _smooth_autoconv_sup(h, beta)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def _ratio(h):
    """Compute the ratio ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.maximum(h, 0)
    sum_h2 = np.sum(h**2)
    sum_h = np.sum(h)
    if sum_h < 1e-15:
        return 0.0
    ff_vals = _autoconv_all(h)
    sup_ff = np.max(ff_vals)
    if sup_ff < 1e-15:
        return 0.0
    return sum_h2 / (sum_h * sup_ff)

def construct():
    start_time = time.time()
    DEADLINE = 145.0

    np.random.seed(42)

    best_h = np.ones(N)
    best_score = _ratio(best_h)

    # Generate diverse initializations
    initializations = []

    # Uniform
    initializations.append(np.ones(N) / N)

    # Peaked at center
    center = N // 2
    for width in [3, 5, 7, 10]:
        h = np.zeros(N)
        for i in range(center - width // 2, center + width // 2 + 1):
            if 0 <= i < N:
                h[i] = 1.0
        h /= np.sum(h)
        initializations.append(h)

    # Random sparse patterns
    for trial in range(30):
        h = np.zeros(N)
        num_ones = np.random.randint(5, 20)
        indices = np.random.choice(N, num_ones, replace=False)
        h[indices] = np.random.uniform(0.3, 3.0, num_ones)
        h /= np.sum(h)
        initializations.append(h)

    # Structured: alternating
    h = np.zeros(N)
    h[::2] = 1.0
    h /= np.sum(h)
    initializations.append(h)

    # Structured: endpoints heavy
    h = np.ones(N)
    h[0] = 3.0
    h[-1] = 3.0
    h /= np.sum(h)
    initializations.append(h)

    # Optimize each initialization with L-BFGS-B on smoothed objective
    for h0 in initializations:
        if time.time() - start_time > DEADLINE - 5:
            break

        def neg_smooth_ratio(x, beta=30.0):
            h = np.maximum(x, 0)
            s = np.sum(h)
            if s < 1e-15:
                return 0.0
            h_norm = h / s
            return -_smooth_ratio(h_norm, beta)

        try:
            result = minimize(neg_smooth_ratio, h0.copy(), method='L-BFGS-B',
                             bounds=[(0, None)] * N,
                             options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-10})

            h_candidate = np.maximum(result.x, 0)
            s = np.sum(h_candidate)
            if s > 1e-15:
                h_candidate = h_candidate / s
                score = _ratio(h_candidate)
                if score > best_score:
                    best_score = score
                    best_h = h_candidate.copy()
        except Exception:
            pass

    # Polish the best result with coordinate descent
    if time.time() - start_time < DEADLINE - 3:
        h = best_h.copy()
        best_score = _ratio(h)
        n = len(h)

        for it in range(300):
            if time.time() - start_time > DEADLINE - 2:
                break

            improved = False
            for i in range(n):
                if h[i] < 1e-15:
                    continue

                for delta in [0.01, 0.005, 0.001, -0.01, -0.005, -0.001]:
                    h_new = h.copy()
                    h_new[i] = max(0, h[i] + delta)

                    s = np.sum(h_new)
                    if s < 1e-15:
                        continue
                    h_new = h_new / s

                    score = _ratio(h_new)
                    if score > best_score + 1e-12:
                        h = h_new
                        best_score = score
                        improved = True
                        break

            if not improved:
                break

        best_h = h

    return best_h.tolist()
