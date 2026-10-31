import numpy as np
import time
from scipy.optimize import minimize

N = 50

def _smooth_max(x, beta=100.0):
    m = np.max(x)
    if m < 0:
        m = 0
    return m + np.log(np.sum(np.exp(beta * (x - m)))) / beta

def _negative_ratio_smooth(params, beta=100.0):
    h = np.exp(params)
    h = h / np.sum(h)
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = _smooth_max(conv, beta)
    if max_conv < 1e-15:
        return 0.0
    return -l2_sq / max_conv

def _exact_ratio(h):
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    l2_sq = np.sum(h**2)
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv < 1e-15:
        return 0.0
    return l2_sq / (l1 * max_conv)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110

    # Initialize with Sidon set
    h_init = np.zeros(N)
    indices = [1, 2, 5, 10, 20, 40]
    for i in indices:
        if i < N:
            h_init[i] = 1.0
    h_init = h_init / np.sum(h_init)

    params_init = np.log(h_init + 1e-10)

    best_score = _exact_ratio(h_init)
    best_h = h_init.copy()

    # Optimize with smooth objective at increasing beta
    for beta in [30.0, 80.0, 150.0]:
        if time.time() > deadline - 15:
            break

        result = minimize(
            _negative_ratio_smooth,
            params_init,
            args=(beta,),
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
        )

        h_result = np.exp(result.x)
        h_result = h_result / np.sum(h_result)

        score = _exact_ratio(h_result)
        if score > best_score:
            best_score = score
            best_h = h_result.copy()
            params_init = result.x

    # Ensure nonnegativity
    best_h = np.maximum(best_h, 0)

    return best_h.tolist()
# EVOLVE-BLOCK-END
