"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s safety margin for 120s limit

    # Objective: maximize R = ||h||_2^2 / (||h||_1 * ||h*h||_inf)
    # Subject to h >= 0.
    # Note: R is scale invariant: R(ch) = R(h).
    # So we can fix a normalization, e.g., ||h||_1 = 1.
    # Then maximize ||h||_2^2 / ||h*h||_inf.

    def compute_ratio(h):
        h = np.asarray(h)
        l1 = np.sum(h)
        if l1 == 0:
            return 0.0
        l2_sq = np.sum(h**2)
        # Compute autocorrelation
        ac = np.correlate(h, h, mode='full')
        sup = np.max(ac)
        if sup == 0:
            return 0.0
        return l2_sq / (l1 * sup)

    # Since scale invariant, we can fix ||h||_1 = 1.
    # Maximize f(h) = ||h||_2^2 / ||h*h||_inf
    # Minimize -f(h) = -||h||_2^2 / ||h*h||_inf
    # Subject to sum(h) = 1, h >= 0.

    def objective(x):
        # x are the log-parameters or just h? Let's use h directly with penalty for constraints.
        # Better: parameterize h to satisfy constraints.
        # Let h_i = exp(y_i) / sum(exp(y_j)). This ensures h >= 0 and sum(h) = 1.
        y = x
        # Avoid overflow in exp
        y_max = np.max(y)
        e_y = np.exp(y - y_max)
        h = e_y / np.sum(e_y)
        
        l1 = 1.0
        l2_sq = np.sum(h**2)
        ac = np.correlate(h, h, mode='full')
        sup = np.max(ac)
        
        if sup < 1e-15:
            return 0.0
            
        return -l2_sq / sup

    # Initial guess: uniform
    y0 = np.zeros(N)
    
    best_h = None
    best_score = -1e9

    # Try a few random restarts and optimization steps
    n_restarts = 5
    for i in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
            
        if i == 0:
            y_init = y0.copy()
        else:
            # Random initialization
            y_init = np.random.randn(N) * 0.5

        try:
            res = minimize(objective, y_init, method='BFGS', options={'maxiter': 1000, 'gtol': 1e-5})
        except Exception:
            continue

        y_opt = res.x
        # Convert back to h
        y_max = np.max(y_opt)
        e_y = np.exp(y_opt - y_max)
        h_opt = e_y / np.sum(e_y)

        # Evaluate on the actual h (not the normalized one used in optimization, though they are same)
        score = compute_ratio(h_opt)
        
        if score > best_score:
            best_score = score
            best_h = h_opt

        # Also try a simple grid search or specific patterns if time allows
        if time.time() > deadline - 10.0:
            break

    # Fallback: if optimization didn't find anything good, use a heuristic
    # Heuristic: A delta function gives ratio 1.
    # Let's check if we beat 1.0.
    delta_h = np.zeros(N)
    delta_h[0] = 1.0
    delta_score = compute_ratio(delta_h)
    
    if best_h is None or best_score < delta_score:
        best_h = delta_h
        best_score = delta_score

    # One more heuristic: concentrated mass
    # Try putting mass on first k points
    for k in [1, 2, 3, 5, 10]:
        if time.time() > deadline - 2.0:
            break
        h_k = np.zeros(N)
        h_k[:k] = 1.0 / k
        score_k = compute_ratio(h_k)
        if score_k > best_score:
            best_score = score_k
            best_h = h_k

    return list(best_h)
# EVOLVE-BLOCK-END
