"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

# EVOLVE-BLOCK-START
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Choose N. N=100 is a good balance.
    N = 100
    step_width = 2.0 / N
    
    # Generate shift candidates (breakpoints)
    # Breakpoints for s in [0, 2]
    # s = 2k/N and s = 1 - 2k/N
    s_candidates = set()
    s_candidates.add(0.0)
    s_candidates.add(1.0)
    s_candidates.add(2.0)
    for k in range(N + 1):
        s1 = 2.0 * k / N
        s2 = 1.0 - 2.0 * k / N
        if 0.0 <= s1 <= 2.0:
            s_candidates.add(s1)
        if 0.0 <= s2 <= 2.0:
            s_candidates.add(s2)
    
    s_vals = sorted(s_candidates)
    num_shifts = len(s_vals)
    
    # Precompute weights W and offsets d
    # O(s) = sum_i W[i, s_idx] * h_i + d[s_idx]
    W = np.zeros((N, num_shifts))
    d = np.zeros(num_shifts)
    
    for idx, s in enumerate(s_vals):
        if s <= 1.0:
            # overlap(s) = ∫₀^{1-s} h dx + ∫_{1+s}² (1-h) dx
            # = sum (m1_i - m2_i) h_i + sum m2_i
            # m1_i = measure(step i ∩ [0, 1-s])
            # m2_i = measure(step i ∩ [1+s, 2])
            for i in range(N):
                step_start = 2.0 * i / N
                step_end = 2.0 * (i + 1) / N
                lo1, hi1 = max(0.0, step_start), min(1.0 - s, step_end)
                lo2, hi2 = max(1.0 + s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                W[i, idx] = m1 - m2
                d[idx] += m2
        else:
            # overlap(s) = ∫₀^{s-1} (1-h) dx + ∫_{2-s}² h dx
            # = sum (-m1_i + m2_i) h_i + sum m1_i
            # m1_i = measure(step i ∩ [0, s-1])
            # m2_i = measure(step i ∩ [2-s, 2])
            for i in range(N):
                step_start = 2.0 * i / N
                step_end = 2.0 * (i + 1) / N
                lo1, hi1 = max(0.0, step_start), min(s - 1.0, step_end)
                lo2, hi2 = max(2.0 - s, step_start), min(2.0, step_end)
                m1 = max(0.0, hi1 - lo1)
                m2 = max(0.0, hi2 - lo2)
                W[i, idx] = -m1 + m2
                d[idx] += m1
                
    # Objective function: Soft-max of overlaps
    # We want to minimize max(O(s)).
    # Soft-max: (1/beta) * log(sum(exp(beta * O)))
    # We can just minimize sum(exp(beta * O)) or log-sum-exp.
    # Log-sum-exp is numerically stable.
    beta = 50.0  # High beta approximates max closely
    
    def objective(h):
        # h is shape (N,)
        O = W.T @ h + d  # shape (num_shifts,)
        # Log-sum-exp
        max_O = np.max(O)
        if max_O > 100: # Prevent overflow
            return max_O
        log_sum = max_O + np.log(np.sum(np.exp(beta * (O - max_O))))
        return log_sum / beta
    
    def gradient(h):
        O = W.T @ h + d
        max_O = np.max(O)
        exp_O = np.exp(beta * (O - max_O))
        sum_exp = np.sum(exp_O)
        softmax_probs = exp_O / sum_exp
        # d/dh (1/beta * log(sum(exp(beta*O))))
        # = 1/beta * sum( exp(beta*O_j) / sum(exp) * beta * dO_j/dh )
        # = sum( softmax_probs_j * dO_j/dh )
        # dO_j/dh = W[:, j]
        grad = W @ softmax_probs
        return grad

    # Initial guess: constant 0.5
    h0 = np.full(N, 0.5)
    
    # Constraints
    # Equality: sum(h) = N/2
    cons = ({'type': 'eq', 'fun': lambda h: np.sum(h) - N/2.0, 'jac': lambda h: np.ones(N)})
    
    # Bounds
    bounds = [(0, 1) for _ in range(N)]
    
    best_h = h0
    best_val = float('inf')
    
    # Try a few random restarts or different betas if time permits
    # But SLSQP can be slow. Let's try one robust run first.
    
    try:
        result = minimize(objective, h0, method='SLSQP', jac=gradient, 
                          bounds=bounds, constraints=cons, 
                          options={'maxiter': 1000, 'ftol': 1e-10})
        if result.success or result.fun < best_val:
            best_h = result.x
            # Calculate actual max overlap for scoring/verification
            O = W.T @ best_h + d
            best_val = np.max(O)
    except Exception:
        pass
    
    # If SLSQP failed or didn't improve, try a simple projected gradient descent on the soft-max
    # This is a structural fallback within the new method family.
    if time.time() < deadline - 10:
        h_curr = best_h.copy()
        lr = 0.01
        for step in range(1000):
            if time.time() > deadline - 5:
                break
            grad = gradient(h_curr)
            h_curr = h_curr - lr * grad
            # Project onto box [0,1]
            h_curr = np.clip(h_curr, 0, 1)
            # Project onto sum constraint (simple adjustment)
            # This is a rough projection, but helps.
            diff = N/2.0 - np.sum(h_curr)
            h_curr += diff / N
            h_curr = np.clip(h_curr, 0, 1)
            
            val = objective(h_curr)
            if val < best_val:
                best_val = val
                best_h = h_curr.copy()
    
    # Final check: ensure sum is close to N/2
    # Adjust slightly if needed
    diff = N/2.0 - np.sum(best_h)
    if abs(diff) > 1e-9:
        # Distribute error
        best_h += diff / N
        best_h = np.clip(best_h, 0, 1)
        
    return best_h

# EVOLVE-BLOCK-END
