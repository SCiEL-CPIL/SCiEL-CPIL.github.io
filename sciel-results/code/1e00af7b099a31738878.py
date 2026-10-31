import numpy as np
import time
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0

def _compute_max_overlap(h):
    """Compute max circular overlap of h with shifted 1-h."""
    n = len(h)
    # O_k = (1/N) * sum_{i=0}^{N-1} h[(i+k) % N] * (1 - h[i])
    # Using FFT for speed? N=100 is small, direct computation is fine.
    # Vectorized computation for all k
    # Create matrix where row k is roll(h, -k)
    # M[k, i] = h[(i+k) % N]
    # O_k = sum_i M[k, i] * (1 - h[i]) / N
    
    # Efficient way:
    # corr = np.correlate(h, 1-h, mode='full') ? No, circular.
    
    # For N=100, a loop is fine.
    max_ov = 0.0
    one_minus_h = 1.0 - h
    for k in range(n):
        shifted = np.roll(h, -k)
        ov = np.dot(shifted, one_minus_h) / n
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _soft_max_overlap(h, beta):
    """
    Smooth approximation of max_k O_k using log-sum-exp.
    J(h) = (1/beta) * log(sum(exp(beta * O_k)))
    """
    n = len(h)
    one_minus_h = 1.0 - h
    overlaps = np.zeros(n)
    for k in range(n):
        shifted = np.roll(h, -k)
        overlaps[k] = np.dot(shifted, one_minus_h) / n
    
    # Scale for numerical stability
    max_ov = np.max(overlaps)
    if max_ov == 0:
        return 0.0
        
    # log-sum-exp trick
    # log(sum(exp(beta * O_k))) = beta * max_ov + log(sum(exp(beta * (O_k - max_ov))))
    scaled = beta * (overlaps - max_ov)
    # Clip to avoid underflow/overflow issues, though exp(-large) is 0
    log_sum = np.log(np.sum(np.exp(scaled)))
    return (max_ov + log_sum / beta)

def _objective(x, beta):
    """
    Objective function for minimization.
    x is the vector of heights.
    """
    h = x
    # Penalty for bounds if SLSQP fails to enforce perfectly, but SLSQP should handle it.
    # We rely on SLSQP constraints.
    return _soft_max_overlap(h, beta)

def _constraints():
    """
    Define constraints for SLSQP.
    """
    # Equality constraint: sum(h) = N/2
    # Inequality constraints: 0 <= h_i <= 1 are handled by bounds in SLSQP.
    
    def sum_constraint(x):
        return np.sum(x) - TARGET_SUM
    
    return {
        'type': 'eq',
        'fun': sum_constraint
    }

def construct():
    """
    Returns a sequence of N=100 step heights in [0, 1] whose sum is exactly N/2 = 50.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_h = np.full(N, 0.5)
    best_score = _compute_max_overlap(best_h)
    
    # Strategy: Try multiple random initializations and optimize.
    # Mix of binary and continuous starts.
    
    np.random.seed(42)
    
    num_starts = 10
    beta = 100.0  # Temperature for soft-max. Higher is closer to hard max.
    
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Generate initial guess
        if i % 2 == 0:
            # Binary start
            h0 = np.zeros(N)
            indices = np.random.choice(N, TARGET_SUM, replace=False)
            h0[indices] = 1.0
        else:
            # Continuous start, centered around 0.5
            h0 = np.full(N, 0.5) + np.random.randn(N) * 0.1
            h0 = np.clip(h0, 0.0, 1.0)
            # Normalize sum
            s = np.sum(h0)
            if s > 0:
                h0 = h0 / s * TARGET_SUM
            h0 = np.clip(h0, 0.0, 1.0)
            # Adjust sum exactly
            diff = TARGET_SUM - np.sum(h0)
            h0 += diff / N
            h0 = np.clip(h0, 0.0, 1.0)
        
        # Ensure exact sum for the optimizer start
        # SLSQP might struggle if start is not feasible.
        # Let's project h0 to the hyperplane sum=N/2.
        s = np.sum(h0)
        if abs(s - TARGET_SUM) > 1e-9:
            h0 = h0 + (TARGET_SUM - s) / N
            h0 = np.clip(h0, 0.0, 1.0)
        
        try:
            res = minimize(
                _objective, 
                h0, 
                args=(beta,),
                method='SLSQP', 
                bounds=[(0, 1)] * N, 
                constraints=[_constraints()],
                options={'maxiter': 200, 'ftol': 1e-12}
            )
            
            if res.success or res.fun < best_score:
                h_opt = res.x
                # Ensure feasibility
                h_opt = np.clip(h_opt, 0.0, 1.0)
                # Adjust sum exactly
                diff = TARGET_SUM - np.sum(h_opt)
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
                
                score = _compute_max_overlap(h_opt)
                if score < best_score:
                    best_h = h_opt.copy()
                    best_score = score
                    
        except Exception:
            continue
            
    # Final polish: Try to improve by pushing towards binary values
    # If the solution is close to binary, rounding might improve it.
    # Also, try a few random binary local searches if time permits.
    
    if time.time() < deadline - 5:
        # Local search on the best solution found
        # If best_h is close to binary, try rounding
        h_binary = (best_h > 0.5).astype(float)
        # Adjust sum if rounding changed it
        # Count ones
        num_ones = np.sum(h_binary)
        if num_ones != TARGET_SUM:
            # This simple rounding might not preserve sum.
            # Skip if it doesn't preserve sum easily, or fix it.
            # For now, just check if the continuous best is already good.
            pass
        else:
            score_bin = _compute_max_overlap(h_binary)
            if score_bin < best_score:
                best_h = h_binary
                best_score = score_bin

    return best_h

