"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: Specification of the value distribution
        support: Support of the value space
        constraints: Additional constraints
        
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 45.0  # Safety margin for 60s limit

    values_per_good = [0, 1, 2, 3]
    all_reports = list(product(values_per_good, values_per_good))
    n_types = 16
    
    # Parse distribution_spec to get probabilities
    if isinstance(distribution_spec, dict):
        probs = np.array([distribution_spec.get(r, 0.0) for r in all_reports])
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
        if len(probs) == 16:
            pass
        else:
            probs = np.array([0.0 for _ in range(16)])
            if isinstance(distribution_spec, dict):
                probs = np.array([distribution_spec.get(r, 0.0) for r in all_reports])
    else:
        probs = np.ones(16) / 16.0
        
    # Normalize probabilities
    total_prob = probs.sum()
    if total_prob > 0:
        probs = probs / total_prob
    else:
        probs = np.ones(16) / 16.0

    def solve_1d_lp(probs_1d):
        """
        Solves the LP for a single good with values 0..3.
        Variables: x_0, x_1, x_2, x_3, p_0, p_1, p_2, p_3
        Objective: Max sum p_i * pr_i
        Constraints:
        - 0 <= x_i <= 1
        - p_i >= 0
        - IC (monotonicity): p_i - v_i x_i >= p_j - v_i x_j for all i, j
        - IR: p_i - v_i x_i <= 0 (since p_i >= 0 and v_i x_i >= 0, IR is p_i <= v_i x_i)
        """
        v = np.array([0, 1, 2, 3], dtype=float)
        n = 4
        n_vars = 8
        
        # Indices
        # x_i: i
        # p_i: 4 + i
        
        c = np.zeros(n_vars)
        for i in range(n):
            c[4 + i] = -probs_1d[i]
            
        A_ub = []
        b_ub = []
        
        # IC Constraints: p_i - v_i x_i >= p_j - v_i x_j
        # => p_i - p_j - v_i x_i + v_i x_j <= 0
        for i in range(n):
            for j in range(n):
                if i == j: continue
                row = np.zeros(n_vars)
                row[4 + i] = 1.0
                row[4 + j] = -1.0
                row[i] = -v[i]
                row[j] = v[i]
                A_ub.append(row)
                b_ub.append(0.0)
                
        # IR Constraints: p_i - v_i x_i <= 0
        for i in range(n):
            row = np.zeros(n_vars)
            row[4 + i] = 1.0
            row[i] = -v[i]
            A_ub.append(row)
            b_ub.append(0.0)
            
        bounds = [(0, 1)] * 4 + [(0, None)] * 4
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x[:4], res.x[4:]
        except Exception:
            pass
        return None

    # Pre-compute fallback (fast, deterministic)
    # Threshold mechanisms are IC and IR.
    # We can just pick a reasonable one or the best one from a small set.
    # For speed, we will just use a simple threshold or the LP result.
    # Let's try to solve the LPs.
    
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    success = True
    
    for g in range(2):
        if time.time() > deadline:
            success = False
            break
            
        # Extract 1D probabilities for good g
        # Reports are (v0, v1). Good 0 is v0, Good 1 is v1.
        probs_1d = np.zeros(4)
        for i, (v0, v1) in enumerate(all_reports):
            if g == 0:
                probs_1d[v0] += probs[i]
            else:
                probs_1d[v1] += probs[i]
                
        sol = solve_1d_lp(probs_1d)
        if sol is None:
            success = False
            break
            
        x_vals, p_vals = sol
        # x_vals[0] is alloc for v=0, x_vals[1] for v=1, etc.
        # p_vals[0] is payment for v=0, etc.
        
        for i, (v0, v1) in enumerate(all_reports):
            v_g = v0 if g == 0 else v1
            allocations[i, g] = x_vals[v_g]
            payments[i] += p_vals[v_g]
            
    if not success or time.time() > deadline:
        # Fallback: Simple threshold mechanism
        # This is a safe fallback that is always IC/IR.
        # We'll use thresholds (1, 1) or similar, or just zeros if we are very late.
        # Actually, the seed program's fallback loop was expensive. 
        # A static threshold is better for safety.
        # Let's use threshold 1 for both.
        for i, (v0, v1) in enumerate(all_reports):
            alloc0 = 1.0 if v0 >= 1 else 0.0
            alloc1 = 1.0 if v1 >= 1 else 0.0
            allocations[i, 0] = alloc0
            allocations[i, 1] = alloc1
            payments[i] = (1.0 if v0 >= 1 else 0.0) + (1.0 if v1 >= 1 else 0.0)
        
        # If even that is too late, return zeros
        if time.time() > deadline:
             return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
            
    # Final validation check (fast)
    # IC: v(t).x(t) - p(t) >= v(t).x(t') - p(t')
    # IR: v(t).x(t) - p(t) >= 0
    # Non-neg: p(t) >= 0
    
    max_regret = 0.0
    min_ir = np.inf
    min_pay = np.inf
    
    for t in range(16):
        v_t = np.array(all_reports[t], dtype=float)
        util_t = v_t @ allocations[t] - payments[t]
        
        if util_t < min_ir:
            min_ir = util_t
        if payments[t] < min_pay:
            min_pay = payments[t]
            
        for tp in range(16):
            util_tp_as_t = v_t @ allocations[tp] - payments[tp]
            regret = util_tp_as_t - util_t
            if regret > max_regret:
                max_regret = regret
                
    if max_regret > 1e-8 or min_ir < -1e-8 or min_pay < -1e-8:
        # Fallback to safe threshold if LP produced numerical errors
        for i, (v0, v1) in enumerate(all_reports):
            alloc0 = 1.0 if v0 >= 1 else 0.0
            alloc1 = 1.0 if v1 >= 1 else 0.0
            allocations[i, 0] = alloc0
            allocations[i, 1] = alloc1
            payments[i] = (1.0 if v0 >= 1 else 0.0) + (1.0 if v1 >= 1 else 0.0)

    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
