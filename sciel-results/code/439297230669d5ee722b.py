import numpy as np
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction.
    
    Strategy: Direct LP formulation with all IC constraints (redundancy is cheap for 16 types)
    using HiGHS solver. This is robust and fast.
    """
    start_time = time.time()
    
    # 1. Parse Input
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        else:
            # Fallback: try to find a list of floats
            vals = list(distribution_spec.values())
            for v in vals:
                if isinstance(v, (list, np.ndarray)):
                    probs = np.array(v, dtype=float)
                    break
            else:
                probs = np.ones(16) / 16.0
    else:
        probs = np.array(distribution_spec, dtype=float)
        
    # Ensure shape is (16,)
    if probs.shape != (16,):
        # Flatten if needed
        probs = probs.flatten()
        if len(probs) != 16:
            # If still wrong, pad or trim? Assume standard 16.
            # For safety, if length is not 16, we can't proceed correctly.
            # But task says 16 types.
            if len(probs) > 16:
                probs = probs[:16]
            else:
                new_probs = np.zeros(16)
                new_probs[:len(probs)] = probs
                probs = new_probs

    # Normalize probabilities
    total_p = np.sum(probs)
    if total_p > 1e-12:
        probs = probs / total_p
    else:
        probs = np.ones(16) / 16.0

    # 2. Define Grid
    # Values in {0, 1, 2, 3} for 2 goods.
    # Index i = v1 * 4 + v2
    grid_values = []
    for v1 in range(4):
        for v2 in range(4):
            grid_values.append((v1, v2))
    
    n_types = 16
    n_vars = 48  # 16 x1, 16 x2, 16 p

    # 3. Formulate LP
    # Maximize Revenue = sum(p_i * probs_i)
    # Minimize -sum(p_i * probs_i)
    c = np.zeros(n_vars)
    for i in range(16):
        c[32 + i] = -probs[i]
    
    A_rows = []
    b_rows = []
    
    # IR Constraints: v1*x1_i + v2*x2_i - p_i >= 0
    # => -v1*x1_i - v2*x2_i + p_i <= 0
    for i in range(16):
        v1, v2 = grid_values[i]
        row = np.zeros(n_vars)
        row[i] = -v1
        row[16 + i] = -v2
        row[32 + i] = 1
        A_rows.append(row)
        b_rows.append(0)
        
    # IC Constraints: u(i) >= u(j) + (v_i - v_j).x_j
    # u(i) = v_i.x_i - p_i
    # v_i.x_i - p_i >= v_j.x_j - p_j + (v_i - v_j).x_j
    # v_i.x_i - p_i >= v_i.x_j - p_j
    # v_i.(x_i - x_j) - (p_i - p_j) >= 0
    # => -v_i.x_i + v_i.x_j + p_i - p_j <= 0
    
    for i in range(16):
        v1_i, v2_i = grid_values[i]
        for j in range(16):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -v_i . x_i
            row[i] = -v1_i
            row[16 + i] = -v2_i
            # + v_i . x_j
            row[j] = v1_i
            row[16 + j] = v2_i
            # + p_i
            row[32 + i] = 1
            # - p_j
            row[32 + j] = -1
            
            A_rows.append(row)
            b_rows.append(0)
            
    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)
    
    # Bounds
    # x in [0, 1]
    # p in [0, inf)
    bounds = []
    for i in range(32):
        bounds.append((0, 1))
    for i in range(16):
        bounds.append((0, None))
        
    # 4. Solve
    from scipy.optimize import linprog
    
    try:
        # Set time limit to 80s (safety margin from 90s kill)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', 
                     options={'presolve': True, 'time_limit': 80})
        
        if res.success:
            x = res.x
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            
            for i in range(16):
                allocations[i, 0] = x[i]
                allocations[i, 1] = x[16 + i]
                payments[i] = x[32 + i]
            
            # Clip to handle numerical noise
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            return {
                "allocations": allocations,
                "payments": payments
            }
        else:
            # Fallback if LP fails
            allocations = np.zeros((16, 2))
            payments = np.zeros(16)
            return {
                "allocations": allocations,
                "payments": payments
            }
    except Exception:
        allocations = np.zeros((16, 2))
        payments = np.zeros(16)
        return {
            "allocations": allocations,
            "payments": payments
        }

# EVOLVE-BLOCK-END
