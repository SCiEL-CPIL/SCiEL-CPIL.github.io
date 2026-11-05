"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # --- Parse distribution_spec into joint probability over (v1, v2) in {0,1,2,3}^2 ---
    p_joint = np.ones((4, 4)) / 16.0  # default uniform
    
    try:
        if isinstance(distribution_spec, (list, tuple, np.ndarray)):
            arr = np.array(distribution_spec, dtype=float)
            if arr.size == 4:
                # Marginal distribution, assume independence
                p_joint = np.outer(arr, arr)
            elif arr.size == 16:
                # Joint distribution
                p_joint = arr.reshape(4, 4)
            else:
                p_joint = np.ones((4, 4)) / 16.0
        elif isinstance(distribution_spec, dict):
            # Check if keys are tuples (joint) or ints (marginal)
            sample_key = next(iter(distribution_spec.keys()), None)
            if sample_key is not None and isinstance(sample_key, tuple) and len(sample_key) == 2:
                p_joint = np.zeros((4, 4))
                for k, v in distribution_spec.items():
                    if isinstance(k, tuple) and len(k) == 2:
                        p_joint[k[0], k[1]] = v
            else:
                marginal = np.zeros(4)
                for k, v in distribution_spec.items():
                    if isinstance(k, (int, np.integer)) and 0 <= k < 4:
                        marginal[k] = v
                p_joint = np.outer(marginal, marginal)
        else:
            p_joint = np.ones((4, 4)) / 16.0
    except Exception:
        p_joint = np.ones((4, 4)) / 16.0
    
    # Normalize
    total = p_joint.sum()
    if total > 0:
        p_joint = p_joint / total
    else:
        p_joint = np.ones((4, 4)) / 16.0

    # --- Determine support (types where IC/IR must hold) ---
    all_types = [(i, j) for i in range(4) for j in range(4)]
    
    if support is None:
        supported_types = all_types
    elif isinstance(support, (list, tuple, set)):
        supported_types = [tuple(r) for r in support]
    elif isinstance(support, np.ndarray):
        supported_types = [tuple(r) for r in support]
    else:
        supported_types = all_types
    
    # Ensure all supported types are valid
    supported_types = [t for t in supported_types if t in [(i, j) for i in range(4) for j in range(4)]]
    if not supported_types:
        supported_types = all_types
    
    # Type index mapping
    type_index = {t: i for i, t in enumerate(all_types)}
    supported_indices = [type_index[t] for t in supported_types]
    
    # --- Build LP: maximize expected revenue ---
    # Variables: x = [alloc_00, alloc_01, ..., alloc_15_1, pay_0, ..., pay_15]
    # 32 allocation vars + 16 payment vars = 48 total
    
    n_vars = 48
    
    # Objective: maximize sum_t p[t] * pay[t] => minimize -sum_t p[t] * pay[t]
    c = np.zeros(n_vars)
    for i in range(16):
        v1, v2 = i // 4, i % 4
        c[32 + i] = -p_joint[v1, v2]
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IR constraints: for each supported type t, payment(t) <= alloc[t,0]*t[0] + alloc[t,1]*t[1]
    # => payment(t) - alloc[t,0]*t[0] - alloc[t,1]*t[1] <= 0
    for t in supported_types:
        ti = type_index[t]
        row = np.zeros(n_vars)
        row[32 + ti] = 1
        row[ti * 2 + 0] = -t[0]
        row[ti * 2 + 1] = -t[1]
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    # IC constraints: for each pair (t, r) in supported_types x supported_types
    # Utility(t, t) >= Utility(t, r)
    # sum_g alloc[t,g]*t[g] - pay[t] >= sum_g alloc[r,g]*t[g] - pay[r]
    # pay[r] - pay[t] - sum_g (alloc[r,g] - alloc[t,g])*t[g] <= 0
    for t in supported_types:
        ti = type_index[t]
        for r in supported_types:
            ri = type_index[r]
            if ti == ri:
                continue  # Skip trivial case
            row = np.zeros(n_vars)
            # pay[r]
            row[32 + ri] = 1
            # -pay[t]
            row[32 + ti] = -1
            # -alloc[r,0]*t[0] + alloc[t,0]*t[0]
            row[ri * 2 + 0] = -t[0]
            row[ti * 2 + 0] = t[0]
            # -alloc[r,1]*t[1] + alloc[t,1]*t[1]
            row[ri * 2 + 1] = -t[1]
            row[ti * 2 + 1] = t[1]
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows) if A_ub_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_ub_vals) if b_ub_vals else np.zeros(0)
    
    # Bounds: alloc in [0,1], pay in [0, inf)
    bounds = [(0, 1)] * 32 + [(0, None)] * 16
    
    # Solve LP
    alloc = np.zeros((16, 2))
    pay = np.zeros(16)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.x is not None:
            alloc = res.x[:32].reshape(16, 2)
            pay = res.x[32:]
            # Clip to valid range
            alloc = np.clip(alloc, 0, 1)
            pay = np.maximum(pay, 0)
        else:
            raise Exception("LP failed")
    except Exception:
        # Fallback: simple threshold mechanism
        for i in range(4):
            for j in range(4):
                idx = i * 4 + j
                if i >= 2:
                    alloc[idx, 0] = 1.0
                    pay[idx] += 2.0
                if j >= 2:
                    alloc[idx, 1] = 1.0
                    pay[idx] += 2.0
    
    # --- Post-LP verification and correction ---
    # Check regret: for all t, r in supported_types, Utility(t,t) >= Utility(t,r) - 1e-8
    # If violated, adjust payments slightly
    
    def utility(t, r, alloc_mat, pay_vec):
        """Utility of true type t when reporting r"""
        ti = type_index[t]
        ri = type_index[r]
        return alloc_mat[ri, 0] * t[0] + alloc_mat[ri, 1] * t[1] - pay_vec[ri]
    
    max_regret = 0.0
    for t in supported_types:
        for r in supported_types:
            regret = utility(t, r, alloc, pay) - utility(t, t, alloc, pay)
            if regret > max_regret:
                max_regret = regret
    
    # If regret exceeds tolerance, try to fix by increasing payments for over-allocated reports
    if max_regret > 1e-8:
        # Simple correction: for each violating pair, increase payment(r) to fix
        # This may break other constraints, so iterate a few times
        for _ in range(5):
            if max_regret <= 1e-8:
                break
            violated = False
            for t in supported_types:
                for r in supported_types:
                    regret = utility(t, r, alloc, pay) - utility(t, t, alloc, pay)
                    if regret > 1e-8:
                        violated = True
                        # Increase pay[r] by regret + small epsilon
                        ri = type_index[r]
                        pay[ri] += regret + 1e-9
            if not violated:
                break
        
        # Re-check IR after corrections
        for t in supported_types:
            ti = type_index[t]
            ir_violation = pay[ti] - (alloc[ti, 0] * t[0] + alloc[ti, 1] * t[1])
            if ir_violation > 1e-8:
                # Reduce payment to satisfy IR
                pay[ti] = alloc[ti, 0] * t[0] + alloc[ti, 1] * t[1]
        
        # Final clip
        pay = np.maximum(pay, 0)
        alloc = np.clip(alloc, 0, 1)
    
    # Check deadline
    if time.time() > deadline:
        return {"allocations": alloc, "payments": pay}
    
    return {"allocations": alloc, "payments": pay}

# EVOLVE-BLOCK-END
