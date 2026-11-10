"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: specification of the type distribution
        support: support of types (values in {0,1,2,3} for each good)
        constraints: additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    # 85s deadline to be safe against the 90s kill
    DEADLINE = 85.0
    
    # Values are in {0,1,2,3} for each of 2 goods -> 16 types
    values_per_good = [0, 1, 2, 3]
    n_types = 16
    
    # Enumerate all type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in values_per_good for v2 in values_per_good]
    type_index = {t: i for i, t in enumerate(types)}
    
    # Get distribution from distribution_spec
    probs = None
    if isinstance(distribution_spec, dict):
        if 'probs' in distribution_spec:
            probs = np.array(distribution_spec['probs'], dtype=float)
        elif 'distribution' in distribution_spec:
            probs = np.array(distribution_spec['distribution'], dtype=float)
        elif 'weights' in distribution_spec:
            probs = np.array(distribution_spec['weights'], dtype=float)
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        probs = np.array(distribution_spec, dtype=float)
    
    if probs is None or len(probs) != n_types:
        # Default: uniform distribution
        probs = np.ones(n_types) / n_types
    else:
        # Normalize
        p_sum = probs.sum()
        if p_sum > 0:
            probs = probs / p_sum
        else:
            probs = np.ones(n_types) / n_types
            
    # Ensure non-negative probabilities
    probs = np.maximum(probs, 0)
    probs = probs / probs.sum()

    def is_valid_mechanism(allocations, payments):
        """Check if the mechanism satisfies IC, IR, and non-negativity."""
        if np.any(allocations < -1e-9) or np.any(allocations > 1 + 1e-9):
            return False
        if np.any(payments < -1e-9):
            return False
            
        # Check IR: p_i <= v_i . x_i
        for i in range(n_types):
            v_val = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
            if payments[i] > v_val + 1e-8:
                return False
                
        # Check IC: u_i(i) >= u_i(i')
        for i in range(n_types):
            v_i = types[i]
            util_true = v_i[0]*allocations[i,0] + v_i[1]*allocations[i,1] - payments[i]
            for ip in range(n_types):
                if i == ip: continue
                util_report = v_i[0]*allocations[ip,0] + v_i[1]*allocations[ip,1] - payments[ip]
                if util_true < util_report - 1e-8:
                    return False
        return True

    def compute_payments_from_allocations(allocations):
        """
        Compute payments using the discrete envelope theorem.
        p(v) = v.x(v) - sum_{v'<v} (x(v) - x(v')) * (v - v')? 
        Actually, for discrete types, a valid payment for monotone allocations is:
        p(v) = v.x(v) - sum_{v' < v} (v - v').(x(v) - x(v'))
        This ensures IC. We must check IR.
        """
        payments = np.zeros(n_types)
        for i in range(n_types):
            v_i = types[i]
            x_i = allocations[i]
            val = v_i[0]*x_i[0] + v_i[1]*x_i[1]
            subtract = 0.0
            for ip in range(n_types):
                v_ip = types[ip]
                # Check if v_ip is strictly less than v_i in some sense
                # For additive values, we can use the path integral approach or simply:
                # p(v) = v.x(v) - \sum_{v' \in S(v)} (v-v') . (x(v)-x(v'))
                # Where S(v) are types "below" v.
                # A simpler robust method for small support:
                # p(v) = min over all v' such that v' <= v component-wise? No, IC is global.
                
                # Let's use the standard discrete formula:
                # p(v) = v.x(v) - \sum_{v' \neq v} \max(0, (v-v') \cdot (x(v)-x(v')))
                # This is not quite right.
                
                # Correct discrete payment for IC:
                # p(v) = v.x(v) - \sum_{v' < v} (v - v') \cdot (x(v) - x(v'))
                # Where v' < v means v'_j <= v_j for all j and v' != v.
                if all(v_ip[j] <= v_i[j] for j in range(2)) and v_ip != v_i:
                    diff_v = np.array([v_i[0]-v_ip[0], v_i[1]-v_ip[1]])
                    diff_x = np.array([x_i[0]-allocations[ip][0], x_i[1]-allocations[ip][1]])
                    subtract += np.dot(diff_v, diff_x)
            payments[i] = val - subtract
        return payments

    # Try to solve the LP for the optimal mechanism
    try:
        from scipy.optimize import linprog
        
        # Decision variables: [x_0_0, x_0_1, p_0, x_1_0, x_1_1, p_1, ...]
        # Index: for type i, goods at 3*i, 3*i+1, payment at 3*i+2
        n_vars = n_types * 3
        
        # Objective: maximize sum_i probs[i] * p[i] -> minimize -sum_i probs[i] * p[i]
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[3*i + 2] = -probs[i]
        
        # Inequality constraints: A_ub @ x <= b_ub
        A_rows = []
        b_rows = []
        
        # IC constraints: for each pair (i, i'), 
        # p_i - v_i·x_i <= p_i' - v_i·x_i'
        # p_i - p_i' - v_i·x_i + v_i·x_i' <= 0
        for i in range(n_types):
            for ip in range(n_types):
                if i == ip:
                    continue
                row = np.zeros(n_vars)
                row[3*i + 2] = 1.0
                row[3*ip + 2] = -1.0
                row[3*i + 0] = -types[i][0]
                row[3*i + 1] = -types[i][1]
                row[3*ip + 0] = types[i][0]
                row[3*ip + 1] = types[i][1]
                A_rows.append(row)
                b_rows.append(0.0)
        
        # IR constraints: p_i - v_i·x_i <= 0
        for i in range(n_types):
            row = np.zeros(n_vars)
            row[3*i + 2] = 1.0
            row[3*i + 0] = -types[i][0]
            row[3*i + 1] = -types[i][1]
            A_rows.append(row)
            b_rows.append(0.0)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        # Bounds: x in [0,1], p in [0, inf)
        bounds = []
        for i in range(n_types):
            bounds.append((0, 1))  # x_i_0
            bounds.append((0, 1))  # x_i_1
            bounds.append((0, None))  # p_i
        
        # Check time before solving
        if time.time() > DEADLINE - 5:
            raise TimeoutError("Not enough time for LP")

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = x[3*i + 0]
                allocations[i, 1] = x[3*i + 1]
                payments[i] = x[3*i + 2]
            
            # Post-process for numerical stability
            allocations = np.clip(allocations, 0, 1)
            payments = np.maximum(payments, 0)
            
            # Check validity
            if is_valid_mechanism(allocations, payments):
                return {"allocations": allocations, "payments": payments}
            else:
                # If LP solution is slightly invalid due to floating point, try to repair
                # by recalculating payments from allocations if they are monotone
                # But first, check if allocations are monotone
                monotone = True
                for i in range(n_types):
                    for ip in range(n_types):
                        if all(types[ip][j] <= types[i][j] for j in range(2)) and types[ip] != types[i]:
                            if allocations[ip][0] > allocations[i][0] + 1e-8 or allocations[ip][1] > allocations[i][1] + 1e-8:
                                monotone = False
                                break
                    if not monotone:
                        break
                
                if monotone:
                    new_payments = compute_payments_from_allocations(allocations)
                    new_payments = np.maximum(new_payments, 0)
                    # Check IR
                    valid = True
                    for i in range(n_types):
                        v_val = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
                        if new_payments[i] > v_val + 1e-8:
                            valid = False
                            break
                    if valid and is_valid_mechanism(allocations, new_payments):
                        return {"allocations": allocations, "payments": new_payments}
                
                # If repair failed, fall through to fallback
                pass
        else:
            pass
            
    except Exception:
        pass

    # Fallback: Simple monotone mechanism
    # Allocate good j if value >= 2
    allocations = np.zeros((n_types, 2))
    for i, (v1, v2) in enumerate(types):
        if v1 >= 2:
            allocations[i, 0] = 1.0
        if v2 >= 2:
            allocations[i, 1] = 1.0
            
    payments = compute_payments_from_allocations(allocations)
    payments = np.maximum(payments, 0)
    
    # Ensure IR
    for i in range(n_types):
        v_val = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
        if payments[i] > v_val:
            payments[i] = v_val
            
    return {"allocations": allocations, "payments": payments}

# EVOLVE-BLOCK-END
