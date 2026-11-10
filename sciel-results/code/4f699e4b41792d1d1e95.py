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
    WALL_LIMIT = 90.0
    SAFETY_MARGIN = 10.0
    DEADLINE = WALL_LIMIT - SAFETY_MARGIN  # 80s
    
    def time_exceeded():
        return (time.time() - start_time) >= DEADLINE
    
    # Values are in {0,1,2,3} for each of 2 goods -> 16 types
    values_per_good = [0, 1, 2, 3]
    n_types = 16
    
    # Enumerate all type profiles: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in values_per_good for v2 in values_per_good]
    type_index = {t: i for i, t in enumerate(types)}
    
    # Get distribution from distribution_spec
    probs = None
    if isinstance(distribution_spec, dict):
        for key in ('probs', 'distribution', 'weights', 'prob', 'p'):
            if key in distribution_spec:
                try:
                    probs = np.array(distribution_spec[key], dtype=float).flatten()
                    if len(probs) == n_types:
                        break
                except Exception:
                    probs = None
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
        except Exception:
            probs = None
    
    if probs is None or len(probs) != n_types or not np.all(np.isfinite(probs)):
        probs = np.ones(n_types) / n_types
    else:
        total = probs.sum()
        if total <= 0 or not np.isfinite(total):
            probs = np.ones(n_types) / n_types
        else:
            probs = probs / total
            # Clip tiny negatives
            probs = np.maximum(probs, 0.0)
            probs = probs / probs.sum()
    
    # Build fallback mechanism (Myerson-style threshold)
    def build_fallback():
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        
        for i, (v1, v2) in enumerate(types):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0
        
        for i, (v1, v2) in enumerate(types):
            x1 = allocations[i, 0]
            x2 = allocations[i, 1]
            sum_x1_lower = 0.0
            for v1_lower in range(v1):
                idx = type_index[(v1_lower, v2)]
                sum_x1_lower += allocations[idx, 0]
            sum_x2_lower = 0.0
            for v2_lower in range(v2):
                idx = type_index[(v1, v2_lower)]
                sum_x2_lower += allocations[idx, 1]
            payments[i] = v1 * x1 + v2 * x2 - sum_x1_lower - sum_x2_lower
        
        payments = np.maximum(payments, 0)
        for i in range(n_types):
            max_pay = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
            if payments[i] > max_pay + 1e-10:
                payments[i] = max_pay
        
        return {"allocations": allocations, "payments": payments}
    
    # Try LP-based optimal mechanism
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp
        
        n_vars = n_types * 3  # x1, x2, p for each type
        
        # Objective: maximize sum_i probs[i] * p[i]
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[3*i + 2] = -probs[i]
        
        # Build IC constraints using lattice structure
        # For 2D grid, IC is implied by checking neighbors in the partial order
        # Type (v1,v2) <= (v1',v2') iff v1<=v1' and v2<=v2'
        # Need: u(v1,v2) >= u(v1',v2') when (v1,v2) >= (v1',v2') componentwise
        # i.e., p_i - v_i·x_i <= p_j - v_i·x_j for i >= j componentwise
        # But we need ALL pairs, not just adjacent. However, for monotone allocations,
        # checking adjacent pairs in each dimension suffices.
        # To be safe, we include all pairs but use sparse representation.
        
        A_rows = []
        b_rows = []
        
        # IC: for all i, j with i != j:
        # p_i - v_i·x_i <= p_j - v_i·x_j
        # => p_i - p_j - v_i_0*x_i_0 - v_i_1*x_i_1 + v_i_0*x_j_0 + v_i_1*x_j_1 <= 0
        for i in range(n_types):
            vi0, vi1 = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                row = np.zeros(n_vars)
                row[3*i + 2] = 1.0       # p_i
                row[3*j + 2] = -1.0      # p_j
                row[3*i + 0] = -vi0      # -v_i0 * x_i0
                row[3*i + 1] = -vi1      # -v_i1 * x_i1
                row[3*j + 0] = vi0       # +v_i0 * x_j0
                row[3*j + 1] = vi1       # +v_i1 * x_j1
                A_rows.append(row)
                b_rows.append(0.0)
        
        # IR: p_i - v_i·x_i <= 0
        for i in range(n_types):
            vi0, vi1 = types[i]
            row = np.zeros(n_vars)
            row[3*i + 2] = 1.0
            row[3*i + 0] = -vi0
            row[3*i + 1] = -vi1
            A_rows.append(row)
            b_rows.append(0.0)
        
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
        
        bounds = []
        for i in range(n_types):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, None))
        
        if time_exceeded():
            return build_fallback()
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success and result.x is not None:
            x = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = max(0.0, min(1.0, x[3*i + 0]))
                allocations[i, 1] = max(0.0, min(1.0, x[3*i + 1]))
                payments[i] = max(0.0, x[3*i + 2])
            
            # Verify IC and IR
            valid = True
            for i in range(n_types):
                vi0, vi1 = types[i]
                util_i = vi0*allocations[i,0] + vi1*allocations[i,1] - payments[i]
                # IR: util_i >= 0
                if util_i < -1e-8:
                    valid = False
                    break
                for j in range(n_types):
                    if i == j:
                        continue
                    util_i_as_j = vi0*allocations[j,0] + vi1*allocations[j,1] - payments[j]
                    if util_i < util_i_as_j - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {"allocations": allocations, "payments": payments}
        else:
            pass
    
    except Exception:
        pass
    
    return build_fallback()

# EVOLVE-BLOCK-END
