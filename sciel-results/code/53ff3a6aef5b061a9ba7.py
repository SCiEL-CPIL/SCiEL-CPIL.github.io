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
    
    # Vectorized type values for faster computation
    # V is (16, 2) where V[i] = [v1, v2]
    V = np.array(types, dtype=float)

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
            probs = np.maximum(probs, 0.0)
            probs = probs / probs.sum()

    # Fallback mechanism (Myerson-style threshold)
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

    if time_exceeded():
        return build_fallback()

    # Try LP-based optimal mechanism
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp

        n_vars = n_types * 3  # x1, x2, p for each type

        # Objective: maximize sum_i probs[i] * p[i]
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[3*i + 2] = -probs[i]
        
        # Build IC and IR constraints using sparse representation
        ic_rows = []
        ic_cols = []
        ic_vals = []
        
        # IC Constraints:
        # u_i(t_i) >= u_i(t_j)
        # v_i[0]*x_i[0] + v_i[1]*x_i[1] - p_i >= v_i[0]*x_j[0] + v_i[1]*x_j[1] - p_j
        # -v_i[0]*x_i[0] - v_i[1]*x_i[1] + v_i[0]*x_j[0] + v_i[1]*x_j[1] + p_i - p_j <= 0
        for i in range(n_types):
            vi0, vi1 = types[i]
            for j in range(n_types):
                if i == j:
                    continue
                row_idx = len(ic_rows)
                # Coefficients for x_i
                if vi0 != 0:
                    ic_rows.append(row_idx); ic_cols.append(3*i + 0); ic_vals.append(-vi0)
                if vi1 != 0:
                    ic_rows.append(row_idx); ic_cols.append(3*i + 1); ic_vals.append(-vi1)
                # Coefficients for x_j
                if vi0 != 0:
                    ic_rows.append(row_idx); ic_cols.append(3*j + 0); ic_vals.append(vi0)
                if vi1 != 0:
                    ic_rows.append(row_idx); ic_cols.append(3*j + 1); ic_vals.append(vi1)
                # Coefficients for p_i and p_j
                ic_rows.append(row_idx); ic_cols.append(3*i + 2); ic_vals.append(1.0)
                ic_rows.append(row_idx); ic_cols.append(3*j + 2); ic_vals.append(-1.0)
        
        # IR Constraints:
        # p_i - v_i[0]*x_i[0] - v_i[1]*x_i[1] <= 0
        ir_rows = []
        ir_cols = []
        ir_vals = []
        for i in range(n_types):
            vi0, vi1 = types[i]
            row_idx = len(ir_rows)
            if vi0 != 0:
                ir_rows.append(row_idx); ir_cols.append(3*i + 0); ir_vals.append(-vi0)
            if vi1 != 0:
                ir_rows.append(row_idx); ir_cols.append(3*i + 1); ir_vals.append(-vi1)
            ir_rows.append(row_idx); ir_cols.append(3*i + 2); ir_vals.append(1.0)
        
        n_constraints = len(ic_rows) + len(ir_rows)
        
        # Combine rows, cols, vals
        all_rows = np.concatenate([ic_rows, ir_rows])
        all_cols = np.concatenate([ic_cols, ir_cols])
        all_vals = np.concatenate([ic_vals, ir_vals])
        
        A_ub = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_constraints, n_vars))
        b_ub = np.zeros(n_constraints)
        
        bounds = []
        for i in range(n_types):
            bounds.append((0.0, 1.0))
            bounds.append((0.0, 1.0))
            bounds.append((0.0, None))
        
        if time_exceeded():
            return build_fallback()

        # Solve LP
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'presolve': True})
        
        if result.success and result.x is not None:
            x = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = max(0.0, min(1.0, x[3*i + 0]))
                allocations[i, 1] = max(0.0, min(1.0, x[3*i + 1]))
                payments[i] = max(0.0, x[3*i + 2])
            
            # Vectorized Validation
            # U[i, j] = v_i[0]*x_j[0] + v_i[1]*x_j[1] - p_j
            # We need U[i, i] >= U[i, j] for all i, j
            # U is (16, 16)
            # V is (16, 2)
            # X is (16, 2)
            # P is (16,)
            
            # Calculate utility of type i when assigned outcome j
            # util[i, j] = dot(V[i], X[j]) - P[j]
            # V @ X.T is (16, 16)
            utils = V @ allocations.T - payments[np.newaxis, :]
            
            # Check IC: utils[i, i] >= utils[i, j] for all j
            # diag(utils) >= row-wise max(utils) - epsilon?
            # Actually, we need utils[i, i] >= utils[i, j] for all j.
            # This is equivalent to: for each row i, the diagonal element is the maximum.
            
            diag_utils = np.diag(utils)
            max_utils = np.max(utils, axis=1)
            
            # Check IR: utils[i, i] >= 0
            # diag_utils >= 0
            
            if np.all(diag_utils >= max_utils - 1e-8) and np.all(diag_utils >= -1e-8):
                return {"allocations": allocations, "payments": payments}
    
    except Exception:
        pass
    
    return build_fallback()
