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

    def verify_mechanism(allocations, payments):
        """
        Verify IC and IR constraints.
        Returns True if valid, False otherwise.
        """
        tol = 1e-8
        for i in range(n_types):
            vi0, vi1 = types[i]
            util_i = vi0*allocations[i,0] + vi1*allocations[i,1] - payments[i]
            
            # IR: util_i >= 0
            if util_i < -tol:
                return False
            
            # Non-negative payments
            if payments[i] < -tol:
                return False

            for j in range(n_types):
                if i == j:
                    continue
                util_i_as_j = vi0*allocations[j,0] + vi1*allocations[j,1] - payments[j]
                if util_i < util_i_as_j - tol:
                    return False
        return True

    def solve_joint_lp():
        """
        Solve the LP for allocations and payments jointly.
        """
        try:
            from scipy.optimize import linprog
            import scipy.sparse as sp

            n_vars = n_types * 3  # x1, x2, p for each type

            # Objective: maximize sum_i probs[i] * p[i]
            c = np.zeros(n_vars)
            for i in range(n_types):
                c[3*i + 2] = -probs[i]
            
            # Build IC constraints
            # u_i(t_i) >= u_i(t_j)
            # v_i[0]*x_i[0] + v_i[1]*x_i[1] - p_i >= v_i[0]*x_j[0] + v_i[1]*x_j[1] - p_j
            # -v_i[0]*x_i[0] - v_i[1]*x_i[1] + v_i[0]*x_j[0] + v_i[1]*x_j[1] + p_i - p_j <= 0
            ic_rows = []
            ic_cols = []
            ic_vals = []
            
            for i in range(n_types):
                vi0, vi1 = types[i]
                for j in range(n_types):
                    if i == j:
                        continue
                    row_idx = len(ic_rows)
                    ic_rows.append(row_idx); ic_cols.append(3*i + 0); ic_vals.append(-vi0)
                    ic_rows.append(row_idx); ic_cols.append(3*i + 1); ic_vals.append(-vi1)
                    ic_rows.append(row_idx); ic_cols.append(3*j + 0); ic_vals.append(vi0)
                    ic_rows.append(row_idx); ic_cols.append(3*j + 1); ic_vals.append(vi1)
                    ic_rows.append(row_idx); ic_cols.append(3*i + 2); ic_vals.append(1.0)
                    ic_rows.append(row_idx); ic_cols.append(3*j + 2); ic_vals.append(-1.0)
            
            n_ic = len(types) * (len(types) - 1)
            
            # IR constraints: p_i - v_i·x_i <= 0
            ir_rows = []
            ir_cols = []
            ir_vals = []
            for i in range(n_types):
                vi0, vi1 = types[i]
                row_idx = len(ir_rows)
                ir_rows.append(row_idx); ir_cols.append(3*i + 0); ir_vals.append(-vi0)
                ir_rows.append(row_idx); ir_cols.append(3*i + 1); ir_vals.append(-vi1)
                ir_rows.append(row_idx); ir_cols.append(3*i + 2); ir_vals.append(1.0)
            
            n_ir = n_types
            
            all_rows = np.concatenate([ic_rows, ir_rows])
            all_cols = np.concatenate([ic_cols, ir_cols])
            all_vals = np.concatenate([ic_vals, ir_vals])
            
            n_constraints = n_ic + n_ir
            A_ub = sp.csr_matrix((all_vals, (all_rows, all_cols)), shape=(n_constraints, n_vars))
            b_ub = np.zeros(n_constraints)
            
            bounds = []
            for i in range(n_types):
                bounds.append((0.0, 1.0))
                bounds.append((0.0, 1.0))
                bounds.append((0.0, None))
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success and result.x is not None:
                x = result.x
                allocations = np.zeros((n_types, 2))
                payments = np.zeros(n_types)
                for i in range(n_types):
                    allocations[i, 0] = max(0.0, min(1.0, x[3*i + 0]))
                    allocations[i, 1] = max(0.0, min(1.0, x[3*i + 1]))
                    payments[i] = max(0.0, x[3*i + 2])
                
                if verify_mechanism(allocations, payments):
                    return {"allocations": allocations, "payments": payments}
                else:
                    # Return allocations for repair
                    return {"allocations": allocations, "payments": None, "repair": True}
        except Exception:
            pass
        return None

    def solve_repair_lp(allocations):
        """
        Given allocations, find minimal payments (maximal utilities) to satisfy IC/IR.
        Minimize sum probs[i] * U[i]
        Subject to:
        U[i] >= 0
        U[i] - U[j] >= v_i . x_i - v_i . x_j
        """
        try:
            from scipy.optimize import linprog
            import scipy.sparse as sp

            n_vars = n_types # U[0]...U[15]
            
            # Objective: Minimize sum probs[i] * U[i]
            c = probs.copy()
            
            # Constraints
            rows = []
            cols = []
            vals = []
            
            # IR: -U[i] <= 0  => U[i] >= 0 handled by bounds, but let's use explicit constraints for clarity? 
            # Bounds are better.
            
            # IC: U[i] - U[j] >= v_i . x_i - v_i . x_j
            # -U[i] + U[j] <= v_i . x_j - v_i . x_i
            for i in range(n_types):
                vi0, vi1 = types[i]
                vi_dot_xi = vi0 * allocations[i, 0] + vi1 * allocations[i, 1]
                for j in range(n_types):
                    if i == j:
                        continue
                    vj0, vj1 = types[j]
                    # Note: The IC constraint is defined for type i reporting as j.
                    # Utility of i when allocated x_j and paying p_j: v_i . x_j - p_j
                    # Utility of i when allocated x_i and paying p_i: v_i . x_i - p_i
                    # We need v_i . x_i - p_i >= v_i . x_j - p_j
                    # p_k = v_k . x_k - U_k
                    # v_i . x_i - (v_i . x_i - U_i) >= v_i . x_j - (v_j . x_j - U_j)
                    # U_i >= v_i . x_j - v_j . x_j + U_j
                    # U_i - U_j >= v_i . x_j - v_j . x_j
                    # -U_i + U_j <= v_j . x_j - v_i . x_j
                    
                    vj_dot_xj = vj0 * allocations[j, 0] + vj1 * allocations[j, 1]
                    row_idx = len(rows)
                    rows.append(row_idx); cols.append(i); vals.append(-1.0)
                    rows.append(row_idx); cols.append(j); vals.append(1.0)
                    
            n_constraints = len(rows)
            if n_constraints > 0:
                A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))
                b_ub = np.zeros(n_constraints)
                # Recalculate b_ub properly
                b_ub = np.zeros(n_constraints)
                for i in range(n_types):
                    vi0, vi1 = types[i]
                    for j in range(n_types):
                        if i == j:
                            continue
                        vj0, vj1 = types[j]
                        vj_dot_xj = vj0 * allocations[j, 0] + vj1 * allocations[j, 1]
                        vi_dot_xj = vi0 * allocations[j, 0] + vi1 * allocations[j, 1]
                        # Row index needs to match
                        # This loop is getting messy. Let's rebuild b_ub in the same loop as A_ub.
                pass # Re-implementing below for clarity

            # Re-implementing A_ub and b_ub construction cleanly
            rows = []
            cols = []
            vals = []
            b_list = []
            
            for i in range(n_types):
                vi0, vi1 = types[i]
                for j in range(n_types):
                    if i == j:
                        continue
                    vj0, vj1 = types[j]
                    vj_dot_xj = vj0 * allocations[j, 0] + vj1 * allocations[j, 1]
                    vi_dot_xj = vi0 * allocations[j, 0] + vi1 * allocations[j, 1]
                    
                    row_idx = len(rows)
                    rows.append(row_idx); cols.append(i); vals.append(-1.0)
                    rows.append(row_idx); cols.append(j); vals.append(1.0)
                    b_list.append(vj_dot_xj - vi_dot_xj)
            
            n_constraints = len(rows)
            A_ub = sp.csr_matrix((vals, (rows, cols)), shape=(n_constraints, n_vars))
            b_ub = np.array(b_list)
            
            bounds = [(0.0, None)] * n_types
            
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if result.success and result.x is not None:
                U = result.x
                payments = np.zeros(n_types)
                for i in range(n_types):
                    vi0, vi1 = types[i]
                    payments[i] = vi0 * allocations[i, 0] + vi1 * allocations[i, 1] - U[i]
                
                # Ensure non-negative payments
                payments = np.maximum(payments, 0.0)
                
                if verify_mechanism(allocations, payments):
                    return {"allocations": allocations, "payments": payments}
        except Exception:
            pass
        return None

    # Main Execution
    result = None
    if not time_exceeded():
        result = solve_joint_lp()
        
    if result is None or (result.get("repair") and result.get("allocations") is not None):
        if time_exceeded():
            return build_fallback()
            
        if result and result.get("allocations") is not None:
            repair_result = solve_repair_lp(result["allocations"])
            if repair_result:
                return repair_result
        
    if result and result.get("allocations") is not None and result.get("payments") is not None:
        return result

    return build_fallback()
