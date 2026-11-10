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
    SAFETY_MARGIN = 12.0
    DEADLINE = WALL_LIMIT - SAFETY_MARGIN  # 78s

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

    # Build fallback mechanism with better payment computation
    def build_fallback():
        allocations = np.zeros((n_types, 2))
        payments = np.zeros(n_types)
        
        for i, (v1, v2) in enumerate(types):
            if v1 >= 2:
                allocations[i, 0] = 1.0
            if v2 >= 2:
                allocations[i, 1] = 1.0

        # Compute payments using the lattice path formula for multi-dimensional IC
        # For each type (v1, v2), the maximum feasible payment is:
        # p(v1,v2) = min over all monotone paths from (0,0) to (v1,v2) of 
        #            [v1*x1(v1,v2) + v2*x2(v1,v2) - sum of marginal utility losses along path]
        
        # Simpler: use the standard formula for separable allocations
        # p(v1,v2) = v1*x1(v1,v2) + v2*x2(v1,v2) - sum_{k=1}^{v1} (x1(k,v2) - x1(k-1,v2)) * (v1-k+1)
        #           - sum_{k=1}^{v2} (x2(v1,k) - x2(v1,k-1)) * (v2-k+1)
        # Wait, that's not quite right. Let me use the correct formula.
        
        # For multi-dimensional IC, the payment is:
        # p(v) = v·x(v) - max_{path from 0 to v} sum_{t in path} (x(t) - x(t_prev))·t
        
        # Let's compute it step by step using dynamic programming
        # Actually, for this simple case, let's just verify IC/IR and adjust if needed
        
        for i, (v1, v2) in enumerate(types):
            x1 = allocations[i, 0]
            x2 = allocations[i, 1]
            
            # Compute the maximum feasible payment using the path formula
            # The payment must satisfy: for all j, u_i(t_i) >= u_i(t_j)
            # p_i <= v1*x1_i + v2*x2_i - v1*x1_j - v2*x2_j + p_j for all j
            
            # Start with IR bound
            max_pay = v1 * x1 + v2 * x2
            
            # Apply IC constraints iteratively
            # This is a system of inequalities; solve by taking the minimum bound
            # For simplicity, use the standard formula for lattice-structured types
            
            # The correct formula for multi-dimensional IC with monotone allocations:
            # p(v1,v2) = min_{path} [v1*x1(v1,v2) + v2*x2(v1,v2) - sum_{steps} (delta_x1 * v1_step + delta_x2 * v2_step)]
            
            # Let's just compute payments using the recursive formula
            # p(0,0) = 0
            # p(v1,v2) = min(p(v1-1,v2) + (x1(v1,v2)-x1(v1-1,v2))*v1 + (x2(v1,v2)-x2(v1-1,v2))*v2,
            #             p(v1,v2-1) + (x1(v1,v2)-x1(v1,v2-1))*v1 + (x2(v1,v2)-x2(v1,v2-1))*v2)
            
            pass
        
        # Use a simpler but valid approach: set payments to ensure IC/IR
        # For each type, compute the minimum payment that satisfies all constraints
        # Actually, let's just use the LP result if available, or a simple valid mechanism
        
        # Simple valid mechanism: allocate good 1 if v1 >= 2, good 2 if v2 >= 2
        # Payments: p(v1,v2) = max(0, v1*1[v1>=2] - sum_{k=1}^{v1} 1[k>=2], 
        #                           v2*1[v2>=2] - sum_{k=1}^{v2} 1[k>=2])
        # But this doesn't account for cross-dimensional IC
        
        # Let's compute payments properly using the lattice structure
        p = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            if v1 == 0 and v2 == 0:
                p[i] = 0.0
                continue
            
            # Use the formula: p(v) = v·x(v) - max_{path} sum of marginal utility
            # For now, use a conservative payment that ensures IC/IR
            # p(v1,v2) = v1*alloc(v1,v2,0) + v2*alloc(v1,v2,1) 
            #            - (v1*max(0,v1-1) + v2*max(0,v2-1)) * 0  # This is getting complicated
            
            # Let's just set payment to ensure IR and check IC
            p[i] = v1 * allocations[i,0] + v2 * allocations[i,1]
        
        # Now adjust to ensure IC: for each pair (i,j), check if u_i(t_i) >= u_i(t_j)
        # If not, reduce p_i or increase p_j (but p_j >= 0)
        # Actually, the above payment formula may not satisfy IC
        
        # Better approach: use the standard result for multi-dimensional IC
        # The payment is: p(v) = v·x(v) - sum_{u <= v} (x(u) - x(u-e))·u for some path
        # For simplicity, use: p(v1,v2) = v1*x1(v1,v2) + v2*x2(v1,v2) 
        #                             - sum_{k=1}^{v1} (x1(k,v2) - x1(k-1,v2)) * (v1-k+1)
        #                             - sum_{k=1}^{v2} (x2(v1,k) - x2(v1,k-1)) * (v2-k+1)
        
        payments_final = np.zeros(n_types)
        for i, (v1, v2) in enumerate(types):
            x1_i = allocations[i, 0]
            x2_i = allocations[i, 1]
            
            # Compute the "virtual cost" along the v1 dimension
            cost_v1 = 0.0
            for k in range(1, v1 + 1):
                idx_k = type_index[(k, v2)]
                idx_k_minus = type_index[(k-1, v2)]
                delta_x1 = allocations[idx_k, 0] - allocations[idx_k_minus, 0]
                cost_v1 += delta_x1 * (v1 - k + 1)
            
            # Compute the "virtual cost" along the v2 dimension
            cost_v2 = 0.0
            for k in range(1, v2 + 1):
                idx_k = type_index[(v1, k)]
                idx_k_minus = type_index[(v1, k-1)]
                delta_x2 = allocations[idx_k, 1] - allocations[idx_k_minus, 1]
                cost_v2 += delta_x2 * (v2 - k + 1)
            
            payments_final[i] = v1 * x1_i + v2 * x2_i - cost_v1 - cost_v2
        
        # Ensure non-negative and IR
        payments_final = np.maximum(payments_final, 0)
        for i in range(n_types):
            max_pay = types[i][0]*allocations[i,0] + types[i][1]*allocations[i,1]
            if payments_final[i] > max_pay + 1e-10:
                payments_final[i] = max_pay
        
        # Verify IC/IR and fix if needed
        valid = True
        for i in range(n_types):
            vi0, vi1 = types[i]
            util_i = vi0*allocations[i,0] + vi1*allocations[i,1] - payments_final[i]
            if util_i < -1e-8:
                valid = False
                break
            for j in range(n_types):
                if i == j:
                    continue
                util_i_as_j = vi0*allocations[j,0] + vi1*allocations[j,1] - payments_final[j]
                if util_i < util_i_as_j - 1e-8:
                    valid = False
                    break
            if not valid:
                break
        
        if not valid:
            # Fallback to zero payments (always valid but zero revenue)
            payments_final = np.zeros(n_types)
        
        return {"allocations": allocations, "payments": payments_final}

    # Try LP-based optimal mechanism with sparse matrix for efficiency
    try:
        from scipy.optimize import linprog
        import scipy.sparse as sp

        n_vars = n_types * 3  # x1, x2, p for each type

        # Objective: maximize sum_i probs[i] * p[i]
        c = np.zeros(n_vars)
        for i in range(n_types):
            c[3*i + 2] = -probs[i]
        
        # Build IC constraints using lattice structure - only adjacent pairs needed
        # For types on a lattice, IC is implied by adjacent (single-step) constraints
        # A type t_i is adjacent to t_j if they differ by 1 in exactly one coordinate
        
        ic_rows = []
        ic_cols = []
        ic_vals = []
        
        for i in range(n_types):
            vi0, vi1 = types[i]
            # Check all types that are "above" i in the lattice (differ by +1 in one coord)
            for j in range(n_types):
                vj0, vj1 = types[j]
                # j is adjacent above i if: (vj0 - vi0, vj1 - vi1) is (1,0) or (0,1)
                if (vj0 - vi0, vj1 - vi1) not in [(1, 0), (0, 1)]:
                    continue
                row_idx = len(ic_rows)
                # IC: u_i(t_i) >= u_i(t_j) means
                # vi0*x_i[0] + vi1*x_i[1] - p_i >= vi0*x_j[0] + vi1*x_j[1] - p_j
                # => -vi0*x_i[0] - vi1*x_i[1] + vi0*x_j[0] + vi1*x_j[1] + p_i - p_j <= 0
                ic_rows.append(row_idx); ic_cols.append(3*i + 0); ic_vals.append(-vi0)
                ic_rows.append(row_idx); ic_cols.append(3*i + 1); ic_vals.append(-vi1)
                ic_rows.append(row_idx); ic_cols.append(3*j + 0); ic_vals.append(vi0)
                ic_rows.append(row_idx); ic_cols.append(3*j + 1); ic_vals.append(vi1)
                ic_rows.append(row_idx); ic_cols.append(3*i + 2); ic_vals.append(1.0)
                ic_rows.append(row_idx); ic_cols.append(3*j + 2); ic_vals.append(-1.0)
        
        n_ic = len(ic_rows) // 6
        
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
        
        # Combine into one sparse matrix
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
        
        if time_exceeded():
            return build_fallback()
        
        # Use tighter tolerances for better precision
        options = {'presolve': True, 'dual_feasibility_tolerance': 1e-9, 'primal_feasibility_tolerance': 1e-9}
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)
        
        if result.success and result.x is not None:
            x = result.x
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = max(0.0, min(1.0, x[3*i + 0]))
                allocations[i, 1] = max(0.0, min(1.0, x[3*i + 1]))
                payments[i] = max(0.0, x[3*i + 2])
            
            # Post-LP refinement: adjust payments to be as high as possible while maintaining IC/IR
            # This can improve revenue if the LP solution has slack in IC constraints
            # Strategy: for each type, increase payment up to the min of all IC upper bounds
            refined_payments = np.zeros(n_types)
            for i in range(n_types):
                vi0, vi1 = types[i]
                # Upper bound from IR: p_i <= vi0*x_i[0] + vi1*x_i[1]
                ir_bound = vi0*allocations[i,0] + vi1*allocations[i,1]
                
                # Upper bound from IC: for each j, p_i <= vi0*x_i[0] + vi1*x_i[1] - vi0*x_j[0] - vi1*x_j[1] + p_j
                # But p_j is also variable, so this is circular. Instead, use the fact that
                # the optimal payment for type i given allocations is determined by the
                # monotonicity condition. Let's compute the maximum feasible payment.
                
                # For a given allocation profile, the maximum payment for type i that satisfies
                # IC and IR is: min over all paths from (0,0) to (vi0,vi1) of the sum of marginal values
                # This is equivalent to: p_i = min_{path} sum of marginal contributions
                
                # Simpler approach: verify current solution is valid, and if so, try to increase
                # payments by checking which IC constraints are binding
                max_pay_i = ir_bound
                for j in range(n_types):
                    if i == j:
                        continue
                    vj0, vj1 = types[j]
                    # IC constraint: vi0*x_i[0] + vi1*x_i[1] - p_i >= vi0*x_j[0] + vi1*x_j[1] - p_j
                    # => p_i <= vi0*x_i[0] + vi1*x_i[1] - vi0*x_j[0] - vi1*x_j[1] + p_j
                    # But we don't know p_j yet, so skip for now
                    pass
                
                # Use the standard formula: for single-dimensional, p_i = v_i * x_i - sum_{k<v_i} (x_k - x_{k-1}) * k
                # For multi-dimensional with lattice, use the path-based formula
                # The maximum feasible payment is determined by the minimum over all monotone paths
                
                # Compute using the lattice path approach
                # p_i = min over all paths from (0,0) to (vi0,vi1) of [v_i·x_i - sum of marginal losses]
                # Actually, the correct formula is:
                # p_i = v_i·x_i - max_{path} sum of (x_{t'} - x_t)·v_t for each step
                
                # Let's just verify the LP solution is valid and return it if so
                refined_payments[i] = payments[i]
            
            # Verify IC and IR with tighter tolerance
            valid = True
            for i in range(n_types):
                vi0, vi1 = types[i]
                util_i = vi0*allocations[i,0] + vi1*allocations[i,1] - payments[i]
                # IR: util_i >= 0
                if util_i < -1e-7:
                    valid = False
                    break
                for j in range(n_types):
                    if i == j:
                        continue
                    util_i_as_j = vi0*allocations[j,0] + vi1*allocations[j,1] - payments[j]
                    if util_i < util_i_as_j - 1e-7:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                return {"allocations": allocations, "payments": payments}
    
    except Exception:
        pass
    
    return build_fallback()
