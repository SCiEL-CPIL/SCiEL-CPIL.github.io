"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    
    Args:
        distribution_spec: dict mapping (v1, v2) to probability, or int index for standard distributions
        support: list of (v1, v2) pairs or similar
        constraints: dict with constraint parameters
        
    Returns:
        dict with 'allocations' (16, 2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 75  # Safety margin before 90s kill
    
    # Define the 16 types: (v1, v2) for v1, v2 in {0, 1, 2, 3}
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)  # 16
    
    # Determine probabilities
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        # Normalize
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        # Three fixed distributions - all uniform on {0,1,2,3}^2
        probs = np.ones(num_types) / num_types
    else:
        # Try to interpret as list/array
        try:
            probs = np.array(distribution_spec, dtype=float).flatten()
            if len(probs) == num_types:
                total = probs.sum()
                if total > 0:
                    probs /= total
                else:
                    probs = np.ones(num_types) / num_types
            else:
                probs = np.ones(num_types) / num_types
        except:
            probs = np.ones(num_types) / num_types
    
    # Formulate as a linear program:
    # Variables: x1[0..15], x2[0..15], t[0..15] (48 variables)
    # Maximize: sum_i probs[i] * t[i]
    # Subject to:
    #   IC: v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= v1[i]*x1[j] + v2[i]*x2[j] - t[j] for all i,j
    #   IR: v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= 0 for all i
    #   0 <= x1[i], x2[i] <= 1 for all i
    #   t[i] >= 0 for all i
    
    # Objective: minimize -sum_i probs[i] * t[i]
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]
    
    # Build inequality constraints: A_ub @ x <= b_ub
    
    A_ub_rows = []
    b_ub_vals = []
    
    # IC constraints: For each i, j:
    # v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= v1[i]*x1[j] + v2[i]*x2[j] - t[j]
    # => -v1[i]*x1[i] - v2[i]*x2[i] + t[i] + v1[i]*x1[j] + v2[i]*x2[j] - t[j] <= 0
    
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            row = np.zeros(3 * num_types)
            # Coefficient for x1[i]: -v1_i
            row[i] = -v1_i
            # Coefficient for x2[i]: -v2_i
            row[num_types + i] = -v2_i
            # Coefficient for t[i]: +1
            row[2 * num_types + i] = 1
            # Coefficient for x1[j]: +v1_i
            row[j] += v1_i
            # Coefficient for x2[j]: +v2_i
            row[num_types + j] += v2_i
            # Coefficient for t[j]: -1
            row[2 * num_types + j] -= 1
            A_ub_rows.append(row)
            b_ub_vals.append(0.0)
    
    # IR constraints: For each i:
    # v1[i]*x1[i] + v2[i]*x2[i] - t[i] >= 0
    # => -v1[i]*x1[i] - v2[i]*x2[i] + t[i] <= 0
    
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = np.zeros(3 * num_types)
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1
        A_ub_rows.append(row)
        b_ub_vals.append(0.0)
    
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub_vals)
    
    # Bounds:
    # x1[i] in [0, 1] for all i
    # x2[i] in [0, 1] for all i
    # t[i] in [0, inf) for all i
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i]
    
    best_result = None
    best_revenue = -np.inf
    
    # Solve the LP with HiGHS (exact solver)
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x = result.x
            x1 = x[:num_types]
            x2 = x[num_types:2 * num_types]
            t = x[2 * num_types:3 * num_types]
            
            # Clip small numerical errors
            x1 = np.clip(x1, 0, 1)
            x2 = np.clip(x2, 0, 1)
            t = np.clip(t, 0, None)
            
            allocations = np.column_stack([x1, x2])
            payments = t
            
            # Verify IC and IR with strict tolerance
            valid = True
            for i in range(num_types):
                v1_i, v2_i = types[i]
                u_ii = v1_i * x1[i] + v2_i * x2[i] - t[i]
                if u_ii < -1e-8:
                    valid = False
                    break
                for j in range(num_types):
                    u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
                    if u_ii < u_ij - 1e-8:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                revenue_val = np.sum(probs * t)
                best_result = {'allocations': allocations, 'payments': payments}
                best_revenue = revenue_val
                return best_result
            else:
                # If verification failed, try to fix small violations
                # by adjusting payments slightly
                x1_adj = x1.copy()
                x2_adj = x2.copy()
                t_adj = t.copy()
                
                # Recompute payments to ensure IC/IR
                # For the optimal mechanism, set u[0] = 0 and derive others
                # But for now, just try a small perturbation
                for i in range(num_types):
                    v1_i, v2_i = types[i]
                    # Ensure IR: t[i] <= v1_i*x1[i] + v2_i*x2[i]
                    max_t_ir = v1_i * x1_adj[i] + v2_i * x2_adj[i]
                    if t_adj[i] > max_t_ir:
                        t_adj[i] = max_t_ir
                
                # Re-verify
                valid = True
                for i in range(num_types):
                    v1_i, v2_i = types[i]
                    u_ii = v1_i * x1_adj[i] + v2_i * x2_adj[i] - t_adj[i]
                    if u_ii < -1e-8:
                        valid = False
                        break
                    for j in range(num_types):
                        u_ij = v1_i * x1_adj[j] + v2_i * x2_adj[j] - t_adj[j]
                        if u_ii < u_ij - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    revenue_val = np.sum(probs * t_adj)
                    best_result = {'allocations': np.column_stack([x1_adj, x2_adj]), 'payments': t_adj}
                    best_revenue = revenue_val
                    return best_result
    
    except Exception as e:
        pass
    
    # If LP failed or verification failed, try SLSQP as fallback
    if time.time() < deadline:
        def unpack(z):
            x1 = z[:num_types]
            x2 = z[num_types:2*num_types]
            t = z[2*num_types:3*num_types]
            return x1, x2, t
        
        def revenue_neg(z):
            _, _, t = unpack(z)
            return -np.sum(probs * t)
        
        def ic_constraints(z):
            x1, x2, t = unpack(z)
            cons = []
            for i in range(num_types):
                v1_i, v2_i = types[i]
                for j in range(num_types):
                    cons.append(
                        v1_i*(x1[i] - x1[j]) + v2_i*(x2[i] - x2[j]) - (t[i] - t[j])
                    )
            return np.array(cons)
        
        def ir_constraints(z):
            x1, x2, t = unpack(z)
            cons = []
            for i in range(num_types):
                v1_i, v2_i = types[i]
                cons.append(v1_i*x1[i] + v2_i*x2[i] - t[i])
            return np.array(cons)
        
        # Initial guess: use LP solution if available, else default
        if best_result is not None:
            z0 = np.concatenate([best_result['allocations'][:, 0], 
                                 best_result['allocations'][:, 1], 
                                 best_result['payments']])
        else:
            z0 = np.zeros(3*num_types)
            for i in range(num_types):
                v1_i, v2_i = types[i]
                z0[i] = 0.5
                z0[num_types + i] = 0.5
                z0[2*num_types + i] = v1_i * 0.5 + v2_i * 0.5
        
        try:
            result_slsqp = minimize(
                revenue_neg, z0, method='SLSQP', bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': ic_constraints},
                    {'type': 'ineq', 'fun': ir_constraints}
                ],
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            if result_slsqp.success:
                x1 = result_slsqp.x[:num_types]
                x2 = result_slsqp.x[num_types:2*num_types]
                t = result_slsqp.x[2*num_types:3*num_types]
                
                x1 = np.clip(x1, 0, 1)
                x2 = np.clip(x2, 0, 1)
                t = np.clip(t, 0, None)
                
                allocations = np.column_stack([x1, x2])
                payments = t
                
                # Verify
                valid = True
                for i in range(num_types):
                    v1_i, v2_i = types[i]
                    u_ii = v1_i * x1[i] + v2_i * x2[i] - t[i]
                    if u_ii < -1e-8:
                        valid = False
                        break
                    for j in range(num_types):
                        u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
                        if u_ii < u_ij - 1e-8:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    revenue_val = np.sum(probs * t)
                    if revenue_val > best_revenue:
                        best_result = {'allocations': allocations, 'payments': payments}
                        best_revenue = revenue_val
                        return best_result
        
        except Exception as e:
            pass
    
    # Final fallback: simple mechanism
    if best_result is not None:
        return best_result
    
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        allocations[i] = [0.5, 0.5]
        payments[i] = v1_i * 0.5 + v2_i * 0.5
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
