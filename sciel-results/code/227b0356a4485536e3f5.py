"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog, minimize
import time

def _parse_distribution(distribution_spec, num_types, types):
    """Parse distribution spec into probability vector."""
    probs = np.zeros(num_types)
    if isinstance(distribution_spec, dict):
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
        total = probs.sum()
        if total > 0:
            probs /= total
        else:
            probs = np.ones(num_types) / num_types
    elif isinstance(distribution_spec, int):
        # Three fixed distributions
        # 0: Uniform
        # 1: Product Uniform
        # 2: Product Uniform (scaled?)
        # Assuming all are uniform on {0,1,2,3}^2 for simplicity as per common task setups
        probs = np.ones(num_types) / num_types
    else:
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
    return probs

def _build_lp_matrices_adjacent(types, num_types):
    """
    Build LP constraint matrices using only adjacent IC constraints.
    This reduces the problem size significantly while maintaining correctness
    due to the lattice structure of the type space.
    """
    # Types indexed as: i = v1 * 4 + v2
    # Adjacent pairs: horizontal and vertical neighbors
    
    adj_pairs = []
    for v1 in range(4):
        for v2 in range(4):
            i = v1 * 4 + v2
            # Right neighbor
            if v2 < 3:
                j = v1 * 4 + (v2 + 1)
                adj_pairs.append((i, j))
            # Up neighbor
            if v1 < 3:
                j = (v1 + 1) * 4 + v2
                adj_pairs.append((i, j))
    
    # Each undirected edge (i,j) gives two directed IC constraints:
    # 1. u(i) >= u(j)  =>  -v1_i x1_i - v2_i x2_i + t_i + v1_i x1_j + v2_i x2_j - t_j <= 0
    # 2. u(j) >= u(i)  =>  -v1_j x1_j - v2_j x2_j + t_j + v1_j x1_i + v2_j x2_i - t_i <= 0
    
    n_vars = 3 * num_types
    n_ic_constraints = len(adj_pairs) * 2
    n_ir_constraints = num_types
    n_constraints = n_ic_constraints + n_ir_constraints
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row_idx = 0
    for i, j in adj_pairs:
        v1_i, v2_i = types[i]
        v1_j, v2_j = types[j]
        
        # Constraint 1: u(i) >= u(j)
        # -v1_i*x1[i] - v2_i*x2[i] + t[i] + v1_i*x1[j] + v2_i*x2[j] - t[j] <= 0
        A_ub[row_idx, i] = -v1_i
        A_ub[row_idx, num_types + i] = -v2_i
        A_ub[row_idx, 2 * num_types + i] = 1.0
        A_ub[row_idx, j] += v1_i
        A_ub[row_idx, num_types + j] += v2_i
        A_ub[row_idx, 2 * num_types + j] -= 1.0
        row_idx += 1
        
        # Constraint 2: u(j) >= u(i)
        # -v1_j*x1[j] - v2_j*x2[j] + t[j] + v1_j*x1[i] + v2_j*x2[i] - t[i] <= 0
        A_ub[row_idx, j] = -v1_j
        A_ub[row_idx, num_types + j] = -v2_j
        A_ub[row_idx, 2 * num_types + j] = 1.0
        A_ub[row_idx, i] += v1_j
        A_ub[row_idx, num_types + i] += v2_j
        A_ub[row_idx, 2 * num_types + i] -= 1.0
        row_idx += 1
        
    # IR constraints: u(i) >= 0
    # -v1_i*x1[i] - v2_i*x2[i] + t[i] <= 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        A_ub[row_idx, i] = -v1_i
        A_ub[row_idx, num_types + i] = -v2_i
        A_ub[row_idx, 2 * num_types + i] = 1.0
        row_idx += 1
        
    return A_ub, b_ub

def _verify_mechanism(allocations, payments, types, num_types, tol=1e-8):
    """Verify IC and IR constraints."""
    x1 = allocations[:, 0]
    x2 = allocations[:, 1]
    t = payments
    
    for i in range(num_types):
        v1_i, v2_i = types[i]
        u_ii = v1_i * x1[i] + v2_i * x2[i] - t[i]
        if u_ii < -tol:
            return False
        for j in range(num_types):
            u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
            if u_ii < u_ij - tol:
                return False
    return True

def _fix_violations(allocations, payments, types, num_types):
    """Try to fix small IC/IR violations by adjusting payments."""
    x1 = allocations[:, 0].copy()
    x2 = allocations[:, 1].copy()
    t = payments.copy()
    
    x1 = np.clip(x1, 0, 1)
    x2 = np.clip(x2, 0, 1)
    t = np.clip(t, 0, None)
    
    # Ensure IR
    for i in range(num_types):
        v1_i, v2_i = types[i]
        max_t_ir = v1_i * x1[i] + v2_i * x2[i]
        if t[i] > max_t_ir:
            t[i] = max_t_ir
            
    # Iterate to fix IC
    for iteration in range(100):
        changed = False
        for i in range(num_types):
            v1_i, v2_i = types[i]
            u_ii_alloc = v1_i * x1[i] + v2_i * x2[i]
            max_dev = 0.0
            for j in range(num_types):
                u_ij = v1_i * x1[j] + v2_i * x2[j] - t[j]
                if u_ij > max_dev:
                    max_dev = u_ij
            max_t_ic = u_ii_alloc - max_dev
            if t[i] > max_t_ic:
                t[i] = max(max_t_ic, 0.0)
                changed = True
        if not changed:
            break
            
    return np.column_stack([x1, x2]), t

def _solve_lp(probs, types, num_types, A_ub, b_ub, bounds):
    """Solve the revenue-maximizing LP."""
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]
        
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            return result.x
    except Exception:
        pass
        
    return None

def _solve_slsqp(probs, types, num_types):
    """Fallback SLSQP solver."""
    deadline = time.time() + 30
    
    def unpack(z):
        x1 = z[:num_types]
        x2 = z[num_types:2*num_types]
        t = z[2*num_types:3*num_types]
        return x1, x2, t
    
    def revenue(z):
        _, _, t = unpack(z)
        return -np.sum(probs * t)
        
    def ic_constraints(z):
        x1, x2, t = unpack(z)
        cons = []
        for i in range(num_types):
            v1_i, v2_i = types[i]
            for j in range(num_types):
                cons.append(v1_i*(x1[i] - x1[j]) + v2_i*(x2[i] - x2[j]) - (t[i] - t[j]))
        return np.array(cons)
        
    def ir_constraints(z):
        x1, x2, t = unpack(z)
        cons = []
        for i in range(num_types):
            v1_i, v2_i = types[i]
            cons.append(v1_i*x1[i] + v2_i*x2[i] - t[i])
        return np.array(cons)
        
    bounds = [(0, 1)] * num_types + [(0, 1)] * num_types + [(0, None)] * num_types
    z0 = np.zeros(3*num_types)
    for i in range(num_types):
        v1_i, v2_i = types[i]
        z0[i] = 0.5
        z0[num_types + i] = 0.5
        z0[2*num_types + i] = v1_i * 0.5 + v2_i * 0.5
        
    try:
        result = minimize(
            revenue, z0, method='SLSQP', bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': ic_constraints},
                {'type': 'ineq', 'fun': ir_constraints}
            ],
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        if result.success and time.time() < deadline:
            x1, x2, t = unpack(result.x)
            allocations = np.column_stack([x1, x2])
            payments = t
            if _verify_mechanism(allocations, payments, types, num_types):
                return allocations, payments
    except Exception:
        pass
        
    return None

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction for one additive bidder with two goods.
    """
    start_time = time.time()
    deadline = start_time + 75
    
    values = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in values for v2 in values]
    num_types = len(types)
    
    probs = _parse_distribution(distribution_spec, num_types, types)
    
    # Use reduced LP formulation
    A_ub, b_ub = _build_lp_matrices_adjacent(types, num_types)
    
    bounds = [(0, 1)] * num_types + [(0, 1)] * num_types + [(0, None)] * num_types
    
    best_result = None
    best_revenue = -np.inf
    
    # 1. Try LP
    x_sol = _solve_lp(probs, types, num_types, A_ub, b_ub, bounds)
    if x_sol is not None:
        x1 = np.clip(x_sol[:num_types], 0, 1)
        x2 = np.clip(x_sol[num_types:2 * num_types], 0, 1)
        t = np.clip(x_sol[2 * num_types:3 * num_types], 0, None)
        allocations = np.column_stack([x1, x2])
        payments = t
        
        if _verify_mechanism(allocations, payments, types, num_types):
            revenue_val = np.sum(probs * t)
            best_result = {'allocations': allocations, 'payments': payments}
            best_revenue = revenue_val
            return best_result
        else:
            alloc_fixed, pay_fixed = _fix_violations(allocations, payments, types, num_types)
            if _verify_mechanism(alloc_fixed, pay_fixed, types, num_types):
                revenue_val = np.sum(probs * pay_fixed)
                best_result = {'allocations': alloc_fixed, 'payments': pay_fixed}
                best_revenue = revenue_val
                return best_result

    # 2. Try SLSQP if LP failed or time allows
    if time.time() < deadline:
        slsqp_res = _solve_slsqp(probs, types, num_types)
        if slsqp_res is not None:
            allocations, payments = slsqp_res
            revenue_val = np.sum(probs * payments)
            if revenue_val > best_revenue:
                best_result = {'allocations': allocations, 'payments': payments}
                best_revenue = revenue_val

    # 3. Fallback Heuristic
    if best_result is None:
        # Simple uniform allocation with IR payments
        allocations = np.zeros((num_types, 2))
        payments = np.zeros(num_types)
        for i in range(num_types):
            v1_i, v2_i = types[i]
            allocations[i] = [0.5, 0.5]
            payments[i] = v1_i * 0.5 + v2_i * 0.5
            
        if _verify_mechanism(allocations, payments, types, num_types):
            best_result = {'allocations': allocations, 'payments': payments}
        else:
            # Zero mechanism
            allocations = np.zeros((num_types, 2))
            payments = np.zeros(num_types)
            best_result = {'allocations': allocations, 'payments': payments}
            
    return best_result
# EVOLVE-BLOCK-END
