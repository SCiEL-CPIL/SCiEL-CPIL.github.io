"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog
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

def _build_lp_matrices(types, num_types):
    """
    Build LP constraint matrices using adjacency-based IC constraints.
    For a 2D grid of types, IC is implied by adjacent (chain) constraints
    plus the lattice structure. We use all adjacent pairs in the 4x4 grid
    which gives a valid but tighter formulation than all-pairs.
    """
    # Types indexed as: i = v1 * 4 + v2, where v1, v2 in {0,1,2,3}
    # Adjacent pairs: horizontal (same v1, v2 differs by 1) and vertical (v1 differs by 1, same v2)
    
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
    
    # For IC: u(i) >= u(i|j) for all i,j
    # With additive utilities and the lattice structure, it suffices to check
    # adjacent pairs in both directions. But to be safe, we include all pairs
    # where one coordinate is >= and the other is <= (dominance pairs).
    # Actually, for correctness we need all pairs. Let's use all pairs but
    # build the matrix more efficiently.
    
    # Use all-pairs IC for correctness (256 constraints) + 16 IR
    # Build sparse-like dense matrix efficiently
    n_vars = 3 * num_types
    n_constraints = num_types * num_types + num_types  # IC + IR
    
    A_ub = np.zeros((n_constraints, n_vars))
    b_ub = np.zeros(n_constraints)
    
    row_idx = 0
    for i in range(num_types):
        v1_i, v2_i = types[i]
        for j in range(num_types):
            row = A_ub[row_idx]
            # IC: v1_i*x1[i] + v2_i*x2[i] - t[i] >= v1_i*x1[j] + v2_i*x2[j] - t[j]
            # => -v1_i*x1[i] - v2_i*x2[i] + t[i] + v1_i*x1[j] + v2_i*x2[j] - t[j] <= 0
            row[i] = -v1_i
            row[num_types + i] = -v2_i
            row[2 * num_types + i] = 1.0
            if i != j:
                row[j] += v1_i
                row[num_types + j] += v2_i
                row[2 * num_types + j] -= 1.0
            # If i == j, the constraint is 0 <= 0 (trivially satisfied)
            row_idx += 1
    
    # IR constraints
    for i in range(num_types):
        v1_i, v2_i = types[i]
        row = A_ub[row_idx]
        row[i] = -v1_i
        row[num_types + i] = -v2_i
        row[2 * num_types + i] = 1.0
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
    """
    Fix IC/IR violations using a more robust approach.
    Uses the fact that for a given allocation, the optimal payment
    that maximizes revenue while maintaining IC/IR can be computed
    via a simple formula.
    """
    x1 = np.clip(allocations[:, 0], 0, 1).copy()
    x2 = np.clip(allocations[:, 1], 0, 1).copy()
    t = np.clip(payments, 0, None).copy()
    
    # For a fixed allocation, the maximum payment for type i that maintains IC
    # is: t[i] = min(IR_bound[i], IC_bound[i])
    # where IR_bound[i] = v1_i*x1[i] + v2_i*x2[i]
    # and IC_bound[i] = v1_i*x1[i] + v2_i*x2[i] - max_j (v1_i*x1[j] + v2_i*x2[j] - t[j])
    #
    # The optimal t that maximizes sum(p_i * t_i) subject to IC/IR
    # can be found by: t[i] = min_j (v1_i*x1[i] + v2_i*x2[i] - v1_i*x1[j] - v2_i*x2[j] + t[j])
    # with the constraint that t[i] >= 0 and t[i] <= IR_bound[i]
    #
    # A simpler approach: iterate until convergence
    
    converged = False
    for iteration in range(100):
        changed = False
        for i in range(num_types):
            v1_i, v2_i = types[i]
            ir_bound = v1_i * x1[i] + v2_i * x2[i]
            
            # IC constraint: u_ii >= u_ij for all j
            # v1_i*x1[i] + v2_i*x2[i] - t[i] >= v1_i*x1[j] + v2_i*x2[j] - t[j]
            # t[i] <= v1_i*x1[i] + v2_i*x2[i] - v1_i*x1[j] - v2_i*x2[j] + t[j]
            ic_bound = np.inf
            for j in range(num_types):
                bound = v1_i * x1[i] + v2_i * x2[i] - v1_i * x1[j] - v2_i * x2[j] + t[j]
                if bound < ic_bound:
                    ic_bound = bound
            
            # t[i] must be <= min(ir_bound, ic_bound) and >= 0
            max_t = min(ir_bound, ic_bound)
            max_t = max(max_t, 0.0)
            
            if t[i] > max_t:
                t[i] = max_t
                changed = True
            elif t[i] < 0:
                t[i] = 0.0
                changed = True
        
        if not changed:
            converged = True
            break
    
    return np.column_stack([x1, x2]), t

def _solve_lp(probs, types, num_types, A_ub, b_ub, bounds):
    """Solve the revenue-maximizing LP with high precision."""
    c = np.zeros(3 * num_types)
    for i in range(num_types):
        c[2 * num_types + i] = -probs[i]
    
    # Try multiple solver configurations for robustness
    solver_configs = [
        {'method': 'highs', 'options': {'presolve': True, 'primal_feasibility_tolerance': 1e-12, 'dual_feasibility_tolerance': 1e-12}},
        {'method': 'highs', 'options': {'presolve': False, 'primal_feasibility_tolerance': 1e-12, 'dual_feasibility_tolerance': 1e-12}},
        {'method': 'highs', 'options': {'presolve': True}},
        {'method': 'highs'},
    ]
    
    best_x = None
    best_obj = -np.inf
    
    for config in solver_configs:
        try:
            result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, **config)
            if result.success:
                obj = -result.fun  # Since we're minimizing -revenue
                if obj > best_obj:
                    best_obj = obj
                    best_x = result.x
        except Exception:
            pass
    
    return best_x

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
    probs = _parse_distribution(distribution_spec, num_types, types)
    
    # Build LP matrices
    A_ub, b_ub = _build_lp_matrices(types, num_types)
    
    # Bounds
    bounds = []
    for i in range(num_types):
        bounds.append((0, 1))  # x1[i]
    for i in range(num_types):
        bounds.append((0, 1))  # x2[i]
    for i in range(num_types):
        bounds.append((0, None))  # t[i]
    
    best_result = None
    best_revenue = -np.inf
    
    # Solve the LP with HiGHS
    x_sol = _solve_lp(probs, types, num_types, A_ub, b_ub, bounds)
    
    if x_sol is not None:
        x1 = np.clip(x_sol[:num_types], 0, 1)
        x2 = np.clip(x_sol[num_types:2 * num_types], 0, 1)
        t = np.clip(x_sol[2 * num_types:3 * num_types], 0, None)
        
        allocations = np.column_stack([x1, x2])
        payments = t
        
        # First try the raw LP solution
        if _verify_mechanism(allocations, payments, types, num_types):
            revenue_val = np.sum(probs * t)
            best_result = {'allocations': allocations, 'payments': payments}
            best_revenue = revenue_val
            return best_result
        
        # Fix violations and verify
        alloc_fixed, pay_fixed = _fix_violations(allocations, payments, types, num_types)
        if _verify_mechanism(alloc_fixed, pay_fixed, types, num_types):
            revenue_val = np.sum(probs * pay_fixed)
            best_result = {'allocations': alloc_fixed, 'payments': pay_fixed}
            best_revenue = revenue_val
            return best_result
        
        # If still not valid, try a tighter polish: solve a secondary LP
        # with the fixed allocations as constraints
        if time.time() < deadline:
            # Build a secondary LP to maximize revenue with fixed allocations
            # Variables: t[0..15]
            c2 = -probs
            A_ub2 = np.zeros((num_types * num_types + num_types, num_types))
            b_ub2 = np.zeros(num_types * num_types + num_types)
            
            row_idx = 0
            for i in range(num_types):
                v1_i, v2_i = types[i]
                u_ii = v1_i * x1[i] + v2_i * x2[i]
                for j in range(num_types):
                    # IC: u_ii - t[i] >= u_ij - t[j]
                    # => t[i] - t[j] <= u_ii - u_ij
                    u_ij = v1_i * x1[j] + v2_i * x2[j]
                    A_ub2[row_idx, i] = 1.0
                    if i != j:
                        A_ub2[row_idx, j] = -1.0
                    b_ub2[row_idx] = u_ii - u_ij
                    row_idx += 1
                
                # IR: t[i] <= u_ii
                A_ub2[row_idx, i] = 1.0
                b_ub2[row_idx] = u_ii
                row_idx += 1
            
            bounds2 = [(0, None)] * num_types
            
            try:
                result2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds2, method='highs',
                                 options={'presolve': True, 'primal_feasibility_tolerance': 1e-12})
                if result2.success:
                    t2 = np.clip(result2.x, 0, None)
                    if _verify_mechanism(allocations, t2, types, num_types):
                        revenue_val = np.sum(probs * t2)
                        best_result = {'allocations': allocations, 'payments': t2}
                        best_revenue = revenue_val
                        return best_result
            except Exception:
                pass
    
    # Fallback: try a few heuristic mechanisms
    if time.time() < deadline:
        # Heuristic 1: proportional allocation
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x1 = np.full(num_types, alpha)
            x2 = np.full(num_types, alpha)
            t = np.array([types[i][0] * alpha + types[i][1] * alpha for i in range(num_types)])
            alloc = np.column_stack([x1, x2])
            if _verify_mechanism(alloc, t, types, num_types):
                rev = np.sum(probs * t)
                if rev > best_revenue:
                    best_result = {'allocations': alloc, 'payments': t}
                    best_revenue = rev
        
        # Heuristic 2: threshold mechanism
        for thresh in range(0, 8):
            x1 = np.array([1.0 if types[i][0] >= thresh else 0.0 for i in range(num_types)])
            x2 = np.array([1.0 if types[i][1] >= thresh else 0.0 for i in range(num_types)])
            # Compute payments to satisfy IC/IR
            t = np.zeros(num_types)
            for i in range(num_types):
                v1_i, v2_i = types[i]
                t[i] = v1_i * x1[i] + v2_i * x2[i]
            alloc = np.column_stack([x1, x2])
            if _verify_mechanism(alloc, t, types, num_types):
                rev = np.sum(probs * t)
                if rev > best_revenue:
                    best_result = {'allocations': alloc, 'payments': t}
                    best_revenue = rev
    
    if best_result is not None:
        return best_result
    
    # Final fallback: zero mechanism
    allocations = np.zeros((num_types, 2))
    payments = np.zeros(num_types)
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
