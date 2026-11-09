"""Direct convex LP over allocation and payment variables for optimal finite-support mechanism."""
import numpy as np
from scipy.optimize import linprog
import time


def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction mechanism.
    
    Args:
        distribution_spec: specification of the value distribution
        support: support of the value space
        constraints: additional constraints
    
    Returns:
        dict with 'allocations': (16, 2) array and 'payments': (16,) array
    """
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    # Parse distribution
    values = [0, 1, 2, 3]
    
    if distribution_spec is not None:
        if isinstance(distribution_spec, dict):
            if 'probabilities' in distribution_spec:
                probs = np.array(distribution_spec['probabilities'], dtype=float)
            elif 'dist' in distribution_spec:
                probs = np.array(distribution_spec['dist'], dtype=float)
            elif 'p' in distribution_spec:
                probs = np.array(distribution_spec['p'], dtype=float)
            else:
                probs = None
        elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = None
    else:
        probs = None
    
    if probs is None or len(probs) != 16:
        probs = np.ones(16) / 16.0
    else:
        probs = probs / probs.sum()
    
    # Parse support
    if support is not None:
        if isinstance(support, (list, tuple)):
            if len(support) == 16:
                types = [tuple(t) for t in support]
            elif len(support) == 2:
                v1_vals = list(support[0])
                v2_vals = list(support[1])
                types = [(v1, v2) for v1 in v1_vals for v2 in v2_vals]
            else:
                types = [(i, j) for i in values for j in values]
        else:
            types = [(i, j) for i in values for j in values]
    else:
        types = [(i, j) for i in values for j in values]
    
    n = 16
    assert len(types) == n, f"Expected 16 types, got {len(types)}"
    
    # Value matrix: val[t] = (v1, v2)
    val_matrix = np.array(types, dtype=float)  # shape (16, 2)
    
    # Direct LP formulation:
    # Variables: p_t (16), x1_t (16), x2_t (16) -> 48 total
    # Maximize: sum_t probs[t] * p_t
    # Subject to:
    #   IC: p_t - p_t' <= v_t . x_t - v_t . x_t'  for all t, t'
    #   IR: p_t <= v_t . x_t  for all t
    #   0 <= p_t
    #   0 <= x1_t, x2_t <= 1
    
    n_vars = 48
    c = np.zeros(n_vars)
    # Maximize revenue -> minimize -revenue
    c[:16] = -probs
    
    A_rows = []
    b_rows = []
    
    # IC constraints: p_t - p_t' - v_t1*x_t1 + v_t1*x_t1' - v_t2*x_t2 + v_t2*x_t2' <= 0
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            row = np.zeros(n_vars)
            # p_t - p_t'
            row[t] = 1.0
            row[t2] = -1.0
            # -v_t1 * x_t1 + v_t1 * x_t1'
            row[16 + t] = -val_matrix[t, 0]
            row[16 + t2] = val_matrix[t, 0]
            # -v_t2 * x_t2 + v_t2 * x_t2'
            row[32 + t] = -val_matrix[t, 1]
            row[32 + t2] = val_matrix[t, 1]
            A_rows.append(row)
            b_rows.append(0.0)
    
    # IR constraints: p_t - v_t1*x_t1 - v_t2*x_t2 <= 0
    for t in range(n):
        row = np.zeros(n_vars)
        row[t] = 1.0
        row[16 + t] = -val_matrix[t, 0]
        row[32 + t] = -val_matrix[t, 1]
        A_rows.append(row)
        b_rows.append(0.0)
    
    A_ub = np.array(A_rows) if A_rows else np.zeros((0, n_vars))
    b_ub = np.array(b_rows) if b_rows else np.zeros(0)
    
    # Bounds: p_t >= 0, x in [0,1]
    bounds = [(0, None)] * 16
    bounds += [(0, 1)] * 32
    
    try:
        # First pass: solve with moderate tolerance
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if not result.success:
            # Fallback: trivial mechanism
            return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
        
        p = result.x[:16]
        x1 = result.x[16:32]
        x2 = result.x[32:48]
        
        # Check if we have time for a high-precision polish pass
        elapsed = time.time() - start_time
        if elapsed < 60.0:
            # Polish pass: tighter tolerances
            result2 = linprog(
                c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                options={'presolve': False, 'dual_feasibility_tolerance': 1e-12,
                         'primal_feasibility_tolerance': 1e-12,
                         'dual_solution_tolerance': 1e-12,
                         'primal_solution_tolerance': 1e-12,
                         'integrality_tolerance': 1e-12,
                         'crossover_presolve': False}
            )
            if result2.success:
                p = result2.x[:16]
                x1 = result2.x[16:32]
                x2 = result2.x[32:48]
        
        allocations = np.column_stack((x1, x2))
        payments = p
        
        # Clip to handle numerical noise
        allocations = np.clip(allocations, 0, 1)
        payments = np.clip(payments, 0, None)
        
        # Verification: check IC and IR constraints
        if _verify_constraints(payments, allocations, val_matrix, probs, tol=1e-8):
            return {'allocations': allocations, 'payments': payments}
        else:
            # If constraints are violated, try to fix with small adjustments
            fixed_payments, fixed_allocs = _fix_violations(payments, allocations, val_matrix, probs)
            if _verify_constraints(fixed_payments, fixed_allocs, val_matrix, probs, tol=1e-8):
                return {'allocations': fixed_allocs, 'payments': fixed_payments}
            else:
                # Final fallback
                return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}
                
    except Exception:
        # Fallback: trivial mechanism
        return {'allocations': np.zeros((16, 2)), 'payments': np.zeros(16)}


def _verify_constraints(payments, allocations, val_matrix, probs, tol=1e-8):
    """Verify IC, IR, and non-negativity constraints."""
    n = len(payments)
    
    # Check non-negativity
    if np.any(payments < -tol):
        return False
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol):
        return False
    
    # Check IR: p_t <= v_t . x_t
    for t in range(n):
        if payments[t] > val_matrix[t] @ allocations[t] + tol:
            return False
    
    # Check IC: p_t - p_t' <= v_t . x_t - v_t . x_t'
    for t in range(n):
        for t2 in range(n):
            if t == t2:
                continue
            lhs = payments[t] - payments[t2]
            rhs = val_matrix[t] @ allocations[t] - val_matrix[t] @ allocations[t2]
            if lhs > rhs + tol:
                return False
    
    return True


def _fix_violations(payments, allocations, val_matrix, probs, max_iters=100):
    """Attempt to fix constraint violations by adjusting payments downward."""
    n = len(payments)
    payments = payments.copy()
    allocations = allocations.copy()
    
    for _ in range(max_iters):
        # Check IR violations
        for t in range(n):
            max_pay = val_matrix[t] @ allocations[t]
            if payments[t] > max_pay + 1e-10:
                payments[t] = max(max_pay, 0.0)
        
        # Check IC violations and adjust
        fixed = False
        for t in range(n):
            for t2 in range(n):
                if t == t2:
                    continue
                lhs = payments[t] - payments[t2]
                rhs = val_matrix[t] @ allocations[t] - val_matrix[t] @ allocations[t2]
                if lhs > rhs + 1e-10:
                    # Reduce payments[t] or increase payments[t2]
                    # Simple fix: reduce payments[t]
                    payments[t] = max(payments[t2] + rhs, 0.0)
                    fixed = True
        
        if not fixed:
            break
    
    return payments, allocations
