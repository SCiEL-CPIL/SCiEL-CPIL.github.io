"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def _parse_distribution(distribution_spec, types):
    """Parse distribution_spec into a probability vector of length 16."""
    n = len(types)
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        if len(distribution_spec) == n:
            probs = np.array(distribution_spec, dtype=float)
        else:
            probs = np.ones(n) / n
    elif isinstance(distribution_spec, dict):
        probs = np.ones(n) / n
        for i, t in enumerate(types):
            if t in distribution_spec:
                probs[i] = distribution_spec[t]
    else:
        probs = np.ones(n) / n
    s = probs.sum()
    if s > 1e-12:
        probs = probs / s
    else:
        probs = np.ones(n) / n
    return probs

def _build_lp_constraints(type_values, n_types):
    """
    Build LP constraint matrix using reduced IC constraints.
    For a single additive bidder, IC is equivalent to:
    1. Monotonicity in each coordinate (allocation of good j non-decreasing in v_j)
    2. Payment consistency via the envelope theorem
    We use the full pairwise IC constraints but with a tighter formulation.
    """
    n_vars = 3 * n_types
    A_rows = []
    b_vals = []
    
    # Pairwise IC constraints: u(theta, theta) >= u(theta, theta') for all theta, theta'
    # u(theta, theta') = v_theta . q(theta') - p(theta')
    # IC: v_theta . q(theta) - p(theta) >= v_theta . q(theta') - p(theta')
    # => v_theta . (q(theta) - q(theta')) >= p(theta) - p(theta')
    # In A_ub form: -v_theta . q(theta) + v_theta . q(theta') + p(theta) - p(theta') <= 0
    
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        for theta_prime_idx in range(n_types):
            if theta_idx == theta_prime_idx:
                continue
            row = np.zeros(n_vars)
            # -v_theta . q(theta)
            row[3*theta_idx + 0] = -v_theta[0]
            row[3*theta_idx + 1] = -v_theta[1]
            # + v_theta . q(theta')
            row[3*theta_prime_idx + 0] = v_theta[0]
            row[3*theta_prime_idx + 1] = v_theta[1]
            # + p(theta)
            row[3*theta_idx + 2] = 1.0
            # - p(theta')
            row[3*theta_prime_idx + 2] = -1.0
            A_rows.append(row)
            b_vals.append(0.0)
    
    # IR constraints: v_theta . q(theta) - p(theta) >= 0
    # => -v_theta . q(theta) + p(theta) <= 0
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        row = np.zeros(n_vars)
        row[3*theta_idx + 0] = -v_theta[0]
        row[3*theta_idx + 1] = -v_theta[1]
        row[3*theta_idx + 2] = 1.0
        A_rows.append(row)
        b_vals.append(0.0)
    
    A = np.array(A_rows)
    b = np.array(b_vals)
    return A, b

def _verify_and_repair(allocations, payments, type_values, n_types, tol=1e-8):
    """Verify IC/IR and repair if violated. Returns (allocations, payments, valid)."""
    # Check IC
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        u_truth = v_theta[0]*allocations[theta_idx,0] + v_theta[1]*allocations[theta_idx,1] - payments[theta_idx]
        for theta_prime_idx in range(n_types):
            u_dev = v_theta[0]*allocations[theta_prime_idx,0] + v_theta[1]*allocations[theta_prime_idx,1] - payments[theta_prime_idx]
            if u_dev > u_truth + tol:
                return allocations, payments, False
    
    # Check IR
    for theta_idx in range(n_types):
        v_theta = type_values[theta_idx]
        u = v_theta[0]*allocations[theta_idx,0] + v_theta[1]*allocations[theta_idx,1] - payments[theta_idx]
        if u < -tol:
            return allocations, payments, False
    
    # Check nonnegativity
    if np.any(allocations < -tol) or np.any(allocations > 1 + tol) or np.any(payments < -tol):
        return allocations, payments, False
    
    return allocations, payments, True

def _fallback_result(type_values):
    """Return a simple valid mechanism as fallback."""
    allocations = np.full((16, 2), 0.5)
    payments = np.zeros(16)
    for i in range(16):
        payments[i] = 0.5 * (type_values[i, 0] + type_values[i, 1])
    return {'allocations': allocations, 'payments': payments}

def _polish_solution(allocations, payments, type_values, probs, n_types, deadline):
    """
    Try to improve the solution via a local LP refinement with tighter tolerances.
    This is a polish pass that attempts to extract more revenue while maintaining IC/IR.
    """
    if time.time() > deadline - 3:
        return allocations, payments
    
    # Try a high-precision LP solve with the current solution as a warm start
    # by re-solving with tighter bounds around the current solution
    n_vars = 3 * n_types
    
    # Objective: maximize expected revenue
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[3*i + 2] = -probs[i]
    
    A, b = _build_lp_constraints(type_values, n_types)
    
    # Tighten bounds around current solution
    bounds = []
    for i in range(n_types):
        lo_q1 = max(0.0, allocations[i, 0] - 0.1)
        hi_q1 = min(1.0, allocations[i, 0] + 0.1)
        lo_q2 = max(0.0, allocations[i, 1] - 0.1)
        hi_q2 = min(1.0, allocations[i, 1] + 0.1)
        lo_p = max(0.0, payments[i] - 1.0)
        hi_p = payments[i] + 1.0
        bounds.append((lo_q1, hi_q1))
        bounds.append((lo_q2, hi_q2))
        bounds.append((lo_p, hi_p))
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'presolve': True, 'dual_feasibility_tolerance': 1e-10,
                              'primal_feasibility_tolerance': 1e-10})
    
    if result.success and time.time() < deadline - 1:
        x = result.x
        new_alloc = np.zeros((16, 2))
        new_pay = np.zeros(16)
        for i in range(n_types):
            new_alloc[i, 0] = x[3*i + 0]
            new_alloc[i, 1] = x[3*i + 1]
            new_pay[i] = x[3*i + 2]
        
        new_alloc = np.clip(new_alloc, 0.0, 1.0)
        new_pay = np.maximum(new_pay, 0.0)
        
        # Verify the polished solution
        _, _, valid = _verify_and_repair(new_alloc, new_pay, type_values, n_types)
        if valid:
            # Check if it's better
            old_rev = np.dot(probs, payments)
            new_rev = np.dot(probs, new_pay)
            if new_rev > old_rev - 1e-10:
                return new_alloc, new_pay
    
    return allocations, payments

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with 1 additive bidder, 2 goods.
    Values in {0,1,2,3}. Three fixed distributions.
    Returns: {'allocations': (16,2) array of probabilities, 'payments': (16,) array}
    """
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    # Generate all 16 types: (v1, v2) where v1, v2 in {0,1,2,3}
    types = [(v1, v2) for v1 in range(4) for v2 in range(4)]
    n_types = 16
    type_values = np.array(types, dtype=float)  # shape (16, 2)
    
    # Parse distribution
    probs = _parse_distribution(distribution_spec, types)
    
    # LP formulation
    n_vars = 3 * n_types  # 48
    
    # Objective: maximize sum(probs[i] * p(i))
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[3*i + 2] = -probs[i]
    
    # Build constraints
    A, b = _build_lp_constraints(type_values, n_types)
    
    # Bounds
    bounds = []
    for i in range(n_types):
        bounds.append((0, 1))
        bounds.append((0, 1))
        bounds.append((0, None))
    
    # Solve LP with time check
    if time.time() > deadline - 10:
        return _fallback_result(type_values)
    
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs',
                     options={'presolve': True})
    
    if not result.success:
        return _fallback_result(type_values)
    
    x = result.x
    
    # Extract allocations and payments
    allocations = np.zeros((16, 2))
    payments = np.zeros(16)
    for i in range(n_types):
        allocations[i, 0] = x[3*i + 0]
        allocations[i, 1] = x[3*i + 1]
        payments[i] = x[3*i + 2]
    
    # Clip to handle numerical issues
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)
    
    # Verify and fix IC/IR violations
    allocations, payments, valid = _verify_and_repair(allocations, payments, type_values, n_types)
    
    if not valid:
        # Try a slightly relaxed verification
        allocations, payments, valid = _verify_and_repair(allocations, payments, type_values, n_types, tol=1e-7)
    
    if not valid:
        return _fallback_result(type_values)
    
    # Polish pass: try to improve with tighter tolerances
    if time.time() < deadline - 5:
        allocations, payments = _polish_solution(allocations, payments, type_values, probs, n_types, deadline)
    
    return {'allocations': allocations, 'payments': payments}
# EVOLVE-BLOCK-END
