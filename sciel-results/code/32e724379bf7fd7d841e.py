"""Incentive-compatible finite auction design - LP-based solution."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import product

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible auction mechanism.
    
    Args:
        distribution_spec: Specification of the type distribution
        support: Support of types (values in {0,1,2,3} for 2 goods)
        constraints: Additional constraints
    
    Returns:
        dict with 'allocations' (16,2) and 'payments' (16,)
    """
    start_time = time.time()
    deadline = start_time + 75.0  # Safety margin before 90s limit
    
    # Parse the distribution and support
    # Types: (v1, v2) where v1, v2 in {0, 1, 2, 3}
    # 16 possible types total
    
    # Determine support - all 16 types
    types = list(product(range(4), range(4)))
    n_types = len(types)  # 16
    type_index = {t: i for i, t in enumerate(types)}
    
    # Parse distribution_spec to get probabilities for each type
    # distribution_spec could be a dict, list, or other format
    probs = _parse_distribution(distribution_spec, types, type_index)
    
    # Solve the LP for the optimal incentive-compatible mechanism
    allocations, payments = _solve_auction_lp(probs, types, type_index, deadline)
    
    return {'allocations': allocations, 'payments': payments}


def _parse_distribution(distribution_spec, types, type_index):
    """Parse distribution spec into a probability vector over types."""
    n_types = len(types)
    probs = np.zeros(n_types)
    
    if distribution_spec is None:
        # Uniform distribution
        probs[:] = 1.0 / n_types
        return probs
    
    if isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == n_types:
            probs = np.array(distribution_spec, dtype=float)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs[:] = 1.0 / n_types
        elif len(distribution_spec) == 2:
            # Two independent marginals
            p1 = np.array(distribution_spec[0], dtype=float)
            p2 = np.array(distribution_spec[1], dtype=float)
            if p1.sum() > 0:
                p1 = p1 / p1.sum()
            else:
                p1 = np.ones(4) / 4
            if p2.sum() > 0:
                p2 = p2 / p2.sum()
            else:
                p2 = np.ones(4) / 4
            for i, (v1, v2) in enumerate(types):
                probs[i] = p1[v1] * p2[v2]
        else:
            probs[:] = 1.0 / n_types
    elif isinstance(distribution_spec, dict):
        for t, p in distribution_spec.items():
            if t in type_index:
                probs[type_index[t]] = p
        total = probs.sum()
        if total > 0:
            probs = probs / total
        else:
            probs[:] = 1.0 / n_types
    else:
        # Try to convert to array
        try:
            arr = np.array(distribution_spec, dtype=float)
            if arr.shape == (n_types,):
                probs = arr
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs[:] = 1.0 / n_types
            elif arr.size == n_types:
                probs = arr.flatten()
                if probs.sum() > 0:
                    probs = probs / probs.sum()
                else:
                    probs[:] = 1.0 / n_types
            else:
                probs[:] = 1.0 / n_types
        except:
            probs[:] = 1.0 / n_types
    
    return probs


def _solve_auction_lp(probs, types, type_index, deadline):
    """
    Solve the LP for the optimal incentive-compatible auction.
    
    Variables:
    - x[i][j] = allocation of good j to type i (for j=0,1)
    - p[i] = payment for type i
    
    The incentive compatibility (IC) constraints are:
    For all types i, i':
      v_i · x_i - p_i >= v_i · x_i' - p_i'
    
    Individual rationality (IR):
    v_i · x_i - p_i >= 0
    
    Feasibility:
    0 <= x[i][j] <= 1
    p[i] >= 0
    
    Objective: Maximize expected revenue = sum_i probs[i] * p[i]
    
    We use the standard transformation: fix p[0] = 0 (or some reference)
    and use the IC constraints to bound payments.
    
    Actually, for finite type spaces, we can directly solve the LP.
    """
    n_types = len(types)
    n_goods = 2
    
    from scipy.optimize import linprog
    
    # Decision variables:
    # x[i][0], x[i][1] for each type i -> 2*n_types variables
    # p[i] for each type i -> n_types variables
    # Total: 3*n_types variables
    
    # Let's set up the LP.
    # We want to maximize sum_i probs[i] * p[i]
    # So minimize -sum_i probs[i] * p[i]
    
    n_vars = 3 * n_types  # 2*n_types for allocations + n_types for payments
    
    def var_x(i, j):
        return i * 2 + j
    
    def var_p(i):
        return 2 * n_types + i
    
    # Objective: minimize -sum_i probs[i] * p[i]
    c = np.zeros(n_vars)
    for i in range(n_types):
        c[var_p(i)] = -probs[i]
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. IC constraints: v_i · x_i - p_i >= v_i · x_i' - p_i' for all i, i'
    #    => v_i · x_i - v_i · x_i' - p_i + p_i' >= 0
    #    => -(v_i · x_i - v_i · x_i' - p_i + p_i') <= 0
    #    => (-v_i[0]*x_i[0] - v_i[1]*x_i[1] + v_i[0]*x_i'[0] + v_i[1]*x_i'[1] + p_i - p_i') <= 0
    
    # 2. IR constraints: v_i · x_i - p_i >= 0
    #    => -(v_i · x_i - p_i) <= 0
    #    => (-v_i[0]*x_i[0] - v_i[1]*x_i[1] + p_i) <= 0
    
    # 3. Non-negativity of allocations: x[i][j] >= 0 (handled by bounds)
    # 4. Upper bound on allocations: x[i][j] <= 1 (handled by bounds)
    # 5. Non-negativity of payments: p[i] >= 0 (handled by bounds)
    
    # Build IC constraint matrix
    A_rows = []
    b_rows = []
    
    for i in range(n_types):
        vi = types[i]
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            vi_prime = types[i_prime]
            row = np.zeros(n_vars)
            # -vi[0]*x_i[0] - vi[1]*x_i[1]
            row[var_x(i, 0)] = -vi[0]
            row[var_x(i, 1)] = -vi[1]
            # +vi[0]*x_i'[0] + vi[1]*x_i'[1]
            row[var_x(i_prime, 0)] = vi[0]
            row[var_x(i_prime, 1)] = vi[1]
            # +p_i - p_i'
            row[var_p(i)] = 1
            row[var_p(i_prime)] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    # IR constraints
    for i in range(n_types):
        vi = types[i]
        row = np.zeros(n_vars)
        row[var_x(i, 0)] = -vi[0]
        row[var_x(i, 1)] = -vi[1]
        row[var_p(i)] = 1
        A_rows.append(row)
        b_rows.append(0)
    
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
    else:
        A_ub = np.zeros((0, n_vars))
        b_ub = np.zeros(0)
    
    # Bounds: 0 <= x[i][j] <= 1, 0 <= p[i] <= inf
    bounds = []
    for i in range(n_types):
        for j in range(n_goods):
            bounds.append((0, 1))
    for i in range(n_types):
        bounds.append((0, None))
    
    # Check time
    if time.time() > deadline - 5:
        # Not enough time, return a simple feasible solution
        return _simple_feasible_solution(types, probs)
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if result.success:
            x_opt = result.x
            allocations = np.zeros((n_types, n_goods))
            payments = np.zeros(n_types)
            for i in range(n_types):
                allocations[i, 0] = x_opt[var_x(i, 0)]
                allocations[i, 1] = x_opt[var_x(i, 1)]
                payments[i] = x_opt[var_p(i)]
            
            # Clip to ensure feasibility
            allocations = np.clip(allocations, 0, 1)
            payments = np.clip(payments, 0, None)
            
            # Verify IC and IR, fix if needed
            allocations, payments = _verify_and_fix(allocations, payments, types, probs)
            
            return allocations, payments
        else:
            # LP failed, return simple feasible solution
            return _simple_feasible_solution(types, probs)
    except Exception as e:
        return _simple_feasible_solution(types, probs)


def _verify_and_fix(allocations, payments, types, probs):
    """Verify IC and IR constraints, fix violations if any."""
    n_types = len(types)
    
    # Check IR: v_i · x_i - p_i >= 0
    for i in range(n_types):
        vi = types[i]
        utility = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - payments[i]
        if utility < -1e-8:
            payments[i] = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1]
    
    # Check IC: v_i · x_i - p_i >= v_i · x_i' - p_i' for all i, i'
    # This is harder to fix directly. Let's re-check and adjust payments.
    # For each type i, the payment must satisfy:
    # p_i <= v_i · x_i (IR)
    # p_i >= v_i · x_i' - (v_i' · x_i' - p_i') ... this gets complex.
    
    # Alternative: for each type i, p_i must be such that:
    # v_i · x_i - p_i >= v_i · x_i' - p_i' for all i'
    # => p_i <= v_i · x_i - v_i · x_i' + p_i' for all i'
    # => p_i <= min_{i'} (v_i · x_i - v_i · x_i' + p_i')
    # Also p_i <= v_i · x_i (IR)
    # And p_i >= 0
    
    # Let's recompute payments to satisfy all constraints.
    # This is a system of linear inequalities. We can solve it iteratively.
    
    # Actually, let's just verify and if there are violations, use a conservative approach.
    
    max_violation = 0
    for i in range(n_types):
        vi = types[i]
        ui = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - payments[i]
        for i_prime in range(n_types):
            if i == i_prime:
                continue
            vi_prime = types[i_prime]
            ui_prime = vi_prime[0] * allocations[i_prime, 0] + vi_prime[1] * allocations[i_prime, 1] - payments[i_prime]
            # Check: vi · x_i - p_i >= vi · x_i' - p_i'
            # i.e., ui >= vi[0]*allocations[i_prime,0] + vi[1]*allocations[i_prime,1] - payments[i_prime]
            rhs = vi[0] * allocations[i_prime, 0] + vi[1] * allocations[i_prime, 1] - payments[i_prime]
            violation = rhs - ui
            if violation > max_violation:
                max_violation = violation
    
    if max_violation > 1e-8:
        # There are IC violations. Let's try to fix by adjusting payments.
        # For each type i, set p_i = min(v_i · x_i, min_{i'} (v_i · x_i' - p_i' + v_i · x_i))
        # This is circular. Let's use a simpler approach: set payments to ensure IR,
        # and hope IC is satisfied. If not, reduce allocations.
        
        # Conservative fix: set each payment to the maximum allowed by IR and IC.
        # p_i <= v_i · x_i
        # p_i <= v_i · x_i - v_i · x_i' + p_i' for all i'
        
        # We'll iterate a few times.
        for _ in range(10):
            for i in range(n_types):
                vi = types[i]
                max_pay = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1]
                for i_prime in range(n_types):
                    if i == i_prime:
                        continue
                    vi_prime = types[i_prime]
                    # p_i <= v_i · x_i - v_i · x_i' + p_i'
                    bound = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - \
                            vi[0] * allocations[i_prime, 0] - vi[1] * allocations[i_prime, 1] + \
                            payments[i_prime]
                    max_pay = min(max_pay, bound)
                max_pay = max(0, max_pay)
                payments[i] = max_pay
        
        # Re-check
        max_violation2 = 0
        for i in range(n_types):
            vi = types[i]
            ui = vi[0] * allocations[i, 0] + vi[1] * allocations[i, 1] - payments[i]
            for i_prime in range(n_types):
                if i == i_prime:
                    continue
                rhs = vi[0] * allocations[i_prime, 0] + vi[1] * allocations[i_prime, 1] - payments[i_prime]
                violation = rhs - ui
                if violation > max_violation2:
                    max_violation2 = violation
        
        if max_violation2 > 1e-8:
            # Still violations. Use a very conservative mechanism: no allocation.
            allocations = np.zeros((n_types, 2))
            payments = np.zeros(n_types)
    
    return allocations, payments


def _simple_feasible_solution(types, probs):
    """Return a simple feasible solution: no allocations, zero payments."""
    n_types = len(types)
    allocations = np.zeros((n_types, 2))
    payments = np.zeros(n_types)
    return allocations, payments
# EVOLVE-BLOCK-END
