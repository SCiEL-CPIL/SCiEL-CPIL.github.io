"""Incentive-compatible finite auction design with utility-based LP formulation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    """
    Construct an incentive-compatible finite auction with one additive bidder, two goods.
    Values in {0,1,2,3} for each good, so 16 possible reports.
    Returns dict with allocations:(16,2) and payments:(16,)
    """
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with 5s safety margin
    
    # The report space: (v1, v2) where v1, v2 in {0,1,2,3}
    # Index: i = v1 * 4 + v2
    values = np.array([(v1, v2) for v1 in range(4) for v2 in range(4)], dtype=float)
    n_reports = 16
    
    # Parse distribution_spec into a probability vector over 16 reports
    probs = _parse_distribution(distribution_spec, n_reports)
    
    # Parse constraints
    regret_bound = 1e-8
    ir_required = True
    if constraints is not None and isinstance(constraints, dict):
        regret_bound = constraints.get('regret_bound', 1e-8)
        ir_required = constraints.get('ir_required', True)
    
    # Utility-based LP formulation:
    # Variables: u_i (utility at report i), x_{i,g} (allocation of good g at report i)
    # For each report i, we have u_i and 2 allocation variables
    # Total variables: 16 * 3 = 48
    #
    # Key insight: For single-bidder IC, we can use the fact that truthful reporting
    # maximizes utility. The IC constraint U(j,j) >= U(i,j) for all i,j can be
    # rephrased using the utility function directly.
    #
    # Payment derivation via envelope: p_i = v_i · x_i - u_i
    # Revenue = sum_i p_i * (v_i · x_i - u_i)
    #
    # Constraints:
    # 1. IC: For all i, j: u_j >= v_j · x_i - p_i = v_j · x_i - (v_i · x_i - u_i)
    #    => u_j - u_i >= v_j · x_i - v_i · x_i = (v_j - v_i) · x_i
    #    This is 16*16 - 16 = 240 constraints (excluding i=j)
    #
    # 2. IR: u_i >= 0 for all i
    #
    # 3. Feasibility: 0 <= x_{i,g} <= 1
    #
    # 4. Nonnegative payments: p_i = v_i · x_i - u_i >= 0 => u_i <= v_i · x_i
    
    n_vars = n_reports * 3  # u_i (16) + x_{i,0} (16) + x_{i,1} (16)
    
    # Objective: maximize expected revenue = sum_i p_i * (v_i · x_i - u_i)
    # In LP form (minimize): -sum_i p_i * (v_i · x_i - u_i)
    # = -sum_i p_i * (v1_i * x_{i,0} + v2_i * x_{i,1} - u_i)
    
    c = np.zeros(n_vars)
    for i in range(n_reports):
        v1_i, v2_i = values[i]
        # Coefficient for u_i: -p_i * (-1) = +p_i (since we're minimizing -revenue)
        # Wait, let me redo this. Revenue = sum_i p_i * (v_i · x_i - u_i)
        # Minimize -Revenue = -sum_i p_i * (v1_i * x_{i,0} + v2_i * x_{i,1} - u_i)
        # Coeff of x_{i,0}: -p_i * v1_i
        # Coeff of x_{i,1}: -p_i * v2_i
        # Coeff of u_i: -p_i * (-1) = +p_i
        c[i] = probs[i]  # coefficient for u_i
        c[n_reports + i * 2 + 0] = -probs[i] * v1_i  # coefficient for x_{i,0}
        c[n_reports + i * 2 + 1] = -probs[i] * v2_i  # coefficient for x_{i,1}
    
    # Build constraints
    A_rows = []
    b_rows = []
    
    # IC constraints: u_j - u_i >= (v_j - v_i) · x_i for all i != j
    # Rearranged: u_j - u_i - (v_j - v_i) · x_i >= 0
    # In A_ub form (<=): -[u_j - u_i - (v_j - v_i) · x_i] <= 0
    for i in range(n_reports):
        v1_i, v2_i = values[i]
        for j in range(n_reports):
            if i == j:
                continue
            v1_j, v2_j = values[j]
            row = np.zeros(n_vars)
            # u_j coefficient: -1 (from -[u_j - ...])
            row[j] = -1
            # u_i coefficient: +1 (from -[-u_i])
            row[i] = 1
            # x_{i,0} coefficient: -(v1_j - v1_i)
            row[n_reports + i * 2 + 0] = -(v1_j - v1_i)
            # x_{i,1} coefficient: -(v2_j - v2_i)
            row[n_reports + i * 2 + 1] = -(v2_j - v2_i)
            A_rows.append(row)
            b_rows.append(0)
    
    # IR constraints: u_i >= 0 => -u_i <= 0
    if ir_required:
        for i in range(n_reports):
            row = np.zeros(n_vars)
            row[i] = -1
            A_rows.append(row)
            b_rows.append(0)
    
    # Nonnegative payments: u_i <= v_i · x_i => u_i - v_i · x_i <= 0
    for i in range(n_reports):
        v1_i, v2_i = values[i]
        row = np.zeros(n_vars)
        row[i] = 1
        row[n_reports + i * 2 + 0] = -v1_i
        row[n_reports + i * 2 + 1] = -v2_i
        A_rows.append(row)
        b_rows.append(0)
    
    # Build A_ub and b_ub
    if A_rows:
        A_ub = np.array(A_rows)
        b_ub = np.array(b_rows)
    else:
        A_ub = None
        b_ub = None
    
    # Bounds:
    # u_i: no explicit bounds (constrained by IR and payment nonnegativity)
    # x_{i,g}: 0 <= x <= 1
    bounds = []
    for i in range(n_reports):
        bounds.append((None, None))  # u_i unbounded
    for i in range(n_reports * 2):
        bounds.append((0, 1))  # x_{i,g} in [0,1]
    
    # Check time before solving
    if time.time() > deadline - 10:
        return _simple_fallback(probs, values, n_reports)
    
    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    except Exception:
        return _simple_fallback(probs, values, n_reports)
    
    if not result.success:
        # Fallback: simple mechanism
        return _simple_fallback(probs, values, n_reports)
    
    # Extract solution
    u_sol = result.x[:n_reports]
    x_sol = result.x[n_reports:n_reports * 3].reshape(n_reports, 2)
    
    # Derive payments: p_i = v_i · x_i - u_i
    p_sol = np.zeros(n_reports)
    for i in range(n_reports):
        v1_i, v2_i = values[i]
        p_sol[i] = v1_i * x_sol[i, 0] + v2_i * x_sol[i, 1] - u_sol[i]
    
    # Clip small negative values due to numerical errors
    x_sol = np.clip(x_sol, 0, 1)
    p_sol = np.clip(p_sol, 0, None)
    
    # Verify constraints
    if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required):
        # Try to fix
        x_sol, p_sol = _fix_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required)
        if not _verify_constraints(x_sol, p_sol, probs, values, n_reports, regret_bound, ir_required):
            return _simple_fallback(probs, values, n_reports)
    
    return {
        'allocations': x_sol,
        'payments': p_sol
    }


def _parse_distribution(distribution_spec, n_reports):
    """Parse distribution_spec into a probability vector of length n_reports."""
    if distribution_spec is None:
        # Uniform distribution
        return np.ones(n_reports) / n_reports
    
    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.shape == (n_reports,):
            # Normalize
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_reports) / n_reports
        elif len(arr) == n_reports:
            total = arr.sum()
            if total > 0:
                return arr / total
            return np.ones(n_reports) / n_reports
        elif arr.ndim == 2 and arr.shape[0] == n_reports:
            # Could be (v1, v2, prob) triples
            probs = np.zeros(n_reports)
            for row in arr:
                if len(row) >= 3:
                    v1, v2, p = int(row[0]), int(row[1]), float(row[2])
                    idx = v1 * 4 + v2
                    if 0 <= idx < n_reports:
                        probs[idx] = p
            total = probs.sum()
            if total > 0:
                return probs / total
            return np.ones(n_reports) / n_reports
    
    if isinstance(distribution_spec, dict):
        probs = np.zeros(n_reports)
        for key, val in distribution_spec.items():
            if isinstance(key, tuple) and len(key) == 2:
                v1, v2 = int(key[0]), int(key[1])
                idx = v1 * 4 + v2
                if 0 <= idx < n_reports:
                    probs[idx] = float(val)
            elif isinstance(key, int) and 0 <= key < n_reports:
                probs[key] = float(val)
        total = probs.sum()
        if total > 0:
            return probs / total
        return np.ones(n_reports) / n_reports
    
    if callable(distribution_spec):
        # Try calling it
        try:
            result = distribution_spec()
            return _parse_distribution(result, n_reports)
        except:
            pass
        # Try with arguments
        try:
            result = distribution_spec(range(4), range(4))
            return _parse_distribution(result, n_reports)
        except:
            pass
    
    # Default: uniform
    return np.ones(n_reports) / n_reports


def _verify_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Verify IC, IR, feasibility, and nonnegative payments."""
    # Feasibility: 0 <= allocations <= 1
    if np.any(allocations < -1e-9) or np.any(allocations > 1 + 1e-9):
        return False
    
    # Nonnegative payments
    if np.any(payments < -1e-9):
        return False
    
    # IR: U(j,j) >= 0 for all j
    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            utility = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
            if utility < -1e-9:
                return False
    
    # IC: U(j,j) >= U(i,j) for all i, j
    for j in range(n_reports):
        v1_j, v2_j = values[j]
        u_true = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
        for i in range(n_reports):
            if i == j:
                continue
            u_misrep = v1_j * allocations[i, 0] + v2_j * allocations[i, 1] - payments[i]
            if u_true - u_misrep < -1e-9:
                return False
    
    return True


def _fix_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Attempt to fix small constraint violations."""
    allocations = np.clip(allocations, 0, 1)
    payments = np.clip(payments, 0, None)
    return allocations, payments


def _simple_fallback(probs, values, n_reports):
    """Simple fallback: always allocate both goods, charge 0."""
    allocations = np.ones((n_reports, 2))
    payments = np.zeros(n_reports)
    return {
        'allocations': allocations,
        'payments': payments
    }
# EVOLVE-BLOCK-END
