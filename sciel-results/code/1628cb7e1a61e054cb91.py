"""Incentive-compatible finite auction design with LP-based optimal mechanism."""
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
    deadline = start_time + 75  # 75 second deadline with 15s safety margin

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

    # Solve the direct LP with all IC/IR constraints explicitly enumerated
    # Variables: x1[0..15], x2[16..31], p[32..47]
    # Total 48 variables
    n_vars = 3 * n_reports  # 48

    # Objective: maximize sum(probs[i] * p_i) -> minimize -sum(probs[i] * p_i)
    c = np.zeros(n_vars)
    for i in range(n_reports):
        c[2 * n_reports + i] = -probs[i]

    A_rows = []
    b_rows = []

    # IR Constraints: v1_i * x1_i + v2_i * x2_i - p_i >= 0
    # => -v1_i * x1_i - v2_i * x2_i + p_i <= 0
    if ir_required:
        for i in range(n_reports):
            v1, v2 = values[i]
            row = np.zeros(n_vars)
            row[i] = -v1
            row[n_reports + i] = -v2
            row[2 * n_reports + i] = 1
            A_rows.append(row)
            b_rows.append(0)

    # IC Constraints: for true type i, reported type j:
    # v_i . x_i - p_i >= v_i . x_j - p_j
    # => v_i . x_i - p_i - v_i . x_j + p_j <= 0
    # => -v_i . x_i + v_i . x_j - p_i + p_j <= 0  (rearranged for linprog form)
    # Actually: v_i . x_i - p_i >= v_i . x_j - p_j
    # v_i . x_i - v_i . x_j - p_i + p_j >= 0
    # => -(v_i . x_i - v_i . x_j - p_i + p_j) <= 0
    # => -v_i . x_i + v_i . x_j + p_i - p_j <= 0

    for i in range(n_reports):
        v1_i, v2_i = values[i]
        for j in range(n_reports):
            if i == j:
                continue
            row = np.zeros(n_vars)
            # -v1_i * x1_i
            row[i] -= v1_i
            # -v2_i * x2_i
            row[n_reports + i] -= v2_i
            # +v1_i * x1_j
            row[j] += v1_i
            # +v2_i * x2_j
            row[n_reports + j] += v2_i
            # +p_i
            row[2 * n_reports + i] += 1
            # -p_j
            row[2 * n_reports + j] -= 1
            A_rows.append(row)
            b_rows.append(0)

    A_ub = np.array(A_rows) if A_rows else None
    b_ub = np.array(b_rows) if b_rows else None

    # Bounds: 0 <= x1, x2 <= 1; 0 <= p
    bounds = [(0, 1)] * n_reports + [(0, 1)] * n_reports + [(0, None)] * n_reports

    # Solve LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs',
                         options={'presolve': True, 'primal_feasibility_tolerance': 1e-9,
                                  'dual_feasibility_tolerance': 1e-9})
    except Exception:
        return _simple_fallback(probs, values, n_reports)

    if not result.success:
        # Fallback: simple mechanism
        return _simple_fallback(probs, values, n_reports)

    # Extract solution
    x1 = result.x[:n_reports]
    x2 = result.x[n_reports:2 * n_reports]
    p_sol = result.x[2 * n_reports:]

    allocations = np.column_stack([x1, x2])
    allocations = np.clip(allocations, 0, 1)
    p_sol = np.clip(p_sol, 0, None)

    # Verify constraints
    if not _verify_constraints(allocations, p_sol, probs, values, n_reports, regret_bound, ir_required):
        # Try to fix or use fallback
        allocations, p_sol = _fix_constraints(allocations, p_sol, probs, values, n_reports, regret_bound, ir_required)
        if not _verify_constraints(allocations, p_sol, probs, values, n_reports, regret_bound, ir_required):
            return _simple_fallback(probs, values, n_reports)

    return {
        'allocations': allocations,
        'payments': p_sol
    }


def _parse_distribution(distribution_spec, n_reports):
    """Parse distribution_spec into a probability vector of length n_reports."""
    if distribution_spec is None:
        return np.ones(n_reports) / n_reports

    if isinstance(distribution_spec, (list, tuple, np.ndarray)):
        arr = np.array(distribution_spec, dtype=float)
        if arr.ndim == 1:
            if len(arr) == n_reports:
                total = arr.sum()
                if total > 0:
                    return arr / total
                return np.ones(n_reports) / n_reports
            elif len(arr) == 16:
                total = arr.sum()
                if total > 0:
                    return arr / total
                return np.ones(n_reports) / n_reports
            else:
                # Try to reshape or pad
                if len(arr) < n_reports:
                    new_probs = np.zeros(n_reports)
                    new_probs[:len(arr)] = arr
                    total = new_probs.sum()
                    if total > 0:
                        return new_probs / total
                    return np.ones(n_reports) / n_reports
                else:
                    return arr[:n_reports] / arr[:n_reports].sum()
        elif arr.ndim == 2:
            if arr.shape == (4, 4):
                flat = arr.flatten()
                total = flat.sum()
                if total > 0:
                    return flat / total
                return np.ones(n_reports) / n_reports
            elif arr.shape[0] == n_reports:
                # Maybe rows are (v1, v2, prob)
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
        try:
            result = distribution_spec()
            return _parse_distribution(result, n_reports)
        except:
            pass
        try:
            result = distribution_spec(range(4), range(4))
            return _parse_distribution(result, n_reports)
        except:
            pass

    return np.ones(n_reports) / n_reports


def _verify_constraints(allocations, payments, probs, values, n_reports, regret_bound, ir_required):
    """Verify IC, IR, feasibility, and nonnegative payments."""
    if np.any(allocations < -1e-9) or np.any(allocations > 1 + 1e-9):
        return False
    if np.any(payments < -1e-9):
        return False

    if ir_required:
        for j in range(n_reports):
            v1_j, v2_j = values[j]
            utility = v1_j * allocations[j, 0] + v2_j * allocations[j, 1] - payments[j]
            if utility < -1e-9:
                return False

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
