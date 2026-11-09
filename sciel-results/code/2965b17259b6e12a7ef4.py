import time
import numpy as np
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
    t0 = time.time()
    DEADLINE = 85.0  # 5 s safety margin under the 90 s evaluator limit

    # ------------------------------------------------------------------
    # 1. Build the problem from the spec (independent of the baseline)
    # ------------------------------------------------------------------
    # distribution_spec: one of "uniform", "decreasing", "increasing" (three fixed distributions)
    # support: list of (v1, v2) value pairs (each in {0,1,2,3})
    # constraints: dict that may carry extra flags; we keep the public contract fixed.

    # Enumerate all allowed reports (each report is a value pair (r1, r2) in the support)
    # The task says: "One additive bidder, two goods, each value in {0,1,2,3}, three fixed distributions.
    #   Every allowed report is enumerated".
    # We treat `support` as the list of (v1, v2) pairs that are possible values/reports.
    # If support is None, default to the full grid.
    if support is None or len(support) == 0:
        vals = list(range(4))
        support = [(v1, v2) for v1 in vals for v2 in vals]
    support = list(support)
    n = len(support)

    # Distribution over types (each type is an index into support)
    # Three fixed distributions:
    #   "uniform":    uniform over support
    #   "decreasing": probability proportional to (3 - v1 - v2) + 1  (or a fixed decreasing schedule)
    #   "increasing": probability proportional to (v1 + v2) + 1
    # We'll define them deterministically.
    if isinstance(distribution_spec, str):
        dist_name = distribution_spec.lower()
    else:
        dist_name = "uniform"

    if dist_name == "uniform":
        probs = np.ones(n, dtype=float) / n
    elif dist_name == "decreasing":
        w = np.array([1.0 / (1 + v1 + v2) for (v1, v2) in support])
        probs = w / w.sum()
    elif dist_name == "increasing":
        w = np.array([1.0 + v1 + v2 for (v1, v2) in support])
        probs = w / w.sum()
    else:
        probs = np.ones(n, dtype=float) / n

    # Additive values: value_i(r) = r1 + r2
    values = np.array([v1 + v2 for (v1, v2) in support], dtype=float)
    # Individual value components
    v1_arr = np.array([v1 for (v1, v2) in support], dtype=float)
    v2_arr = np.array([v2 for (v1, v2) in support], dtype=float)

    # ------------------------------------------------------------------
    # 2. Mechanism variables
    #    allocations: shape (n, 2) -- probability of allocating each good to the bidder for each report
    #    payments: shape (n,) -- expected payment for each report
    # ------------------------------------------------------------------
    # Decision variables: x1[r], x2[r] (allocation probs for good 1 and 2 at report r), p[r] (payment)
    # Total variables: 2*n (allocations) + n (payments) = 3*n

    # Constraints:
    # (a) Feasible probabilities: 0 <= x1[r] <= 1, 0 <= x2[r] <= 1
    # (b) Nonnegative payments: p[r] >= 0
    # (c) Individual rationality: for every type t, expected utility >= 0
    #     u(t | r=t) = value_t * (x1[t] + x2[t]) - p[t] >= 0
    #     i.e., p[t] <= value_t * (x1[t] + x2[t])
    # (d) Incentive compatibility (truthfulness): for every type t and every report r,
    #     u(t | r=t) >= u(t | r)
    #     u(t | r) = value_t * (x1[r] + x2[r]) - p[r]
    #     So: value_t*(x1[t]+x2[t]) - p[t] >= value_t*(x1[r]+x2[r]) - p[r]
    #     => value_t*(x1[t]+x2[t]) - value_t*(x1[r]+x2[r]) >= p[t] - p[r]
    #     => p[t] - p[r] <= value_t*(x1[t]+x2[t] - x1[r] - x2[r])
    #     for all t, r.
    #     Also the symmetric form is implied by t,r swap.
    #
    # We also need the "regret <= 1e-8" constraint which is the same as IC (no benefit from misreporting).
    # The task says "regret<=1e-8" which we interpret as the IC constraint with tolerance.
    # For LP, we enforce exact IC (which satisfies regret <= 1e-8).

    # ------------------------------------------------------------------
    # 3. Objective: maximize expected revenue
    #    Revenue = sum_t probs[t] * p[t]
    #    But the task says: "Maximize expected revenue normalized by an independently solved finite-support mechanism LP."
    #    This is ambiguous. Let me re-read.
    #    "Maximize expected revenue normalized by an independently solved finite-support mechanism LP.
    #     Truthfulness holds only on the declared report/value support."
    #    I think the normalization is by the optimal revenue of the LP. But the evaluator measures raw value.
    #    Actually, the task says "The evaluator measures the raw value (higher is better)."
    #    So I think we just maximize expected revenue. The "normalized by" might be a description of the metric,
    #    but the raw value is what's measured. Let me just maximize expected revenue.
    # ------------------------------------------------------------------

    # LP formulation:
    # Variables: x1[0..n-1], x2[0..n-1], p[0..n-1]
    # c = [0]*n + [0]*n + [-probs]  (maximize sum probs[t]*p[t], so minimize -sum)
    #
    # Constraints:
    # (a) x1[r] <= 1 for all r  => x1[r] <= 1
    #     x2[r] <= 1 for all r  => x2[r] <= 1
    #     -x1[r] <= 0, -x2[r] <= 0 (nonnegativity, handled by bounds)
    #
    # (b) p[r] >= 0 (handled by bounds)
    #
    # (c) IR: p[t] - value_t*(x1[t]+x2[t]) <= 0 for all t
    #     => -value_t*x1[t] - value_t*x2[t] + p[t] <= 0
    #
    # (d) IC: for all t, r:
    #     p[t] - p[r] <= value_t*(x1[t] + x2[t] - x1[r] - x2[r])
    #     => value_t*x1[t] + value_t*x2[t] - value_t*x1[r] - value_t*x2[r] + p[t] - p[r] <= 0
    #     Wait, let me re-derive.
    #     IC: value_t*(x1[t]+x2[t]) - p[t] >= value_t*(x1[r]+x2[r]) - p[r]
    #     => value_t*(x1[t]+x2[t]) - p[t] - value_t*(x1[r]+x2[r]) + p[r] >= 0
    #     => -value_t*(x1[t]+x2[t]) + p[t] + value_t*(x1[r]+x2[r]) - p[r] <= 0
    #     So the constraint is:
    #     -value_t*x1[t] - value_t*x2[t] + p[t] + value_t*x1[r] + value_t*x2[r] - p[r] <= 0
    #
    # Let me build the LP matrix.

    # Variable order: x1[0..n-1], x2[0..n-1], p[0..n-1]
    nv = 3 * n

    # Objective: minimize -sum_t probs[t] * p[t]
    c = np.zeros(nv)
    c[2*n:] = -probs  # p variables

    # Inequality constraints: A_ub @ vars <= b_ub
    # We'll collect all constraints.

    A_rows = []
    b_rows = []

    # (a) x1[r] <= 1
    for r in range(n):
        row = np.zeros(nv)
        row[r] = 1.0  # x1[r]
        A_rows.append(row)
        b_rows.append(1.0)
    # x2[r] <= 1
    for r in range(n):
        row = np.zeros(nv)
        row[n + r] = 1.0  # x2[r]
        A_rows.append(row)
        b_rows.append(1.0)

    # (c) IR: -value_t*x1[t] - value_t*x2[t] + p[t] <= 0
    for t in range(n):
        row = np.zeros(nv)
        row[t] = -values[t]       # x1[t]
        row[n + t] = -values[t]   # x2[t]
        row[2*n + t] = 1.0        # p[t]
        A_rows.append(row)
        b_rows.append(0.0)

    # (d) IC: for all t, r:
    #     -value_t*x1[t] - value_t*x2[t] + p[t] + value_t*x1[r] + value_t*x2[r] - p[r] <= 0
    for t in range(n):
        for r in range(n):
            if t == r:
                continue  # trivially satisfied
            row = np.zeros(nv)
            row[t] = -values[t]          # x1[t]
            row[n + t] = -values[t]      # x2[t]
            row[2*n + t] = 1.0           # p[t]
            row[r] = values[t]           # x1[r]
            row[n + r] = values[t]       # x2[r]
            row[2*n + r] = -1.0          # p[r]
            A_rows.append(row)
            b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds: all variables >= 0, x1,x2 <= 1, p unbounded above (but constrained by IR)
    # Actually p >= 0, x1 >= 0, x2 >= 0
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, None)] * n

    # Check time
    if time.time() - t0 > DEADLINE:
        # Fallback: return a simple mechanism
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {"allocations": allocations, "payments": payments}

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

    if time.time() - t0 > DEADLINE:
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {"allocations": allocations, "payments": payments}

    if not res.success:
        # Fallback: return a simple mechanism
        allocations = np.zeros((n, 2))
        payments = np.zeros(n)
        return {"allocations": allocations, "payments": payments}

    x = res.x
    x1 = x[:n]
    x2 = x[n:2*n]
    p = x[2*n:]

    # Clamp small numerical errors
    x1 = np.clip(x1, 0.0, 1.0)
    x2 = np.clip(x2, 0.0, 1.0)
    p = np.clip(p, 0.0, None)

    allocations = np.column_stack([x1, x2])
    payments = p

    # Verify constraints with tolerance
    # IR check
    for t in range(n):
        util = values[t] * (x1[t] + x2[t]) - p[t]
        if util < -1e-8:
            # Fix by reducing payment
            p[t] = max(0.0, values[t] * (x1[t] + x2[t]))
    payments = p

    # IC check
    for t in range(n):
        for r in range(n):
            u_t_t = values[t] * (x1[t] + x2[t]) - p[t]
            u_t_r = values[t] * (x1[r] + x2[r]) - p[r]
            if u_t_t < u_t_r - 1e-8:
                # Violation: need to fix. Reduce p[t] or adjust allocations.
                # Simple fix: reduce p[t] to make u_t_t >= u_t_r
                needed = u_t_r - values[t] * (x1[t] + x2[t])
                if needed < p[t]:
                    p[t] = max(0.0, needed)
                # Re-check IR
                util = values[t] * (x1[t] + x2[t]) - p[t]
                if util < 0:
                    p[t] = max(0.0, values[t] * (x1[t] + x2[t]))
    payments = p

    # Final clamp
    payments = np.clip(payments, 0.0, None)

    return {"allocations": allocations, "payments": payments}
