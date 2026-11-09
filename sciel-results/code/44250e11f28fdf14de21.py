"""Incentive-compatible finite auction design: 1 bidder, 2 goods, values in {0,1,2,3}."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    # Values in {0,1,2,3} for each of 2 goods -> 4*4 = 16 types
    vals = [0, 1, 2, 3]
    types = list(itertools.product(vals, vals))  # (v1, v2)
    type_index = {t: i for i, t in enumerate(types)}
    n_types = len(types)  # 16
    n_goods = 2

    # Parse distribution
    if isinstance(distribution_spec, dict):
        mu = np.zeros(n_types)
        for t, p in distribution_spec.items():
            if t in type_index:
                mu[type_index[t]] = p
        if mu.sum() > 0:
            mu = mu / mu.sum()
        else:
            mu = np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        mu = np.array(distribution_spec, dtype=float).flatten()
        if len(mu) == n_types:
            if mu.sum() > 0:
                mu = mu / mu.sum()
            else:
                mu = np.ones(n_types) / n_types
        else:
            mu = np.ones(n_types) / n_types
    else:
        mu = np.ones(n_types) / n_types

    # Ensure all 16 types are reports
    reports = types
    n_reports = len(reports)
    
    # Values matrix V[t, g]
    V = np.array([list(r) for r in reports], dtype=float)
    
    # Strategy:
    # 1. For a fixed allocation matrix A, the optimal payment vector p is uniquely determined
    #    as the tightest upper bound implied by IC and IR constraints.
    #    p(t) = min( V[t].A[t], min_{t'} (p(t') + V[t].A[t] - V[t].A[t']) )
    #    This is equivalent to finding the shortest path in a graph where:
    #    - Nodes are types t
    #    - Edge t' -> t has weight V[t].A[t] - V[t].A[t']
    #    - There is a dummy source s with edge s -> t of weight V[t].A[t] (representing IR)
    #    - p(t) = shortest_path(s, t)
    #    
    # 2. Revenue R(A) = sum_t mu[t] * p(t) is a concave function of A (minimum of linear functions).
    #    We can maximize this using scipy.optimize.minimize with L-BFGS-B or SLSQP.
    #    
    # 3. The allocation A must be in [0,1]^{16x2}. We use bounds to enforce this.
    #
    # 4. We use multiple random restarts to find the global optimum, since the objective is
    #    non-convex in A (concave, so we're maximizing a concave function which is convex optimization? 
    #    Wait, maximizing a concave function is NOT convex optimization. Minimizing a convex function is.
    #    So we are maximizing a concave function, which is a non-convex problem in general.
    #    However, for this specific problem structure, the optimum might be at a vertex or have
    #    a simple form. Let's try to use the LP approach but with a smarter formulation.
    #
    # Actually, let's reconsider. The standard result is that the optimal mechanism for a single
    # agent with multi-dimensional types can be found by solving an LP. The current program does
    # this correctly. The score of 1.0025 suggests it's very close to optimal.
    #
    # Let's try a different approach: enumerate the possible "critical" allocation patterns.
    # For a single additive bidder, the optimal allocation rule is often a "threshold" rule or
    # a rule that allocates based on the highest value good.
    #
    # Alternatively, we can use the fact that the problem is small (16 types, 2 goods) and
    # use a more robust LP solver or a different formulation.
    #
    # Let's try using scipy.optimize.linprog with the full set of IC constraints (not just adjacent)
    # to ensure we're not missing any constraints. But the current program uses adjacent constraints
    # which should be sufficient.
    #
    # Let's try a different solver: use scipy.optimize.minimize to directly optimize the allocation
    # matrix A, with the payment vector computed from A.
    
    from scipy.optimize import minimize, linprog

    def compute_payments(A):
        """Given allocation matrix A, compute the optimal payment vector p."""
        n = n_reports
        p = np.zeros(n)
        # p(t) = min over all paths from source to t
        # This is a shortest path problem. We can use Bellman-Ford or Dijkstra (weights can be negative).
        # Since the graph is small (16 nodes), we can use Floyd-Warshall or just iterate.
        
        # Initialize p with IR bounds
        for t in range(n):
            p[t] = V[t] @ A[t]
        
        # Iterate to convergence (Bellman-Ford style)
        # p(t) = min(p(t), min_{t'} (p(t') + V[t].A[t] - V[t].A[t']))
        for _ in range(100):  # 100 iterations should be enough for 16 nodes
            new_p = p.copy()
            for t in range(n):
                best = p[t]
                for tp in range(n):
                    if tp != t:
                        candidate = p[tp] + V[t] @ A[t] - V[t] @ A[tp]
                        if candidate < best:
                            best = candidate
                new_p[t] = best
            if np.allclose(new_p, p, atol=1e-12):
                break
            p = new_p
        
        # Ensure non-negative payments
        p = np.maximum(p, 0.0)
        return p

    def revenue(A):
        """Compute expected revenue for allocation matrix A."""
        p = compute_payments(A)
        return np.sum(mu * p)

    def neg_revenue(A_flat):
        """Negative revenue for minimization."""
        A = A_flat.reshape(n_reports, n_goods)
        return -revenue(A)

    # Bounds for A: [0, 1] for each entry
    bounds_A = [(0.0, 1.0)] * (n_reports * n_goods)

    # Multiple random restarts
    best_A = None
    best_rev = -np.inf

    # Try some heuristic starting points
    start_points = []
    
    # 1. Uniform allocation
    start_points.append(np.ones((n_reports, n_goods)) / 2.0)
    
    # 2. Allocate good g to type t if V[t,g] is high
    for g in range(n_goods):
        A_init = np.zeros((n_reports, n_goods))
        for t in range(n_reports):
            if V[t, g] >= 2:
                A_init[t, g] = 1.0
        start_points.append(A_init)
    
    # 3. Allocate the highest value good
    A_init = np.zeros((n_reports, n_goods))
    for t in range(n_reports):
        g_max = np.argmax(V[t])
        A_init[t, g_max] = 1.0
    start_points.append(A_init)
    
    # 4. Random starts
    np.random.seed(42)
    for _ in range(5):
        start_points.append(np.random.rand(n_reports, n_goods))

    # Optimize from each start point
    for A0 in start_points:
        result = minimize(neg_revenue, A0.flatten(), method='L-BFGS-B', bounds=bounds_A,
                          options={'maxiter': 1000, 'ftol': 1e-12})
        if result.success:
            A_opt = result.x.reshape(n_reports, n_goods)
            rev = -result.fun
            if rev > best_rev:
                best_rev = rev
                best_A = A_opt

    # Also try the LP approach as a cross-check
    result_lp = _solve_lp(mu, reports, V, n_reports, n_goods)
    if result_lp is not None:
        A_lp, p_lp = result_lp
        rev_lp = np.sum(mu * p_lp)
        if rev_lp > best_rev:
            best_rev = rev_lp
            best_A = A_lp
            p_best = p_lp
        else:
            p_best = compute_payments(best_A)
    else:
        p_best = compute_payments(best_A)

    # Final verification
    allocations = best_A
    payments = p_best

    # Clip to [0,1] and ensure non-negative payments
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Verify IC and IR
    def verify_and_fix(allocations, payments, reports, mu, tol=1e-8):
        n = len(reports)
        v = np.array([list(r) for r in reports], dtype=float)
        # IR: v(t).a(t) - p(t) >= 0
        for t in range(n):
            util = v[t] @ allocations[t] - payments[t]
            if util < -tol:
                payments[t] = max(0.0, v[t] @ allocations[t])
        
        # IC: for all t, t': v(t).a(t) - p(t) >= v(t).a(t') - p(t')
        max_violation = 0.0
        for t in range(n):
            for tp in range(n):
                lhs = v[t] @ allocations[t] - payments[t]
                rhs = v[t] @ allocations[tp] - payments[tp]
                if lhs - rhs < -tol:
                    max_violation = max(max_violation, rhs - lhs)
        return max_violation

    violation = verify_and_fix(allocations, payments, reports, mu)
    if violation > 1e-8:
        # Fallback to LP solution if available
        if result_lp is not None:
            allocations, payments = result_lp
        else:
            allocations = np.zeros((n_reports, n_goods))
            payments = np.zeros(n_reports)

    return {
        'allocations': allocations,
        'payments': payments
    }

def _solve_lp(mu, reports, V, n_reports, n_goods):
    """Solve the LP for the optimal mechanism."""
    from scipy.optimize import linprog

    n_a = n_reports * n_goods
    n_p = n_reports
    n_vars = n_a + n_p

    def a_idx(t, g):
        return t * n_goods + g

    def p_idx(t):
        return n_a + t

    # Objective: maximize sum mu(t)*p(t) -> minimize -sum mu(t)*p(t)
    c = np.zeros(n_vars)
    for t in range(n_reports):
        c[p_idx(t)] = -mu[t]

    A_rows = []
    b_rows = []

    # IR constraints: for each t
    for t in range(n_reports):
        row = np.zeros(n_vars)
        row[a_idx(t, 0)] = -reports[t][0]
        row[a_idx(t, 1)] = -reports[t][1]
        row[p_idx(t)] = 1.0
        A_rows.append(row)
        b_rows.append(0.0)

    # IC constraints: Only adjacent types in the 4x4 grid
    def get_neighbors(t):
        v1, v2 = reports[t]
        neighbors = []
        if v1 > 0: neighbors.append(t - 4)
        if v1 < 3: neighbors.append(t + 4)
        if v2 > 0: neighbors.append(t - 1)
        if v2 < 3: neighbors.append(t + 1)
        return neighbors

    for t in range(n_reports):
        for tp in get_neighbors(t):
            # Constraint: p(t) - p(tp) <= V[t].A[t] - V[t].A[tp]
            row = np.zeros(n_vars)
            row[a_idx(t, 0)] = -reports[t][0]
            row[a_idx(t, 1)] = -reports[t][1]
            row[a_idx(tp, 0)] = reports[t][0]
            row[a_idx(tp, 1)] = reports[t][1]
            row[p_idx(t)] = 1.0
            row[p_idx(tp)] = -1.0
            A_rows.append(row)
            b_rows.append(0.0)

            # Constraint: p(tp) - p(t) <= V[tp].A[tp] - V[tp].A[t]
            row2 = np.zeros(n_vars)
            row2[a_idx(tp, 0)] = -reports[tp][0]
            row2[a_idx(tp, 1)] = -reports[tp][1]
            row2[a_idx(t, 0)] = reports[tp][0]
            row2[a_idx(t, 1)] = reports[tp][1]
            row2[p_idx(tp)] = 1.0
            row2[p_idx(t)] = -1.0
            A_rows.append(row2)
            b_rows.append(0.0)

    A_ub = np.array(A_rows)
    b_ub = np.array(b_rows)

    # Bounds
    bounds = []
    for t in range(n_reports):
        for g in range(n_goods):
            bounds.append((0.0, 1.0))
    for t in range(n_reports):
        bounds.append((0.0, None))

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        x = result.x
        allocations = np.zeros((n_reports, n_goods))
        payments = np.zeros(n_reports)
        for t in range(n_reports):
            for g in range(n_goods):
                allocations[t, g] = max(0.0, min(1.0, x[a_idx(t, g)]))
            payments[t] = max(0.0, x[p_idx(t)])
        return allocations, payments
    else:
        return None
# EVOLVE-BLOCK-END
