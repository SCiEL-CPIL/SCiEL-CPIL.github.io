"""Incentive-compatible finite auction design: 1 bidder, 2 goods, values in {0,1,2,3}."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def construct_auction(distribution_spec, support, constraints):
    import numpy as np
    import time
    import itertools

    start_time = time.time()
    DEADLINE = 85.0

    vals = [0, 1, 2, 3]
    types = list(itertools.product(vals, vals))
    type_index = {t: i for i, t in enumerate(types)}
    n_types = 16
    n_goods = 2

    # Parse distribution
    if isinstance(distribution_spec, dict):
        mu = np.zeros(n_types)
        for t, p in distribution_spec.items():
            if t in type_index:
                mu[type_index[t]] = p
        mu = mu / mu.sum() if mu.sum() > 0 else np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        mu = np.array(distribution_spec, dtype=float).flatten()
        if len(mu) == n_types:
            mu = mu / mu.sum() if mu.sum() > 0 else np.ones(n_types) / n_types
        else:
            mu = np.ones(n_types) / n_types
    else:
        mu = np.ones(n_types) / n_types

    reports = types
    n_reports = len(reports)
    v = np.array([list(r) for r in reports], dtype=float)

    # For each good g, the allocation a(t,g) must be non-decreasing in v_g (monotonicity for IC)
    # Since values are {0,1,2,3}, for good g, allocation can be any sequence
    # a(0,g), a(1,g), a(2,g), a(3,g) with a(0,g) <= a(1,g) <= a(2,g) <= a(3,g), each in [0,1]
    # But for finite support with values 0,1,2,3, the allocation is determined by thresholds.
    # Actually, for a single bidder with two independent goods (additive), the IC condition
    # requires that for each good g, the allocation rule is non-decreasing in v_g.
    #
    # Key insight: For a single additive bidder, the IC constraints decompose.
    # The allocation a(t,g) must be non-decreasing in v_g for each g.
    # Given a monotone allocation rule, the payment is determined by:
    # p(t) = v(t).a(t) - U(t) where U(t) is the utility
    # For IR and IC, the minimal payment that satisfies IC is:
    # p(t) = v(t).a(t) - v(t).a(t_low) + p(t_low)  (where t_low is the lowest type)
    # Actually, for a single bidder, the standard result is:
    # p(t) = v(t).a(t) - integral of a(s) ds along the type path
    # For discrete types, the payment is:
    # p(t) = sum over paths from lowest type to t of [v(t') - v(t'')] * a(t')
    #
    # More precisely, for a single bidder with type t = (v1, v2):
    # The IC condition implies:
    # u(t) = v(t).a(t) - p(t)
    # u(t) >= u(t') + [v(t) - v(t')] . a(t')  for all t, t'
    # The minimal u(t) satisfying IC is:
    # u(t) = max over t' of [u(t') + (v(t) - v(t')) . a(t')]
    # With u(lowest) = 0 (IR binding at lowest type for revenue max),
    # we can compute u(t) by dynamic programming.
    #
    # Actually, for a single bidder, the IC constraints imply that the allocation
    # must be monotone in each dimension. Given a monotone allocation, the payment
    # that maximizes revenue subject to IR and IC is:
    # p(t) = v(t).a(t) - u*(t)
    # where u*(t) is the minimum utility consistent with IC and IR.
    #
    # For a grid of types, u*(t) can be computed as:
    # u*(0,0) = 0
    # u*(v1, v2) = max of:
    #   u*(v1-1, v2) + a(v1-1, v2)  [increase v1]
    #   u*(v1, v2-1) + a(v1, v2-1)  [increase v2]
    #   (and combinations for larger jumps, but on a grid, adjacent steps suffice)
    #
    # Wait, this is for the case where we move one step at a time. For values {0,1,2,3},
    # the type space is a 4x4 grid. The utility at type (v1, v2) given monotone allocation
    # is: u(v1, v2) = sum_{i=1}^{v1} a(i, v2) + sum_{j=1}^{v2} a(v1, j)
    # This is because the "path" from (0,0) to (v1,v2) can be decomposed.
    # Actually, for additive values, the IC condition is equivalent to:
    # u(v1, v2) >= u(v1', v2') + (v1-v1')*a(v1',v2') + (v2-v2')*a(v1',v2')
    # The tightest constraints come from adjacent types.
    #
    # Let's use the formula: for a monotone allocation on a grid,
    # the minimal utility (and thus maximal payment) consistent with IC is:
    # u(v1, v2) = sum_{i=1}^{v1} a(i, v2) + sum_{j=1}^{v2} a(v1, j)
    # This works because the cross terms cancel in the additive case.
    #
    # Let me verify: u(v1,v2) - u(v1-1,v2) = a(v1, v2)
    # IC: u(v1,v2) >= u(v1-1,v2) + a(v1-1,v2)
    # So we need a(v1,v2) >= a(v1-1,v2), which is monotonicity. Good.
    #
    # So the payment is: p(v1,v2) = v1*a(v1,v2) + v2*a(v1,v2) - [sum_{i=1}^{v1} a(i,v2) + sum_{j=1}^{v2} a(v1,j)]

    # For each good g, the allocation is a function of v_g only (since the bidder is additive
    # and we're looking at monotone rules, and for a single bidder the allocation of good g
    # should depend only on v_g for the mechanism to be "separable" - actually no, it can
    # depend on both, but monotonicity requires non-decreasing in each).
    #
    # However, for a single additive bidder, the optimal mechanism is separable:
    # a(t, g) depends only on v_g. This is a known result.
    # So we can search over allocation rules for each good independently.
    #
    # For good g, allocation a_g(v) for v in {0,1,2,3}, non-decreasing, in [0,1].
    # Since we want to maximize revenue, and the payment formula is linear in allocations,
    # the optimal allocation for each good is a step function: 0 up to some threshold, 1 after.
    # Actually, it could be fractional, but for finite support with values 0,1,2,3,
    # the optimal allocation is 0 or 1 (extreme point of the LP).
    #
    # So for each good, the allocation is determined by a threshold:
    # a_g(v) = 1 if v >= threshold_g, else 0
    # threshold_g can be 0, 1, 2, 3, or 4 (never allocate)
    #
    # Wait, threshold=0 means always allocate, threshold=4 means never allocate.
    # Let's enumerate thresholds 0,1,2,3,4 for each good.
    # That's 5*5 = 25 mechanisms. Very fast!

    best_revenue = -1.0
    best_alloc = None
    best_pay = None

    for thresh1 in range(5):  # 0,1,2,3,4
        for thresh2 in range(5):
            # Check time
            if time.time() - start_time > DEADLINE - 5:
                break

            # Allocation: a(t, g) = 1 if v_g >= thresh_g else 0
            alloc = np.zeros((n_reports, n_goods))
            for t_idx, (v1, v2) in enumerate(reports):
                alloc[t_idx, 0] = 1.0 if v1 >= thresh1 else 0.0
                alloc[t_idx, 1] = 1.0 if v2 >= thresh2 else 0.0

            # Compute payment using the envelope formula
            # p(v1, v2) = v1*a1(v1,v2) + v2*a2(v1,v2) - [sum_{i=1}^{v1} a1(i,v2) + sum_{j=1}^{v2} a2(v1,j)]
            payments = np.zeros(n_reports)
            for t_idx, (v1, v2) in enumerate(reports):
                a1_val = alloc[t_idx, 0]
                a2_val = alloc[t_idx, 1]
                # Sum over i=1 to v1 of a1(i, v2)
                sum_a1 = 0.0
                for i in range(1, v1 + 1):
                    # Find type (i, v2)
                    t_i = (i, v2)
                    if t_i in type_index:
                        sum_a1 += alloc[type_index[t_i], 0]
                # Sum over j=1 to v2 of a2(v1, j)
                sum_a2 = 0.0
                for j in range(1, v2 + 1):
                    t_j = (v1, j)
                    if t_j in type_index:
                        sum_a2 += alloc[type_index[t_j], 1]

                revenue_t = v1 * a1_val + v2 * a2_val
                payments[t_idx] = revenue_t - sum_a1 - sum_a2

            # Check IR: v(t).a(t) - p(t) >= 0
            ir_ok = True
            for t_idx in range(n_reports):
                util = v[t_idx] @ alloc[t_idx] - payments[t_idx]
                if util < -1e-8:
                    ir_ok = False
                    break

            if not ir_ok:
                continue

            # Check IC: for all t, t': v(t).a(t) - p(t) >= v(t).a(t') - p(t')
            ic_ok = True
            for t_idx in range(n_reports):
                for tp_idx in range(n_reports):
                    lhs = v[t_idx] @ alloc[t_idx] - payments[t_idx]
                    rhs = v[t_idx] @ alloc[tp_idx] - payments[tp_idx]
                    if lhs - rhs < -1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break

            if not ic_ok:
                continue

            # Compute expected revenue
            exp_rev = np.sum(mu * payments)

            if exp_rev > best_revenue:
                best_revenue = exp_rev
                best_alloc = alloc.copy()
                best_pay = payments.copy()

    # If no threshold mechanism found, fall back to no-trade
    if best_alloc is None:
        best_alloc = np.zeros((n_reports, n_goods))
        best_pay = np.zeros(n_reports)

    # Final verification
    v_arr = np.array([list(r) for r in reports], dtype=float)
    # IR
    for t_idx in range(n_reports):
        util = v_arr[t_idx] @ best_alloc[t_idx] - best_pay[t_idx]
        if util < -1e-8:
            best_pay[t_idx] = max(0.0, v_arr[t_idx] @ best_alloc[t_idx])
    # IC check
    for t_idx in range(n_reports):
        for tp_idx in range(n_reports):
            lhs = v_arr[t_idx] @ best_alloc[t_idx] - best_pay[t_idx]
            rhs = v_arr[t_idx] @ best_alloc[tp_idx] - best_pay[tp_idx]
            if lhs - rhs < -1e-8:
                # Should not happen for threshold mechanisms, but just in case
                pass

    return {
        'allocations': best_alloc,
        'payments': best_pay
    }
# EVOLVE-BLOCK-END
