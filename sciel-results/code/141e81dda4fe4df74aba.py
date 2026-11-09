"""Incentive-compatible finite auction design: 1 bidder, 2 goods, values in {0,1,2,3}."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import linprog

def construct_auction(distribution_spec, support, constraints):
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
        if mu.sum() == 0:
            mu = np.ones(n_types) / n_types
        else:
            mu = mu / mu.sum()
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        mu = np.array(distribution_spec, dtype=float).flatten()
        if len(mu) == n_types:
            if mu.sum() == 0:
                mu = np.ones(n_types) / n_types
            else:
                mu = mu / mu.sum()
        else:
            mu = np.ones(n_types) / n_types
    else:
        mu = np.ones(n_types) / n_types

    v = np.array([list(r) for r in types], dtype=float)

    # Variables: allocations a[t, g] for t in 0..15, g in 0..1 (32 variables)
    # Payments are determined by the allocation function via Myerson's lemma:
    # p(t) = v(t) * a(t) - sum_{t' <= t} (a(t') - a(t'' )) ... 
    # Actually, for discrete types, the payment is:
    # p(t) = v(t) * a(t) - sum_{t' < t} (v(t) - v(t')) * (a(t') - a(t''))? No.
    # The standard discrete IC payment identity for a single additive bidder is:
    # p(t) = v(t) * a(t) - \int_0^{v(t)} a(t') dt'
    # Since values are discrete integers 0,1,2,3, the integral becomes a sum.
    # Specifically, for good g:
    # p_g(t) = v_g(t) * a_g(t) - sum_{k=1}^{v_g(t)} a_g(k, t_other)
    # Where a_g(k, t_other) is the allocation of good g when the value for g is k and the other good has value t_other.
    # Wait, the allocation a_g depends on both values.
    # Let's define the payment more carefully.
    # Utility U(t) = v(t) * a(t) - p(t).
    # IC implies U(t) - U(t') >= (v(t) - v(t')) * a(t').
    # The maximal utility (and thus minimal payment) for a given allocation is:
    # U(t) = sum_{t' < t} (v(t) - v(t')) * (a(t') - a(t''))? No.
    # The envelope theorem says U(t) = U(t_min) + \int_{t_min}^t a(t') dt'.
    # With IR, U(t_min) >= 0. To maximize revenue, we set U(t_min) = 0.
    # So U(t) = \int_{0}^{t} a(s) ds.
    # In discrete case:
    # U(t) = sum over all paths from (0,0) to t of the "increments".
    # Actually, a simpler formula for discrete values:
    # p(t) = v(t) * a(t) - sum_{t' <= t, t' != t} (a(t') - a(t'' ))? 
    # Let's use the explicit sum formula for discrete types:
    # p(t) = v(t) * a(t) - sum_{k=1}^{v_1(t)} a(k, v_2(t)) - sum_{k=1}^{v_2(t)} a(v_1(t), k)
    # This formula holds if the allocation is monotone.
    # Let's verify:
    # If a is monotone, the payment that makes the bidder indifferent between t and 0 is:
    # p(t) = v(t) * a(t) - \int_0^{v(t)} a(s) ds
    # For discrete values, \int_0^{v} a(s) ds = sum_{k=1}^{v} a(k, other)
    # So p(t) = v_1(t) a_1(t) + v_2(t) a_2(t) - sum_{k=1}^{v_1(t)} a_1(k, v_2(t)) - sum_{k=1}^{v_2(t)} a_2(v_1(t), k)
    # This payment satisfies IC and IR (U(0,0)=0).
    # Revenue = sum_t mu(t) p(t).
    # We want to maximize revenue, which is equivalent to minimizing the "virtual cost" or maximizing the objective.
    # Let's formulate the LP over allocations a[t,g].
    # Objective: Maximize sum_t mu(t) [ v(t) a(t) - sum_{k=1}^{v_1(t)} a_1(k, v_2(t)) - sum_{k=1}^{v_2(t)} a_2(v_1(t), k) ]
    # This is linear in a.
    # Constraints:
    # 1. Monotonicity: a[t,g] <= a[t',g] if v_g[t'] >= v_g[t]
    # 2. 0 <= a[t,g] <= 1

    n_vars = 32
    a_idx = lambda t, g: t * 2 + g

    # Objective coefficients
    c = np.zeros(n_vars)
    for t in range(n_types):
        v1, v2 = types[t]
        # Term from v(t) * a(t)
        c[a_idx(t, 0)] += mu[t] * v1
        c[a_idx(t, 1)] += mu[t] * v2
        
        # Term from - sum_{k=1}^{v1} a_1(k, v2)
        for k in range(1, v1 + 1):
            t_k = (k, v2)
            c[a_idx(type_index[t_k], 0)] -= mu[t]
            
        # Term from - sum_{k=1}^{v2} a_2(v1, k)
        for k in range(1, v2 + 1):
            t_k = (v1, k)
            c[a_idx(type_index[t_k], 1)] -= mu[t]

    # linprog minimizes, so we use c = -objective
    c = -c

    # Constraints: Monotonicity
    A_ub = []
    b_ub = []
    for t in range(n_types):
        v_t = types[t]
        for g in range(n_goods):
            for t_prime in range(n_types):
                v_tp = types[t_prime]
                if v_tp[g] > v_t[g]:
                    # a[t, g] - a[t_prime, g] <= 0
                    row = np.zeros(n_vars)
                    row[a_idx(t, g)] = 1.0
                    row[a_idx(t_prime, g)] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bounds
    bounds = [(0, 1)] * n_vars

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            # Extract allocations
            alloc = np.zeros((n_types, n_goods))
            for t in range(n_types):
                for g in range(n_goods):
                    alloc[t, g] = res.x[a_idx(t, g)]
            
            # Compute payments using the Myerson identity
            payments = np.zeros(n_types)
            for t in range(n_types):
                v1, v2 = types[t]
                p = v1 * alloc[t, 0] + v2 * alloc[t, 1]
                for k in range(1, v1 + 1):
                    t_k = (k, v2)
                    p -= alloc[type_index[t_k], 0]
                for k in range(1, v2 + 1):
                    t_k = (v1, k)
                    p -= alloc[type_index[t_k], 1]
                payments[t] = p
            
            # Verify IR and IC
            ir_ok = True
            for t in range(n_types):
                util = v[t] @ alloc[t] - payments[t]
                if util < -1e-8:
                    ir_ok = False
                    break
            if not ir_ok:
                # Fallback: clamp payments to ensure IR (should not happen if LP is correct)
                for t in range(n_types):
                    util = v[t] @ alloc[t] - payments[t]
                    if util < 0:
                        payments[t] = v[t] @ alloc[t]
                
            ic_ok = True
            for t in range(n_types):
                for tp in range(n_types):
                    lhs = v[t] @ alloc[t] - payments[t]
                    rhs = v[t] @ alloc[tp] - payments[tp]
                    if lhs - rhs < -1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            
            if ic_ok:
                return {
                    'allocations': alloc,
                    'payments': payments
                }
        else:
            # Fallback if LP fails
            pass
            
    except Exception:
        pass

    # Fallback: Simple threshold mechanism
    best_revenue = -1.0
    best_alloc = None
    best_pay = None

    for thresh1 in range(5):
        for thresh2 in range(5):
            alloc = np.zeros((n_types, n_goods))
            for t_idx, (v1, v2) in enumerate(types):
                alloc[t_idx, 0] = 1.0 if v1 >= thresh1 else 0.0
                alloc[t_idx, 1] = 1.0 if v2 >= thresh2 else 0.0

            payments = np.zeros(n_types)
            for t_idx, (v1, v2) in enumerate(types):
                a1_val = alloc[t_idx, 0]
                a2_val = alloc[t_idx, 1]
                sum_a1 = 0.0
                for i in range(1, v1 + 1):
                    t_i = (i, v2)
                    if t_i in type_index:
                        sum_a1 += alloc[type_index[t_i], 0]
                sum_a2 = 0.0
                for j in range(1, v2 + 1):
                    t_j = (v1, j)
                    if t_j in type_index:
                        sum_a2 += alloc[type_index[t_j], 1]
                payments[t_idx] = v1 * a1_val + v2 * a2_val - sum_a1 - sum_a2

            ir_ok = True
            for t_idx in range(n_types):
                util = v[t_idx] @ alloc[t_idx] - payments[t_idx]
                if util < -1e-8:
                    ir_ok = False
                    break
            if not ir_ok:
                continue

            ic_ok = True
            for t_idx in range(n_types):
                for tp_idx in range(n_types):
                    lhs = v[t_idx] @ alloc[t_idx] - payments[t_idx]
                    rhs = v[t_idx] @ alloc[tp_idx] - payments[tp_idx]
                    if lhs - rhs < -1e-8:
                        ic_ok = False
                        break
                if not ic_ok:
                    break
            if not ic_ok:
                continue

            exp_rev = np.sum(mu * payments)
            if exp_rev > best_revenue:
                best_revenue = exp_rev
                best_alloc = alloc.copy()
                best_pay = payments.copy()

    if best_alloc is None:
        best_alloc = np.zeros((n_types, n_goods))
        best_pay = np.zeros(n_types)

    return {
        'allocations': best_alloc,
        'payments': best_pay
    }
# EVOLVE-BLOCK-END
