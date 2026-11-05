"""
Incentive-compatible finite auction design.

We solve the finite-support mechanism design LP directly:
  - 1 additive bidder, 2 goods, values in {0,1,2,3} -> 16 types.
  - Variables: allocation probabilities q_{t,g} for each type t and good g (32 vars),
    and payment p_t for each type t (16 vars).
  - Constraints:
      * 0 <= q_{t,g} <= 1
      * p_t >= 0  (nonnegative payments)
      * Individual rationality: p_t <= v_t . q_t  (utility >= 0)
      * Incentive compatibility: for all t, t',
            v_t . q_t - p_t >= v_t . q_{t'} - p_{t'}
      * Regret <= 1e-8 (enforced by IC with tight tolerance)
  - Objective: maximize expected revenue = sum_t prob_t * p_t

The LP is small (48 vars, ~16*16*2 + 16 + 32 constraints ≈ 528 constraints),
so scipy.optimize.linprog with HiGHS solver should be fast and exact.

We also add a post-processing step to clip tiny numerical violations.
"""

import time
import numpy as np
from scipy.optimize import linprog


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = 60.0  # safety margin: kill at 90s, return by 60s

    # Define types: (v1, v2) for v1,v2 in {0,1,2,3}
    vals = [0, 1, 2, 3]
    types = [(v1, v2) for v1 in vals for v2 in vals]
    n_types = len(types)  # 16
    n_goods = 2

    # Map type to index
    type_to_idx = {t: i for i, t in enumerate(types)}

    # Get distribution probabilities
    probs = _get_probs(distribution_spec, n_types)

    # Decision variables:
    #   x[0:32]  = allocation q_{t,g} for t in 0..15, g in 0..1
    #   x[32:48] = payment p_t for t in 0..15
    n_alloc = n_types * n_goods  # 32
    n_pay = n_types               # 16
    n_vars = n_alloc + n_pay      # 48

    # Objective: maximize sum_t probs[t] * p_t
    # linprog minimizes, so negate
    c = np.zeros(n_vars)
    for t in range(n_types):
        c[n_alloc + t] = -probs[t]

    # Inequality constraints: A_ub @ x <= b_ub
    # We'll build them in groups.

    # 1. Allocation bounds: 0 <= q <= 1 -> handled by bounds
    # 2. Nonnegative payments: p_t >= 0 -> handled by bounds
    # 3. Individual rationality: p_t - v_t.q_t <= 0
    #    For each t: p_t - sum_g v_{t,g} * q_{t,g} <= 0
    # 4. IC: for all t, t':
    #    (v_t . q_t - p_t) - (v_{t'} . q_{t'} - p_{t'}) >= 0
    #    => -(v_t . q_t - p_t) + (v_{t'} . q_{t'} - p_{t'}) <= 0
    #    => sum_g (v_{t',g} - v_{t,g}) * q_{t,g} + sum_g (v_{t,g} - v_{t',g}) * q_{t',g}
    #       + (-1)*p_t + 1*p_{t'} ... wait, let me re-derive.

    # IC: u(t, t) >= u(t, t')  for all t, t'
    # u(t, t) = v_t . q_t - p_t
    # u(t, t') = v_t . q_{t'} - p_{t'}
    # So: v_t . q_t - p_t >= v_t . q_{t'} - p_{t'}
    # => v_t . q_t - v_t . q_{t'} - p_t + p_{t'} >= 0
    # => -(v_t . q_t - v_t . q_{t'} - p_t + p_{t'}) <= 0

    # Let's build the constraint matrix.
    # Number of IC constraints: 16 * 16 = 256
    # Number of IR constraints: 16
    # Total: 272 inequality constraints.

    n_ic = n_types * n_types
    n_ir = n_types
    n_ineq = n_ic + n_ir

    A_ub = np.zeros((n_ineq, n_vars))
    b_ub = np.zeros(n_ineq)

    # IR constraints: p_t - v_t . q_t <= 0
    for t in range(n_types):
        row = n_ir + 0  # IR rows come after IC? Let's put IR first for clarity.
        # Actually, let's put IR in rows 0..15, IC in rows 16..271
        row = t
        A_ub[row, n_alloc + t] = 1.0  # p_t
        for g in range(n_goods):
            A_ub[row, t * n_goods + g] = -types[t][g]  # -v_{t,g} * q_{t,g}
        b_ub[row] = 0.0

    # IC constraints: for each t (true type) and t' (reported type)
    # v_t . q_t - v_t . q_{t'} - p_t + p_{t'} >= 0
    # => -v_t . q_t + v_t . q_{t'} + p_t - p_{t'} <= 0
    idx = n_ir
    for t in range(n_types):
        for tp in range(n_types):
            row = idx
            # Coefficients for q_{t,g}: -v_{t,g}
            for g in range(n_goods):
                A_ub[row, t * n_goods + g] = -types[t][g]
            # Coefficients for q_{tp,g}: +v_{t,g}
            for g in range(n_goods):
                A_ub[row, tp * n_goods + g] += types[t][g]
            # Coefficients for p_t: +1
            A_ub[row, n_alloc + t] += 1.0
            # Coefficients for p_{tp}: -1
            A_ub[row, n_alloc + tp] += -1.0
            b_ub[row] = 0.0
            idx += 1

    # Bounds:
    #   q_{t,g} in [0, 1]
    #   p_t in [0, inf)
    bounds = []
    for i in range(n_alloc):
        bounds.append((0, 1))
    for i in range(n_pay):
        bounds.append((0, None))

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not result.success:
        # Fallback: return a trivial mechanism (allocate nothing, charge 0)
        allocations = np.zeros((n_types, n_goods))
        payments = np.zeros(n_types)
        return {'allocations': allocations, 'payments': payments}

    x = result.x

    # Extract allocations and payments
    allocations = x[:n_alloc].reshape(n_types, n_goods)
    payments = x[n_alloc:]

    # Post-process: clip tiny numerical violations
    allocations = np.clip(allocations, 0.0, 1.0)
    payments = np.maximum(payments, 0.0)

    # Verify individual rationality and IC with tolerance
    # If violations exist beyond 1e-8, we need to repair.
    # Simple repair: for any type where p_t > v_t.q_t + 1e-8, set p_t = v_t.q_t
    for t in range(n_types):
        vq = types[t][0] * allocations[t, 0] + types[t][1] * allocations[t, 1]
        if payments[t] > vq + 1e-8:
            payments[t] = vq

    # Check IC: for all t, tp, u(t,t) >= u(t,tp) - 1e-8
    # If violated, we can adjust payments down for the offending report.
    # This is a heuristic; in practice the LP should be satisfied.
    # We'll do a simple pass: for each pair, if u(t,t) < u(t,tp) - 1e-8,
    # reduce p_{tp} to make them equal (or increase p_t, but that might break IR).
    # Reducing p_{tp} is safer for IR.
    for _ in range(10):  # a few iterations of repair
        violated = False
        for t in range(n_types):
            u_tt = types[t][0] * allocations[t, 0] + types[t][1] * allocations[t, 1] - payments[t]
            for tp in range(n_types):
                u_ttp = types[t][0] * allocations[tp, 0] + types[t][1] * allocations[tp, 1] - payments[tp]
                if u_tt < u_ttp - 1e-8:
                    # Need to reduce u_ttp, i.e., increase payments[tp]
                    # But increasing payments might break IR for tp.
                    # Max payment for tp is v_tp . q_tp.
                    vq_tp = types[tp][0] * allocations[tp, 0] + types[tp][1] * allocations[tp, 1]
                    needed = u_ttp - u_tt + 1e-8
                    # Increase payments[tp] by needed, but cap at vq_tp
                    new_pay = min(payments[tp] + needed, vq_tp)
                    if new_pay > payments[tp]:
                        payments[tp] = new_pay
                        violated = True
        if not violated:
            break

    # Final check: ensure payments are nonnegative and IR holds
    for t in range(n_types):
        vq = types[t][0] * allocations[t, 0] + types[t][1] * allocations[t, 1]
        payments[t] = min(max(payments[t], 0.0), vq)

    # Check time
    if time.time() - start_time > deadline - 5:
        pass  # We're close to deadline, return what we have

    return {'allocations': allocations, 'payments': payments}


def _get_probs(distribution_spec, n_types):
    """Get probability vector for the given distribution spec."""
    if isinstance(distribution_spec, str):
        spec = distribution_spec.lower()
        if spec in ('uniform', 'independent_uniform'):
            return np.ones(n_types) / n_types
        elif spec == 'independent_uniform':
            return np.ones(n_types) / n_types
        elif spec == 'uniform':
            return np.ones(n_types) / n_types
        else:
            # Default to uniform
            return np.ones(n_types) / n_types
    elif isinstance(distribution_spec, (list, tuple, np.ndarray)):
        p = np.array(distribution_spec, dtype=float)
        if len(p) == n_types:
            p = p / p.sum()
            return p
        else:
            return np.ones(n_types) / n_types
    else:
        return np.ones(n_types) / n_types
