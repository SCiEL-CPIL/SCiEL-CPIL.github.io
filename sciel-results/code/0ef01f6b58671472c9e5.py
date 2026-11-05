import time
import numpy as np
from scipy.optimize import linprog


def construct_auction(distribution_spec, support, constraints):
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s kill

    # Parse inputs
    values = [0, 1, 2, 3]

    if isinstance(distribution_spec, dict):
        p1 = distribution_spec.get('good1', distribution_spec.get('p1', [0.25, 0.25, 0.25, 0.25]))
        p2 = distribution_spec.get('good2', distribution_spec.get('p2', [0.25, 0.25, 0.25, 0.25]))
    elif isinstance(distribution_spec, (list, tuple)):
        if len(distribution_spec) == 4:
            p1 = list(distribution_spec)
            p2 = list(distribution_spec)
        elif len(distribution_spec) == 8:
            p1 = list(distribution_spec[:4])
            p2 = list(distribution_spec[4:])
        else:
            p1 = [0.25, 0.25, 0.25, 0.25]
            p2 = [0.25, 0.25, 0.25, 0.25]
    else:
        p1 = [0.25, 0.25, 0.25, 0.25]
        p2 = [0.25, 0.25, 0.25, 0.25]

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # Normalize probabilities
    sum_p1 = p1.sum()
    sum_p2 = p2.sum()
    if sum_p1 > 0:
        p1 = p1 / sum_p1
    else:
        p1 = np.ones(4) / 4
    if sum_p2 > 0:
        p2 = p2 / sum_p2
    else:
        p2 = np.ones(4) / 4

    joint_probs = np.outer(p1, p2).flatten()
    n_types = 16
    types = [(v1, v2) for v1 in values for v2 in values]
    V = np.array(types, dtype=float)  # shape (16, 2)

    # ============================================================
    # STAGE 1: Solve the allocation LP
    # Variables: x_t in [0,1] for t=0..15, good=0,1 -> 32 vars
    # Objective: maximize sum_t prob_t * (V_t @ x_t)  [expected value]
    #   (Revenue-equivalence: max revenue = max expected value - constant)
    # Constraints:
    #   IC: For all t, tp: V_t@(x_t - x_tp) >= 0  (allocation IC)
    #     => -V_t@x_t + V_t@x_tp <= 0
    #   IR: V_t @ x_t >= 0  (always true since V_t >= 0, x_t >= 0)
    #   Feasibility: 0 <= x_t <= 1
    # ============================================================

    n_x = 32  # 16 types * 2 goods
    c = np.zeros(n_x)
    # Maximize sum_t prob_t * V_t @ x_t  =>  minimize -sum_t prob_t * V_t @ x_t
    for t in range(16):
        c[2*t] = -joint_probs[t] * V[t, 0]
        c[2*t+1] = -joint_probs[t] * V[t, 1]

    # IC constraints: -V_t @ x_t + V_t @ x_tp <= 0  for all t != tp
    n_ic = 16 * 15
    A_ub = np.zeros((n_ic, n_x))
    b_ub = np.zeros(n_ic)

    idx = 0
    for t in range(16):
        for tp in range(16):
            if t == tp:
                continue
            A_ub[idx, 2*t] = -V[t, 0]
            A_ub[idx, 2*t+1] = -V[t, 1]
            A_ub[idx, 2*tp] = V[t, 0]
            A_ub[idx, 2*tp+1] = V[t, 1]
            b_ub[idx] = 0
            idx += 1

    bounds = [(0, 1)] * n_x

    best_allocations = np.zeros((16, 2))
    best_payments = np.zeros(16)

    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

        if res.success:
            x = res.x.reshape(16, 2)

            # Clip to [0,1] to handle numerical noise
            x = np.clip(x, 0, 1)

            # ============================================================
            # STAGE 2: Derive payments via revenue-equivalence / payment characterization
            # For a direct-revelation mechanism with IC, the payment is:
            #   p_t = V_t @ x_t - u_t
            # where u_t is the utility. Under IR, u_t >= 0.
            # 
            # The standard payment characterization (Myerson):
            #   p_t = V_t @ x_t - sum_{tp} prob_{tp} * max(0, V_t @ (x_tp - x_t))
            # 
            # But for finite support with IC constraints as written, the tightest feasible
            # payment that satisfies IC and IR is:
            #   p_t = V_t @ x_t - min_{tp} [V_t @ x_tp - p_tp]  ... circular
            #
            # Better approach: Use the LP dual / envelope theorem.
            # The payment that is IC-consistent and IR-binding is:
            #   p_t = V_t @ x_t - u_t^*
            # where u_t^* = min_{tp} [V_t @ x_tp - p_tp]  (the worst-case utility)
            #
            # Since we want to MAXIMIZE revenue, we want to MINIMIZE u_t (utility).
            # The minimum utility consistent with IC is:
            #   u_t = min_{tp} [V_t @ x_tp - p_tp]
            # But this is circular. Instead, use the fact that for the optimal IC mechanism,
            # the payment is determined by:
            #   p_t = V_t @ x_t - u_t
            #   u_t = min_{tp} [V_t @ x_tp - p_tp]
            #
            # A clean way: solve for payments given x using a small LP or closed form.
            # Since IC is: V_t @ x_t - p_t >= V_t @ x_tp - p_tp  for all tp
            #   => p_t - p_tp <= V_t @ (x_t - x_tp)
            # And IR: p_t <= V_t @ x_t, p_t >= 0
            #
            # To maximize revenue (sum prob_t * p_t), we want p_t as large as possible.
            # The tightest upper bound on p_t from IC and IR:
            #   p_t <= V_t @ x_t  (IR)
            #   p_t <= p_tp + V_t @ (x_t - x_tp)  for all tp  (IC rearranged)
            #
            # So: p_t <= min( V_t @ x_t, min_{tp} [p_tp + V_t @ (x_t - x_tp)] )
            #
            # This is a system of inequalities. We can solve it by:
            # 1. Start with p_t = V_t @ x_t (IR binding)
            # 2. Iterate: p_t = min(p_t, min_{tp} [p_tp + V_t @ (x_t - x_tp)])
            #    until convergence (this is a shortest-path / Bellman-Ford style)
            #
            # Actually, the cleanest: the maximum feasible payment vector is the solution to:
            #   p_t = min_{tp} [V_t @ x_tp - u_tp + V_t @ (x_t - x_tp)]  ... still circular
            #
            # Let's use the standard result: for a single additive bidder, the IC + IR
            # mechanism that maximizes revenue given allocations x has payments:
            #   p_t = V_t @ x_t - u_t
            #   u_t = min_{tp} [V_t @ x_tp - p_tp]
            #
            # This can be solved by value iteration. Initialize u_t = 0 for all t.
            # Then iterate: u_t = min_{tp} [V_t @ x_tp - (V_tp @ x_tp - u_tp)]
            #              = min_{tp} [V_t @ x_tp - V_tp @ x_tp + u_tp]
            #
            # This converges in at most 16 iterations (finite state space).
            # ============================================================

            # Value iteration for utilities
            u = np.zeros(16)
            for iteration in range(50):
                u_new = np.zeros(16)
                for t in range(16):
                    # u_t = min_{tp} [V_t @ x_tp - p_tp] = min_{tp} [V_t @ x_tp - (V_tp @ x_tp - u_tp)]
                    min_val = np.inf
                    for tp in range(16):
                        val = V[t] @ x[tp] - (V[tp] @ x[tp] - u[tp])
                        if val < min_val:
                            min_val = val
                    u_new[t] = min_val
                if np.max(np.abs(u_new - u)) < 1e-12:
                    u = u_new
                    break
                u = u_new

            # Ensure u_t >= 0 (IR)
            u = np.maximum(u, 0.0)

            # Compute payments
            p = V @ x.T - u  # p_t = V_t @ x_t - u_t
            p = np.maximum(p, 0.0)  # non-negative payments

            # ============================================================
            # STAGE 3: Feasibility verification and projection (deterministic repair)
            # Check IC and IR. If violated by > 1e-8, project payments to feasible region.
            # ============================================================

            tol = 1e-8
            valid = True

            # Check IR
            for t in range(16):
                if p[t] > V[t] @ x[t] + tol:
                    valid = False
                    break

            # Check IC
            if valid:
                for t in range(16):
                    u_tt = V[t] @ x[t] - p[t]
                    for tp in range(16):
                        if t == tp:
                            continue
                        u_ttp = V[t] @ x[tp] - p[tp]
                        if u_tt < u_ttp - tol:
                            valid = False
                            break
                    if not valid:
                        break

            if not valid:
                # Deterministic feasibility projection:
                # For each t, the maximum feasible payment is:
                #   p_t <= min( V_t @ x_t, min_{tp} [p_tp + V_t @ (x_t - x_tp)] )
                # We solve this by iterating from the "worst" types.
                # Use a topological / iterative approach:
                # Initialize p_t = 0 for all t.
                # Then for each t, set p_t = min( V_t @ x_t, min_{tp} [p_tp + V_t @ (x_t - x_tp)] )
                # Iterate until convergence.

                p_proj = np.zeros(16)
                for iteration in range(100):
                    p_new = np.zeros(16)
                    for t in range(16):
                        # IR bound
                        bound = V[t] @ x[t]
                        # IC bounds
                        for tp in range(16):
                            if t == tp:
                                continue
                            ic_bound = p_proj[tp] + V[t] @ (x[t] - x[tp])
                            if ic_bound < bound:
                                bound = ic_bound
                        p_new[t] = max(0.0, bound)
                    if np.max(np.abs(p_new - p_proj)) < 1e-12:
                        p_proj = p_new
                        break
                    p_proj = p_new

                p = p_proj

                # Re-verify
                valid = True
                for t in range(16):
                    if p[t] > V[t] @ x[t] + tol:
                        valid = False
                        break
                if valid:
                    for t in range(16):
                        u_tt = V[t] @ x[t] - p[t]
                        for tp in range(16):
                            if t == tp:
                                continue
                            u_ttp = V[t] @ x[tp] - p[tp]
                            if u_tt < u_ttp - tol:
                                valid = False
                                break
                        if not valid:
                            break

                if not valid:
                    # Last resort: zero mechanism
                    x = np.zeros((16, 2))
                    p = np.zeros(16)

            # Final safety check
            final_valid = True
            for t in range(16):
                if p[t] < -tol or p[t] > V[t] @ x[t] + tol:
                    final_valid = False
                    break
            if final_valid:
                for t in range(16):
                    u_tt = V[t] @ x[t] - p[t]
                    for tp in range(16):
                        if t == tp:
                            continue
                        u_ttp = V[t] @ x[tp] - p[tp]
                        if u_tt < u_ttp - tol:
                            final_valid = False
                            break
                    if not final_valid:
                        break

            if final_valid:
                best_allocations = x
                best_payments = p
            else:
                best_allocations = np.zeros((16, 2))
                best_payments = np.zeros(16)

        else:
            # LP failed, fallback to zero
            best_allocations = np.zeros((16, 2))
            best_payments = np.zeros(16)

    except Exception:
        best_allocations = np.zeros((16, 2))
        best_payments = np.zeros(16)

    # Deadline guard
    if time.time() > deadline:
        # Return whatever we have (should be fine since we're near the end)
        pass

    return {'allocations': best_allocations, 'payments': best_payments}
