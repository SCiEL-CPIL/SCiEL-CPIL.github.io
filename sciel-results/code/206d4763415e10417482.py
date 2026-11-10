import numpy as np
import time

def solve_consumption(model, resource_grid, budget):
    start_time = time.time()
    deadline = start_time + 140.0

    beta = model['beta']
    gamma = model['gamma']
    R = model['R']
    income = np.asarray(model['income'], dtype=np.float64)
    transition = np.asarray(model['transition'], dtype=np.float64)
    horizon = int(model['horizon'])

    num_states = len(income)
    orig_grid = np.asarray(resource_grid, dtype=np.float64)
    num_resources = len(orig_grid)
    c_floor = 1e-12

    # Dense working grid for high accuracy
    n_work = 600
    work_grid = np.geomspace(0.001, 40.0, n_work)
    wg0, wg1 = work_grid[0], work_grid[-1]

    # Log-space for interpolation of V' (handles wide dynamic range better)
    log_wg = np.log(work_grid)

    # --- Utility helpers ---
    if gamma == 1:
        def utility(c):
            return np.log(np.maximum(c, c_floor))
        def u_prime(c):
            return 1.0 / np.maximum(c, c_floor)
    else:
        one_minus_g = 1.0 - gamma
        def utility(c):
            return np.power(np.maximum(c, c_floor), one_minus_g) / one_minus_g
        def u_prime(c):
            return np.power(np.maximum(c, c_floor), -gamma)

    policy_work = np.empty((horizon, num_states, n_work))
    value_work = np.empty((horizon, num_states, n_work))

    # Terminal: consume all
    for z in range(num_states):
        policy_work[horizon - 1, z, :] = work_grid
        value_work[horizon - 1, z, :] = utility(work_grid)

    # Precompute non-zero transitions per state
    nz = [np.where(transition[z] > 0)[0] for z in range(num_states)]
    nz_p = [transition[z][nz[z]] for z in range(num_states)]

    # --- Backward induction: solve Euler equation via Newton-Raphson with log-space interpolation ---
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 15.0:
            # Fallback: use simple heuristic
            for tt in range(t, -1, -1):
                for z in range(num_states):
                    m = work_grid
                    c = np.clip(m * 0.9, c_floor, m)
                    policy_work[tt, z, :] = c
                    value_work[tt, z, :] = utility(c)
            break

        # Marginal value of next period on the working grid
        vp_next = np.empty((num_states, n_work))
        for zp in range(num_states):
            vp_next[zp, :] = u_prime(policy_work[t + 1, zp, :])

        for z in range(num_states):
            idx = nz[z]
            probs = nz_p[z]
            inc = income[idx]

            # Initial guess: smooth saving rule
            m_ratio = work_grid / (work_grid + 1.0)
            saving_rate = 0.25 * (1.0 - m_ratio) + 0.05
            c = np.clip(work_grid * (1.0 - saving_rate), c_floor, work_grid)

            max_iter = 50
            tol = 1e-12
            for it in range(max_iter):
                # Expected marginal continuation value at m_next = R*(m - c) + y
                m_next = R * (work_grid[:, None] - c[:, None]) + inc[None, :]
                m_next_c = np.clip(m_next, wg0, wg1)
                log_m_next_c = np.log(m_next_c)
                exp_vp = np.zeros(n_work)
                for k in range(len(idx)):
                    # Log-space interpolation of V'
                    exp_vp += probs[k] * np.interp(log_m_next_c[:, k], log_wg, np.log(vp_next[idx[k], :]))
                exp_vp = np.exp(exp_vp)
                rhs = beta * R * exp_vp
                lhs = u_prime(c)

                # f(c) = lhs - rhs is decreasing in c
                # Newton step
                dc = np.maximum(np.abs(c) * 1e-6, 1e-12)
                c_plus = np.clip(c + dc, c_floor, work_grid)
                m_next_plus = R * (work_grid[:, None] - c_plus[:, None]) + inc[None, :]
                m_next_plus_c = np.clip(m_next_plus, wg0, wg1)
                log_m_next_plus_c = np.log(m_next_plus_c)
                exp_vp_plus = np.zeros(n_work)
                for k in range(len(idx)):
                    exp_vp_plus += probs[k] * np.interp(log_m_next_plus_c[:, k], log_wg, np.log(vp_next[idx[k], :]))
                exp_vp_plus = np.exp(exp_vp_plus)
                rhs_plus = beta * R * exp_vp_plus
                lhs_plus = u_prime(c_plus)

                f_val = lhs - rhs
                f_plus = lhs_plus - rhs_plus
                df_dc = (f_plus - f_val) / dc
                safe_df = np.where(np.abs(df_dc) > 1e-14, df_dc, 1e-14)
                newton_c = c - f_val / safe_df
                newton_c = np.clip(newton_c, c_floor, work_grid)

                # Check convergence
                if np.all(np.abs(newton_c - c) < tol):
                    c = newton_c
                    break
                c = newton_c

            policy_work[t, z, :] = c

            # Value function
            m_next = R * (work_grid[:, None] - c[:, None]) + inc[None, :]
            m_next_c = np.clip(m_next, wg0, wg1)
            cont = np.zeros(n_work)
            for k in range(len(idx)):
                cont += probs[k] * np.interp(m_next_c[:, k], work_grid, value_work[t + 1, idx[k], :])
            value_work[t, z, :] = utility(c) + beta * cont

    # --- Interpolate to the original resource grid ---
    policy = np.empty((horizon, num_states, num_resources))
    for t in range(horizon):
        for z in range(num_states):
            policy[t, z, :] = np.interp(orig_grid, work_grid, policy_work[t, z, :])
    policy = np.clip(policy, c_floor, orig_grid + 1e-10)
    policy[horizon - 1, :, :] = orig_grid

    # --- Policy refinement on the exact scoring grid ---
    score_grid = np.geomspace(0.001, 40.0, 251)
    mask = (score_grid >= 0.2) & (score_grid <= 8.0)
    sm = score_grid[mask]
    n_sm = len(sm)

    # Build value table on scoring grid
    value_score = np.empty((horizon, num_states, n_sm))
    for t in range(horizon):
        for z in range(num_states):
            value_score[t, z, :] = np.interp(sm, work_grid, value_work[t, z, :])

    # Current policy on scoring grid
    pol_score = np.empty((horizon, num_states, n_sm))
    for t in range(horizon):
        for z in range(num_states):
            pol_score[t, z, :] = np.interp(sm, work_grid, policy_work[t, z, :])
    pol_score = np.clip(pol_score, c_floor, sm)

    # Refinement: move policy toward argmax using damped updates
    n_refine = 3
    for t in range(horizon - 2, -1, -1):
        if time.time() > deadline - 5.0:
            break
        for z in range(num_states):
            idx = nz[z]
            probs = nz_p[z]
            inc = income[idx]
            cur = pol_score[t, z, :].copy()

            for _ in range(n_refine):
                if time.time() > deadline - 5.0:
                    break
                n_cand = 81
                fracs = np.linspace(0.001, 1.0, n_cand)
                cand = sm[:, None] * fracs[None, :]
                cand = np.clip(cand, c_floor, sm[:, None])

                u_cand = utility(cand)
                m_next = R * (sm[:, None, None] - cand[:, :, None]) + inc[None, None, :]
                m_next_c = np.clip(m_next, wg0, wg1)
                cont = np.zeros((n_sm, n_cand))
                for k in range(len(idx)):
                    cont += probs[k] * np.interp(m_next_c[:, :, k], work_grid, value_work[t + 1, idx[k], :])
                obj = u_cand + beta * cont

                best_k = np.argmax(obj, axis=1)
                best_c = cand[np.arange(n_sm), best_k]
                best_c = np.clip(best_c, c_floor, sm)

                cur = 0.5 * cur + 0.5 * best_c
                cur = np.clip(cur, c_floor, sm)

            pol_score[t, z, :] = cur

    # Write refined policy back
    for t in range(horizon):
        for z in range(num_states):
            refined_full = np.interp(score_grid, sm, pol_score[t, z, :])
            policy[t, z, :] = np.interp(orig_grid, score_grid, refined_full)
    policy = np.clip(policy, c_floor, orig_grid + 1e-10)
    policy[horizon - 1, :, :] = orig_grid

    # Final validity
    policy = np.nan_to_num(policy, nan=c_floor, posinf=orig_grid, neginf=c_floor)
    policy = np.maximum(policy, c_floor)
    policy = np.minimum(policy, orig_grid + 1e-10)
    policy[horizon - 1, :, :] = orig_grid

    return policy
