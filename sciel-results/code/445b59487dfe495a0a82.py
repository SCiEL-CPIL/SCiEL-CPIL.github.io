def _compute_optimal_policy(params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps

    n_q = 2 * Q + 1
    q_range = np.arange(-Q, Q + 1)

    # Initial coarse grid: reduced size to save time
    depth_min = 0.001
    depth_max = 10.0 / k
    n_depths_init = 50
    depth_grid = np.geomspace(depth_min, depth_max, n_depths_init)

    exp_term = np.exp(-k * depth_grid)
    p_fill = 1.0 - np.exp(-A * exp_term * dt)
    depth_p = depth_grid * p_fill

    V = np.zeros((steps + 1, n_q))
    bid_table = np.zeros((steps, n_q))
    ask_table = np.zeros((steps, n_q))

    # First pass: full backward DP with coarse grid
    for n in range(steps - 1, -1, -1):
        if time.time() - _START_TIME > TIME_LIMIT - 30:
            # Fallback
            for n2 in range(n, -1, -1):
                for qi in range(n_q):
                    q = q_range[qi]
                    d = 1.0 / k + gamma * sigma**2 * q * (T - n2 * dt) / (A * k)
                    d = max(d, depth_min)
                    bid_table[n2, qi] = d
                    ask_table[n2, qi] = d
            bid_table[:, n_q - 1] = np.inf
            ask_table[:, 0] = np.inf
            return bid_table, ask_table

        inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
        Vn1 = V[n + 1]

        V_plus = np.empty(n_q)
        V_minus = np.empty(n_q)
        V_plus[:-1] = Vn1[1:]
        V_plus[-1] = Vn1[-1]
        V_minus[1:] = Vn1[:-1]
        V_minus[0] = Vn1[0]

        D1 = V_plus - Vn1
        Dm1 = V_minus - Vn1
        D2 = V_plus + V_minus - 2.0 * Vn1

        for qi in range(n_q):
            q_val = q_range[qi]
            cost = inv_cost_factor * q_val * q_val

            if qi == n_q - 1:  # q = Q, no bid fill
                p_bid_eff = np.zeros(n_depths_init)
            else:
                p_bid_eff = p_fill

            if qi == 0:  # q = -Q, no ask fill
                p_ask_eff = np.zeros(n_depths_init)
            else:
                p_ask_eff = p_fill

            Vn1_q = Vn1[qi]
            Vn1_q_plus = Vn1[qi + 1] if qi < n_q - 1 else Vn1_q
            Vn1_q_minus = Vn1[qi - 1] if qi > 0 else Vn1_q

            if qi == 0:  # q = -Q
                term1 = np.full(n_depths_init, Vn1_q_plus)
                term2 = np.full(n_depths_init, Vn1_q)
                ask_depth_p = np.zeros(n_depths_init)
            elif qi == n_q - 1:  # q = Q
                term1 = p_ask_eff * Vn1_q + (1 - p_ask_eff) * Vn1_q_plus
                term2 = p_ask_eff * Vn1_q_minus + (1 - p_ask_eff) * Vn1_q
                ask_depth_p = depth_p
            else:
                term1 = p_ask_eff * Vn1_q + (1 - p_ask_eff) * Vn1_q_plus
                term2 = p_ask_eff * Vn1_q_minus + (1 - p_ask_eff) * Vn1_q
                ask_depth_p = depth_p

            diff = term1 - term2
            ask_term = term2 + ask_depth_p

            if qi == n_q - 1:
                best_j = np.argmax(ask_term)
                best_val = ask_term[best_j] - cost
                best_bid = depth_grid[0]
                best_ask = depth_grid[best_j]
            else:
                M = np.outer(p_bid_eff, diff) + ask_term[None, :]
                M += (p_bid_eff * depth_grid)[:, None]
                M -= cost

                flat_idx = np.argmax(M)
                i_max = flat_idx // n_depths_init
                j_max = flat_idx % n_depths_init

                best_val = M[i_max, j_max]
                best_bid = depth_grid[i_max]
                best_ask = depth_grid[j_max]

            V[n, qi] = best_val
            bid_table[n, qi] = best_bid
            ask_table[n, qi] = best_ask

    # Refinement passes
    n_refine = 7
    for refine_pass in range(n_refine):
        if time.time() - _START_TIME > TIME_LIMIT - 20:
            break

        # Increase resolution in later passes
        if refine_pass < 3:
            n_local = 100
        else:
            n_local = 200

        new_V = np.zeros((steps + 1, n_q))
        new_bid_table = np.zeros((steps, n_q))
        new_ask_table = np.zeros((steps, n_q))

        for n in range(steps - 1, -1, -1):
            if time.time() - _START_TIME > TIME_LIMIT - 15:
                new_V[n+1:] = V[n+1:]
                new_bid_table[n:] = bid_table[n:]
                new_ask_table[n:] = ask_table[n:]
                break

            inv_cost_factor = (gamma / 2.0) * sigma**2 * dt
            Vn1 = new_V[n + 1]

            V_plus = np.empty(n_q)
            V_minus = np.empty(n_q)
            V_plus[:-1] = Vn1[1:]
            V_plus[-1] = Vn1[-1]
            V_minus[1:] = Vn1[:-1]
            V_minus[0] = Vn1[0]

            D1 = V_plus - Vn1
            Dm1 = V_minus - Vn1
            D2 = V_plus + V_minus - 2.0 * Vn1

            for qi in range(n_q):
                q_val = q_range[qi]
                cost = inv_cost_factor * q_val * q_val

                prev_bid = bid_table[n, qi]
                prev_ask = ask_table[n, qi]

                if qi == n_q - 1:  # q = Q, no bid
                    local_bid_depths = np.array([depth_min])
                    local_ask_depths = np.geomspace(
                        max(prev_ask * 0.9, depth_min),
                        min(prev_ask * 1.1, depth_max),
                        n_local
                    )
                else:
                    local_bid_depths = np.geomspace(
                        max(prev_bid * 0.9, depth_min),
                        min(prev_bid * 1.1, depth_max),
                        n_local
                    )
                    if qi == 0:  # q = -Q, no ask
                        local_ask_depths = np.array([depth_min])
                    else:
                        local_ask_depths = np.geomspace(
                            max(prev_ask * 0.9, depth_min),
                            min(prev_ask * 1.1, depth_max),
                            n_local
                        )

                exp_b = np.exp(-k * local_bid_depths)
                p_b = 1.0 - np.exp(-A * exp_b * dt)
                dp_b = local_bid_depths * p_b

                exp_a = np.exp(-k * local_ask_depths)
                p_a = 1.0 - np.exp(-A * exp_a * dt)
                dp_a = local_ask_depths * p_a

                nb = len(local_bid_depths)
                na = len(local_ask_depths)

                term1 = dp_b[:, None] + dp_a[None, :]
                term2 = Vn1[qi] + p_b[:, None] * D1[qi] + p_a[None, :] * Dm1[qi]
                term3 = - (p_b[:, None] * p_a[None, :]) * D2[qi]

                J = term1 + term2 + term3 - cost

                idx_max = np.unravel_index(np.argmax(J), J.shape)
                i_max, j_max = idx_max

                new_V[n, qi] = J[i_max, j_max]
                new_bid_table[n, qi] = local_bid_depths[i_max]
                new_ask_table[n, qi] = local_ask_depths[j_max]

        V = new_V
        bid_table = new_bid_table
        ask_table = new_ask_table

    # Post-process boundaries
    bid_table[:, n_q - 1] = np.inf
    ask_table[:, 0] = np.inf

    return bid_table, ask_table
