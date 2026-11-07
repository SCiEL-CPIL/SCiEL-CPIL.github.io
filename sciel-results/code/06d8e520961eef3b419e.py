import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320  # safety margin before 330s kill
_cache = {}

def _compute_dp(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid/ask depth tables via dynamic programming with vectorized bisection."""
    dt = T / steps
    n_inv = 2 * Q + 1

    # V[n, q_idx] where q_idx = q + Q
    V = np.zeros((steps + 1, n_inv), dtype=np.float64)
    opt_bid = np.full((steps, n_inv), np.inf, dtype=np.float64)
    opt_ask = np.full((steps, n_inv), np.inf, dtype=np.float64)

    q_vals = np.arange(-Q, Q + 1)
    risk = (gamma / 2.0) * sigma * sigma * q_vals * q_vals * dt

    # d range for bisection: [d_lo, d_hi]
    # At d_lo, fill prob ~ A*dt (small). At d_hi, fill prob ~ 1.
    d_lo = 1e-8
    d_hi = 20.0 / k  # generous upper bound

    # Number of bisection iterations for high precision
    n_bisect = 60  # gives ~1e-18 relative precision

    def solve_one_side_vectorized(delta_V_array, can_fill_mask):
        """
        For each inventory level, maximize p(d)*(d + delta_V[i]) over d in [d_lo, d_hi].
        Returns (max_val_array, d_opt_array).
        Uses bisection on the first-order condition:
        f'(d) = p'(d)*(d+delta) + p(d) = 0
        where p(d) = 1 - exp(-A*exp(-k*d)*dt)
        p'(d) = A*k*exp(-k*d)*dt*exp(-A*exp(-k*d)*dt)
        """
        n = len(delta_V_array)
        d_lo_arr = np.full(n, d_lo, dtype=np.float64)
        d_hi_arr = np.full(n, d_hi, dtype=np.float64)

        # f(d) = p'(d)*(d + delta) + p(d)
        # We want to find d where f(d) = 0.
        # f is decreasing in d (can verify), so bisection works.
        # At d_lo: p is small, p' is large -> f > 0 typically
        # At d_hi: p ~ 1, p' ~ 0 -> f ~ 1 > 0... 
        # Actually let's think more carefully.
        # 
        # g(d) = p(d)*(d + delta)
        # g'(d) = p'(d)*(d+delta) + p(d)
        # We want g'(d) = 0.
        # g'(d) starts positive at d=0 (if delta > -something) and becomes negative.
        # So g' is decreasing? Let's check g''.
        # This is getting complex. Let's just use bisection on g(d) directly
        # by finding where the marginal gain is zero, or just do a fine grid + local refinement.
        # 
        # Alternative: since g(d) is unimodal, we can use ternary search or golden section.
        # But bisection on g'(d)=0 is cleaner if g' is monotone.
        #
        # Let me just use a dense grid evaluation + parabolic interpolation for speed.
        
        # Actually, for 61 inventory levels, even calling minimize_scalar 61*2*1000 times
        # might be slow. Let's use a vectorized approach with a coarse grid + refinement.
        
        # Coarse grid
        n_grid = 200
        d_grid = np.linspace(d_lo, d_hi, n_grid)
        
        # p(d_grid) for all d
        p_grid = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
        
        # For each delta_V, compute g(d_grid) = p_grid * (d_grid + delta_V)
        # shape: (n_inv, n_grid)
        g_matrix = p_grid[None, :] * (d_grid[None, :] + delta_V_array[:, None])
        
        # Find argmax for each row
        idx_max = np.argmax(g_matrix, axis=1)
        
        # Refine with local bisection around the grid max
        d_opt = np.empty(n, dtype=np.float64)
        val_opt = np.empty(n, dtype=np.float64)
        
        for i in range(n):
            if not can_fill_mask[i]:
                d_opt[i] = np.inf
                val_opt[i] = 0.0
                continue
            
            # Local refinement: bisection on g'(d) = 0 in neighborhood of grid max
            center = d_grid[idx_max[i]]
            # Search in [max(d_lo, center - 2*grid_step), min(d_hi, center + 2*grid_step)]
            grid_step = (d_hi - d_lo) / (n_grid - 1)
            lo = max(d_lo, center - 3 * grid_step)
            hi = min(d_hi, center + 3 * grid_step)
            
            # Bisection on g'(d) = p'(d)*(d+delta) + p(d) = 0
            delta_i = delta_V_array[i]
            
            for _ in range(40):
                mid = (lo + hi) / 2.0
                # Compute p(mid) and p'(mid)
                exp_term = np.exp(-k * mid)
                p_mid = 1.0 - np.exp(-A * exp_term * dt)
                # p'(d) = A * k * exp(-k*d) * dt * exp(-A*exp(-k*d)*dt)
                p_prime_mid = A * k * exp_term * dt * np.exp(-A * exp_term * dt)
                g_prime = p_prime_mid * (mid + delta_i) + p_mid
                
                if g_prime > 0:
                    lo = mid
                else:
                    hi = mid
            
            d_opt[i] = (lo + hi) / 2.0
            # Compute optimal value
            exp_term = np.exp(-k * d_opt[i])
            p_opt = 1.0 - np.exp(-A * exp_term * dt)
            val_opt[i] = p_opt * (d_opt[i] + delta_i)
            if val_opt[i] < 0:
                val_opt[i] = 0.0
                d_opt[i] = np.inf
        
        return val_opt, d_opt

    # Backward induction
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            # Fallback: fill remaining with a simple heuristic
            for nn in range(n, -1, -1):
                for qi in range(n_inv):
                    q = qi - Q
                    base = 1.0 / k
                    skew = gamma * sigma * sigma * (T - nn * dt) * q * 0.05
                    b = max(1e-6, base * (1.0 + skew))
                    a = max(1e-6, base * (1.0 - skew))
                    if q >= Q:
                        b = np.inf
                    if q <= -Q:
                        a = np.inf
                    opt_bid[nn, qi] = b
                    opt_ask[nn, qi] = a
                    V[nn, qi] = 0.0
            break

        # Vectorized computation for all inventory levels at step n
        # Bid side: delta_b[i] = V[n+1, i+1] - V[n+1, i] for i < Q
        # Ask side: delta_a[i] = V[n+1, i-1] - V[n+1, i] for i > -Q
        
        Vn1 = V[n + 1]
        
        # Bid deltas
        delta_b = np.full(n_inv, 0.0, dtype=np.float64)
        delta_b[:-1] = Vn1[1:] - Vn1[:-1]
        can_fill_b = np.arange(n_inv) < (n_inv - 1)  # q < Q means qi < 2Q
        
        # Ask deltas  
        delta_a = np.full(n_inv, 0.0, dtype=np.float64)
        delta_a[1:] = Vn1[:-1] - Vn1[1:]
        can_fill_a = np.arange(n_inv) > 0  # q > -Q means qi > 0
        
        bid_val, bid_d = solve_one_side_vectorized(delta_b, can_fill_b)
        ask_val, ask_d = solve_one_side_vectorized(delta_a, can_fill_a)
        
        V[n, :] = Vn1 - risk + bid_val + ask_val
        opt_bid[n, :] = bid_d
        opt_ask[n, :] = ask_d

    return opt_bid, opt_ask


def _get_dp_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'],
           params['T'], params['steps'], params['Q'])
    if key not in _cache:
        sigma, A, k, gamma, T, steps, Q = key
        opt_bid, opt_ask = _compute_dp(sigma, A, k, gamma, T, steps, Q)
        _cache[key] = (opt_bid, opt_ask)
    return _cache[key]


def quote(t, inventory, mid, params):
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']

    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))

    opt_bid, opt_ask = _get_dp_tables(params)

    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)

    bid_depths = opt_bid[n, q_idx].astype(np.float64)
    ask_depths = opt_ask[n, q_idx].astype(np.float64)

    # Safety: ensure valid values
    min_valid = -3.0 / k + 1e-6
    bid_depths = np.where(np.isfinite(bid_depths) & (bid_depths >= min_valid), bid_depths, np.inf)
    ask_depths = np.where(np.isfinite(ask_depths) & (ask_depths >= min_valid), ask_depths, np.inf)

    # Boundary: no bid at Q, no ask at -Q
    bid_depths = np.where(inventory >= Q, np.inf, bid_depths)
    ask_depths = np.where(inventory <= -Q, np.inf, ask_depths)

    return bid_depths, ask_depths
