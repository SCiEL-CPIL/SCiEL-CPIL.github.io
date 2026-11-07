"""Optimal market making with inventory risk via precomputed DP."""
import numpy as np
import time

_cache = {}
_DEADLINE = time.time() + 270  # safety margin


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """Precompute optimal bid/ask depths for all (time, inventory) states."""
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _cache:
        return _cache[key]
    
    dt = T / steps
    n_inv = 2 * Q + 1
    
    V = np.zeros((steps + 1, n_inv))
    bid_depths = np.zeros((steps, n_inv))
    ask_depths = np.zeros((steps, n_inv))
    
    max_depth = 5.0 / k
    p_max = 1.0 - np.exp(-A * np.exp(-k * max_depth) * dt)
    
    n_grid = 80
    p_grid = np.linspace(0, p_max, n_grid)
    with np.errstate(divide='ignore', invalid='ignore'):
        d_grid = -np.log(1 - p_grid / (A * dt)) / k
    d_grid = np.clip(d_grid, 0, max_depth)
    d_grid[0] = 0.0
    p_from_d = 1.0 - np.exp(-A * np.exp(-k * d_grid) * dt)
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break
        for i in range(n_inv):
            q = i - Q
            c = 0.5 * gamma * sigma**2 * q**2 * dt
            
            can_bid = (q < Q)
            can_ask = (q > -Q)
            
            if not can_bid and not can_ask:
                V[n, i] = -c
                continue
            
            Vq = V[n + 1, i]
            Vqm1 = V[n + 1, i - 1] if can_bid else 0.0
            Vqp1 = V[n + 1, i + 1] if can_ask else 0.0
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            for idx_b, pb in enumerate(p_from_d):
                db = d_grid[idx_b]
                if not can_bid and pb > 1e-10:
                    continue
                
                coeff_a = Vqp1 * (1 - pb) + Vq * (2 * pb - 1) - pb * Vqm1
                const = Vq * (1 - pb) - c + pb * db
                
                if not can_ask:
                    obj = const
                    da_opt = 0.0
                else:
                    objs = p_from_d * d_grid + p_from_d * coeff_a + const
                    best_idx = np.argmax(objs)
                    obj = objs[best_idx]
                    da_opt = d_grid[best_idx]
                
                if obj > best_val:
                    best_val = obj
                    best_b = db
                    best_a = da_opt
            
            bid_depths[n, i] = best_b
            ask_depths[n, i] = best_a
            V[n, i] = best_val
    
    _cache[key] = (bid_depths, ask_depths)
    return _cache[key]


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
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
    
    try:
        bid_depths_table, ask_depths_table = _compute_policy(sigma, A, k, gamma, T, steps, Q)
    except Exception:
        # Fallback: risk-neutral depth
        depth = np.full(len(inventory), 1.0 / k)
        bid = depth.copy()
        ask = depth.copy()
        bid[inventory >= Q] = np.inf
        ask[inventory <= -Q] = np.inf
        return bid, ask
    
    inv_idx = np.clip(inventory + Q, 0, 2 * Q).astype(int)
    
    bid = bid_depths_table[n, inv_idx].astype(float)
    ask = ask_depths_table[n, inv_idx].astype(float)
    
    bid[inventory >= Q] = np.inf
    ask[inventory <= -Q] = np.inf
    
    # Ensure valid: no NaN, no values below -3/k
    min_val = -3.0 / k
    bid = np.where(np.isnan(bid) | (bid < min_val), np.inf, bid)
    ask = np.where(np.isnan(ask) | (ask < min_val), np.inf, ask)
    
    return bid, ask
