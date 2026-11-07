"""Optimal market making with inventory risk using backward-induction DP."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
_DEADLINE = 320.0  # safety margin before 330s kill

# Cache for DP tables keyed by regime parameters
_dp_cache = {}

def _compute_dp_table(sigma, A, k, gamma, T, steps, Q):
    """Compute optimal bid and ask depths for each (time, inventory) pair."""
    dt = T / steps
    n_inv = 2 * Q + 1  # inventory from -Q to Q
    
    # V[t][q_idx] = optimal expected value from time t with inventory q
    # We do backward induction from t=steps down to t=0
    # V[steps][q] = 0 (no terminal charge)
    
    V = np.zeros((steps + 1, n_inv))
    # Store optimal depths: bid_depth[t][q_idx], ask_depth[t][q_idx]
    bid_depths = np.full((steps, n_inv), np.inf)
    ask_depths = np.full((steps, n_inv), np.inf)
    
    # For depth optimization, we need to find optimal bid_depth and ask_depth
    # Given: P_bid = 1 - exp(-A * exp(-k * d_bid) * dt)
    # Reward from bid: d_bid * P_bid (if we get filled, we earn d_bid)
    # But also inventory changes affect future value
    
    # The optimal depth for each side can be found by maximizing:
    # d * (1 - exp(-A*exp(-k*d)*dt)) + E[V_{t+1}(q + delta)]
    # where delta = -1 for bid fill, +1 for ask fill
    
    # Since bid and ask are independent, we can optimize them separately
    # but the expected future value couples them.
    
    # For a given state (t, q), let's denote:
    # p_bid(d) = 1 - exp(-A * exp(-k*d) * dt)
    # p_ask(d) = 1 - exp(-A * exp(-k*d) * dt)
    
    # The expected value is:
    # d_bid * p_bid + d_ask * p_ask - (gamma/2)*sigma^2*q^2*dt
    # + p_bid*p_ask * V[t+1][q]  (no fill)
    # + p_bid*(1-p_ask) * V[t+1][q-1]  (bid fill only)
    # + (1-p_bid)*p_ask * V[t+1][q+1]  (ask fill only)
    # + (1-p_bid)*(1-p_ask) * V[t+1][q]  -- wait, this is wrong
    
    # Actually:
    # No fill: (1-p_bid)*(1-p_ask) -> q stays, value V[t+1][q]
    # Bid only: p_bid*(1-p_ask) -> q-1, value V[t+1][q-1]  
    # Ask only: (1-p_bid)*p_ask -> q+1, value V[t+1][q+1]
    # Both: p_bid*p_ask -> q stays (bid -1, ask +1), value V[t+1][q]
    
    # So expected future value = 
    #   (1-p_bid)*(1-p_ask)*V[t+1][q] + p_bid*(1-p_ask)*V[t+1][q-1]
    #   + (1-p_bid)*p_ask*V[t+1][q+1] + p_bid*p_ask*V[t+1][q]
    # = V[t+1][q] + p_bid*(1-p_ask)*(V[t+1][q-1]-V[t+1][q])
    #   + (1-p_bid)*p_ask*(V[t+1][q+1]-V[t+1][q])
    
    # This is complex. Let me use a grid search over depths for each side.
    
    # Depth grid: from small to large
    # The optimal depth is typically O(1/k) to O(10/k)
    max_depth = 5.0 / k
    depth_grid = np.linspace(0.01, max_depth, 200)
    
    # Inventory indices: q_idx = q + Q, so q = q_idx - Q
    # q ranges from -Q to Q
    
    # Boundary conditions: no bid fill at q=Q, no ask fill at q=-Q
    # At q=Q: bid_depth = inf (no bid quote)
    # At q=-Q: ask_depth = inf (no ask quote)
    
    for t in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE - 5.0:
            break
        for q_idx in range(n_inv):
            q = q_idx - Q
            
            # If at boundary, one side is disabled
            if q == Q:
                # No bid quote
                bid_depths[t, q_idx] = np.inf
                # Optimize ask only
                best_val = -np.inf
                best_ask = np.inf
                for d_ask in depth_grid:
                    p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
                    # Immediate reward from ask
                    imm = d_ask * p_ask - 0.5 * gamma * sigma**2 * q**2 * dt
                    # Future: ask fill -> q+1, no fill -> q
                    fut = p_ask * V[t+1, min(q_idx+1, n_inv-1)] + (1-p_ask) * V[t+1, q_idx]
                    total = imm + fut
                    if total > best_val:
                        best_val = total
                        best_ask = d_ask
                V[t, q_idx] = best_val
                ask_depths[t, q_idx] = best_ask
            elif q == -Q:
                # No ask quote
                ask_depths[t, q_idx] = np.inf
                best_val = -np.inf
                best_bid = np.inf
                for d_bid in depth_grid:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
                    imm = d_bid * p_bid - 0.5 * gamma * sigma**2 * q**2 * dt
                    fut = p_bid * V[t+1, max(q_idx-1, 0)] + (1-p_bid) * V[t+1, q_idx]
                    total = imm + fut
                    if total > best_val:
                        best_val = total
                        best_bid = d_bid
                V[t, q_idx] = best_val
                bid_depths[t, q_idx] = best_bid
            else:
                # Both sides active
                best_val = -np.inf
                best_bid = np.inf
                best_ask = np.inf
                for d_bid in depth_grid:
                    p_bid = 1.0 - np.exp(-A * np.exp(-k * d_bid) * dt)
                    for d_ask in depth_grid:
                        p_ask = 1.0 - np.exp(-A * np.exp(-k * d_ask) * dt)
                        imm = d_bid * p_bid + d_ask * p_ask - 0.5 * gamma * sigma**2 * q**2 * dt
                        # Future value
                        no_fill = (1-p_bid)*(1-p_ask) * V[t+1, q_idx]
                        bid_only = p_bid*(1-p_ask) * V[t+1, q_idx-1]
                        ask_only = (1-p_bid)*p_ask * V[t+1, q_idx+1]
                        both = p_bid*p_ask * V[t+1, q_idx]
                        fut = no_fill + bid_only + ask_only + both
                        total = imm + fut
                        if total > best_val:
                            best_val = total
                            best_bid = d_bid
                            best_ask = d_ask
                V[t, q_idx] = best_val
                bid_depths[t, q_idx] = best_bid
                ask_depths[t, q_idx] = best_ask
    
    return bid_depths, ask_depths


def _get_tables(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _dp_cache:
        if time.time() > _DEADLINE - 10.0:
            # Fallback to seed strategy
            k = params['k']
            depth = np.full(len(params.get('inventory', [0])), 1.0 / k)
            _dp_cache[key] = (None, None)
            return None, None
        _dp_cache[key] = _compute_dp_table(
            params['sigma'], params['A'], params['k'], params['gamma'],
            params['T'], params['steps'], params['Q']
        )
    return _dp_cache[key]


def quote(t, inventory, mid, params):
    """Return (bid_depths, ask_depths) arrays of shape (4000,)."""
    n_paths = len(inventory)
    Q = params['Q']
    steps = params['steps']
    T = params['T']
    
    # Determine current step
    dt = T / steps
    step_idx = int(round(t / dt))
    step_idx = max(0, min(step_idx, steps - 1))
    
    # Try to get DP tables
    bid_table, ask_table = _get_tables(params)
    
    if bid_table is None:
        # Fallback: seed strategy
        depth = np.full(n_paths, 1.0 / params['k'])
        return depth, depth.copy()
    
    # Map inventory to index: q_idx = q + Q
    q_indices = (inventory + Q).astype(int)
    q_indices = np.clip(q_indices, 0, 2 * Q)
    
    # Look up optimal depths
    bid_out = bid_table[step_idx, q_indices]
    ask_out = ask_table[step_idx, q_indices]
    
    # Ensure valid: values must be >= -3/k (which is very negative, so mostly fine)
    # and not NaN or -inf
    k = params['k']
    min_valid = -3.0 / k
    
    # Replace any invalid values
    bid_out = np.where(np.isfinite(bid_out) & (bid_out >= min_valid), bid_out, np.inf)
    ask_out = np.where(np.isfinite(ask_out) & (ask_out >= min_valid), ask_out, np.inf)
    
    return bid_out, ask_out
# EVOLVE-BLOCK-END
