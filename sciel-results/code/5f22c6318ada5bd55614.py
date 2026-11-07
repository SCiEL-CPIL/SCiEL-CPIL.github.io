# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # safety margin before 330s kill

_OPT_TABLES = {}


def _get_optimal_table(params):
    """
    Pre-compute optimal bid and ask depths for all (step, inventory) states
    using vectorized backward induction.
    """
    key = (params["sigma"], params["A"], params["k"], params["gamma"], params["Q"], params["steps"])
    if key in _OPT_TABLES:
        return _OPT_TABLES[key]

    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    dt = T / steps

    # Depth grid
    n_depths = 100
    d_min = 1e-6
    d_max = 3.0 / k
    depths = np.geomspace(d_min, d_max, n_depths)
    
    # Precompute probabilities for the grid
    # p = 1 - exp(-A * exp(-k * d) * dt)
    exponents = -A * np.exp(-k * depths) * dt
    probs = 1.0 - np.exp(exponents)
    
    # V has shape (steps+1, 2Q+1)
    # Index 0 corresponds to q=-Q, index 2Q corresponds to q=Q
    n_q = 2 * Q + 1
    V = np.zeros((steps + 1, n_q))
    
    # Policy tables: store indices into 'depths' to save memory and ensure consistency
    # Or store the actual depths. Storing indices is safer.
    bid_idx = np.full((steps + 1, n_q), -1, dtype=np.int32)
    ask_idx = np.full((steps + 1, n_q), -1, dtype=np.int32)
    
    # Backward induction
    for n in range(steps - 1, -1, -1):
        # Check deadline
        if time.time() > _DEADLINE - 30:
            # Fallback: set indices to middle of grid
            mid_idx = n_depths // 2
            bid_idx[n, :] = mid_idx
            ask_idx[n, :] = mid_idx
            break
        
        V_next = V[n+1]
        
        # Interior states: q from -Q+1 to Q-1
        # Indices in V: 1 to 2Q-1
        if n_q > 2:
            q_interior = np.arange(1, n_q - 1)
            q_vals = q_interior - Q
            
            # Delta V+ = V[q+1] - V[q]
            # q+1 corresponds to index q_idx + 1
            dV_plus = V_next[q_interior + 1] - V_next[q_interior]
            # Delta V- = V[q-1] - V[q]
            # q-1 corresponds to index q_idx - 1
            dV_minus = V_next[q_interior - 1] - V_next[q_interior]
            
            # Penalty C = 0.5 * gamma * sigma^2 * q^2 * dt
            C = 0.5 * gamma * sigma**2 * q_vals**2 * dt
            
            # We want to maximize over b_idx, a_idx
            # b = depths[b_idx], a = depths[a_idx]
            # pb = probs[b_idx], pa = probs[a_idx]
            
            # Broadcast:
            # B: (1, n_depths, n_interior)
            # A: (n_depths, 1, n_interior)
            # This might be memory heavy: 100 * 100 * 59 = 590k floats. Fine.
            
            B = depths[np.newaxis, :, np.newaxis] * np.ones((1, 1, len(q_interior)))
            A_depth = depths[:, np.newaxis, np.newaxis] * np.ones((1, 1, len(q_interior)))
            
            PB = probs[np.newaxis, :, np.newaxis] * np.ones((1, 1, len(q_interior)))
            PA = probs[:, np.newaxis, np.newaxis] * np.ones((1, 1, len(q_interior)))
            
            # Gains = b*pb + a*pa
            Gains = B * PB + A_depth * PA
            
            # Transitions = pb*(1-pa)*dV_plus + pa*(1-pb)*dV_minus
            # dV_plus, dV_minus shape: (n_interior,)
            # Need to broadcast to (1, 1, n_interior)
            dVp = dV_plus[np.newaxis, np.newaxis, :]
            dVm = dV_minus[np.newaxis, np.newaxis, :]
            
            Transitions = PB * (1.0 - PA) * dVp + PA * (1.0 - PB) * dVm
            
            # Base = V_next[q] - C
            Base = (V_next[q_interior] - C)[np.newaxis, np.newaxis, :]
            
            Total = Gains + Transitions + Base
            
            # Maximize
            flat_idx = np.argmax(Total, axis=(0, 1))
            
            # flat_idx is index into (n_depths * n_depths) for each q
            # We need to convert to (b_idx, a_idx)
            # The axis order is (b_idx, a_idx). So flat index = b_idx * n_depths + a_idx
            b_idx_flat = flat_idx // n_depths
            a_idx_flat = flat_idx % n_depths
            
            bid_idx[n, q_interior] = b_idx_flat
            ask_idx[n, q_interior] = a_idx_flat
            V[n, q_interior] = Total.flat[flat_idx * (n_depths * n_depths) + np.arange(len(q_interior))] 
            # Wait, Total.flat indexing is tricky. 
            # Let's just grab the max values directly.
            V[n, q_interior] = np.max(Total, axis=(0, 1))

        # Boundary q = Q (index 2Q)
        # No bid fill. Maximize over a.
        q_val = Q
        C = 0.5 * gamma * sigma**2 * q_val**2 * dt
        # Transition to Q-1 (index 2Q-1)
        dV_minus = V_next[2*Q - 1] - V_next[2*Q]
        # Gain = a * pa
        # Transition = pa * dV_minus
        # Total = a*pa + pa*dV_minus + V_next[2Q] - C
        # = pa * (a + dV_minus) + V_next[2Q] - C
        
        a_depths = depths
        pa = probs
        vals = pa * (a_depths + dV_minus) + V_next[2*Q] - C
        best_a_idx = np.argmax(vals)
        ask_idx[n, 2*Q] = best_a_idx
        bid_idx[n, 2*Q] = -1 # No bid
        V[n, 2*Q] = vals[best_a_idx]
        
        # Boundary q = -Q (index 0)
        # No ask fill. Maximize over b.
        q_val = -Q
        C = 0.5 * gamma * sigma**2 * q_val**2 * dt
        # Transition to -Q+1 (index 1)
        dV_plus = V_next[1] - V_next[0]
        # Gain = b * pb
        # Transition = pb * dV_plus
        # Total = b*pb + pb*dV_plus + V_next[0] - C
        # = pb * (b + dV_plus) + V_next[0] - C
        
        b_depths = depths
        pb = probs
        vals = pb * (b_depths + dV_plus) + V_next[0] - C
        best_b_idx = np.argmax(vals)
        bid_idx[n, 0] = best_b_idx
        ask_idx[n, 0] = -1 # No ask
        V[n, 0] = vals[best_b_idx]
        
        # Fill in -1 indices with a default valid depth (e.g., middle) for safety in quote function
        # Though quote function should handle -1 by returning inf or a default.
        # The task says np.inf means no quote.
        # So -1 in idx table should map to np.inf in quote.
        
    # Convert indices to actual depths for return, or return indices and let quote convert?
    # Returning depths is easier for quote.
    bid_depths_table = np.full((steps + 1, n_q), np.inf)
    ask_depths_table = np.full((steps + 1, n_q), np.inf)
    
    # Fill valid indices
    valid_bid = bid_idx >= 0
    bid_depths_table[valid_bid] = depths[bid_idx[valid_bid]]
    
    valid_ask = ask_idx >= 0
    ask_depths_table[valid_ask] = depths[ask_idx[valid_ask]]
    
    _OPT_TABLES[key] = (bid_depths_table, ask_depths_table)
    return _OPT_TABLES[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    # Check deadline
    if time.time() > _DEADLINE - 5:
        # Fallback to simple heuristic
        depth = 1.0 / params["k"]
        n = len(inventory)
        return np.full(n, depth), np.full(n, depth)
    
    # Get or compute optimal table
    bid_table, ask_table = _get_optimal_table(params)
    
    Q = params["Q"]
    steps = params["steps"]
    dt = params["T"] / steps
    n = int(round(t / dt))
    n = min(max(n, 0), steps - 1)
    
    # Map inventory to index
    # inventory is in [-Q, Q]
    q_idx = inventory + Q
    
    # Clip to valid range
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    # Get depths for each path
    bid_depths = bid_table[n, q_idx]
    ask_depths = ask_table[n, q_idx]
    
    # Ensure values are valid: not below -3/k, not -inf, not NaN
    # np.inf is allowed as "no quote".
    # But the problem says "np.inf means no quote".
    # And "Values below -3/k ... make the whole result invalid".
    # So we just need to ensure no NaN and no values < -3/k.
    # Our depths are positive, so they are > -3/k.
    
    # Replace any remaining NaNs with a safe value
    min_valid = -3.0 / params["k"]
    bid_depths = np.where(np.isnan(bid_depths), 1.0 / params["k"], bid_depths)
    ask_depths = np.where(np.isnan(ask_depths), 1.0 / params["k"], ask_depths)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
