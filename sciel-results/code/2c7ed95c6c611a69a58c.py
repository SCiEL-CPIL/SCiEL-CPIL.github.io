import numpy as np
import time
import threading

_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    # Precompute interpolation grid for the optimality condition
    # phi(x) = g(x)/h(x) - x
    # g(x) = 1 - exp(-A exp(-kx) dt)
    # h(x) = k A dt exp(-kx) exp(-A exp(-kx) dt)
    # We solve phi(x) = C for x
    
    x_max = 10.0 / k
    n_grid = 10000  # Increased resolution
    x_grid = np.linspace(0, x_max, n_grid)
    
    # Calculate phi(x)
    exp_kx = np.exp(-k * x_grid)
    exp_Au = np.exp(-A * exp_kx * dt)
    g = 1.0 - exp_Au
    h = k * A * dt * exp_kx * exp_Au
    
    # Avoid division by zero
    # h is very small at large x.
    # phi(x) ~ 1/k - x.
    # We can just compute it.
    phi = g / h - x_grid
    
    # phi is decreasing.
    # For np.interp, we need xp to be increasing.
    # So we use -phi as xp and x_grid as fp.
    # We want to find x given C.
    # -phi(x) = -C.
    # x = interp(-C, -phi, x_grid)
    
    neg_phi = -phi
    # Ensure neg_phi is strictly increasing? 
    # phi is strictly decreasing, so neg_phi is strictly increasing.
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    q_vals = np.arange(n_states) - Q
    tau = np.arange(N, 0, -1) * dt # Time remaining for step n is (N-n)*dt? 
    # Step n is at t = n*dt. Time remaining is (N-1-n)*dt? Or (N-n)*dt?
    # The loop is for n in range(N-1, -1, -1).
    # At n=N-1, time remaining is dt.
    # At n=0, time remaining is N*dt = T.
    # So tau = (N - n) * dt.

    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Fallback: use AS heuristic for remaining steps
            for n_fill in range(n, N):
                tau_rem = (N - n_fill) * dt
                # AS depths
                bid_h = 1.0/k + gamma * sigma**2 * q_vals * tau_rem
                ask_h = 1.0/k - gamma * sigma**2 * q_vals * tau_rem
                # Clip to valid range
                bid_h = np.clip(bid_h, 0, x_max)
                ask_h = np.clip(ask_h, 0, x_max)
                # Handle boundaries
                bid_h[-1] = np.inf # q=Q, no bid
                ask_h[0] = np.inf  # q=-Q, no ask
                opt_bid[n_fill, :] = bid_h
                opt_ask[n_fill, :] = ask_h
            return opt_bid, opt_ask
        
        V_curr = np.zeros(n_states)
        
        # Initialize b and a using Avellaneda-Stoikov heuristic
        tau_rem = (N - n) * dt
        bid_init = 1.0/k + gamma * sigma**2 * q_vals * tau_rem
        ask_init = 1.0/k - gamma * sigma**2 * q_vals * tau_rem
        # Clip to [0, x_max] for safety
        bid_init = np.clip(bid_init, 0, x_max)
        ask_init = np.clip(ask_init, 0, x_max)
        
        b = bid_init.copy()
        a = ask_init.copy()
        
        # Fixed point iteration
        # We can reduce iterations since we start close to solution
        for _iter in range(20): # Reduced max iterations due to better init
            if _check_deadline():
                 break
                
            # Compute p_a from current a
            p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
            
            # Compute C_b for each q
            # C_b = (1 - p_a) * Delta_plus - p_a * Delta_minus
            # Delta_plus = V_next[q+1] - V_next[q]
            # Delta_minus = V_next[q-1] - V_next[q]
            
            delta_plus = np.zeros(n_states)
            delta_minus = np.zeros(n_states)
            
            # q < Q => idx < 2Q
            # V_next[idx+1] - V_next[idx]
            delta_plus[:-1] = V_next[1:] - V_next[:-1]
            # q > -Q => idx > 0
            # V_next[idx-1] - V_next[idx]
            delta_minus[1:] = V_next[:-1] - V_next[1:]
            
            C_b = (1.0 - p_a) * delta_plus - p_a * delta_minus
            
            # Solve for b: phi(b) = C_b
            # b = interp(-C_b, neg_phi, x_grid)
            b_new = np.interp(-C_b, neg_phi, x_grid)
            
            # Update p_b
            p_b = 1.0 - np.exp(-A * np.exp(-k * b_new) * dt)
            
            # Compute C_a
            C_a = delta_minus * (1.0 - p_b) - p_b * delta_plus
            
            a_new = np.interp(-C_a, neg_phi, x_grid)
            
            # Convergence check
            if np.allclose(b_new, b, atol=1e-6) and np.allclose(a_new, a, atol=1e-6):
                b = b_new
                a = a_new
                break
            
            b = b_new
            a = a_new
        
        # Store optimal depths
        opt_bid[n, :] = b
        opt_ask[n, :] = a
        
        # Compute V_curr
        # p_b, p_a are already computed from final b, a
        
        # Transitions
        prob_q_plus = p_b * (1.0 - p_a)
        prob_q_minus = (1.0 - p_b) * p_a
        
        # Handle boundaries
        prob_q_plus[-1] = 0.0
        prob_q_minus[0] = 0.0
        
        # Reward
        reward = p_b * b + p_a * a
        reward[-1] = p_a[-1] * a[-1] # Only ask fill possible
        reward[0] = p_b[0] * b[0]     # Only bid fill possible
        
        # Risk
        risk = (gamma / 2.0) * (sigma**2) * (q_vals**2) * dt
        
        # Expectation
        V_next_plus = np.zeros(n_states)
        V_next_minus = np.zeros(n_states)
        V_next_plus[:-1] = V_next[1:]
        V_next_minus[1:] = V_next[:-1]
        
        exp_val = prob_q_plus * V_next_plus + prob_q_minus * V_next_minus
        
        # The "stay" probability term:
        prob_q = 1.0 - prob_q_plus - prob_q_minus
        exp_val += prob_q * V_next
        
        V_curr = reward - risk + exp_val
        
        V_next = V_curr

    return opt_bid, opt_ask

def _compute_regime_async(key, params):
    try:
        opt_bid, opt_ask = _precompute_regime(
            params['sigma'], params['A'], params['k'], params['gamma'], 
            params['T'], params['steps'], params['Q']
        )
        with _TABLE_LOCK:
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
    except Exception as e:
        # On error, cache a fallback
        with _TABLE_LOCK:
            if key not in _TABLE_CACHE:
                Q = params['Q']
                N = params['steps']
                k = params['k']
                opt_bid = np.full((N, 2*Q+1), 1.0/k)
                opt_ask = np.full((N, 2*Q+1), 1.0/k)
                opt_bid[:, -1] = np.inf
                opt_ask[:, 0] = np.inf
                _TABLE_CACHE[key] = (opt_bid, opt_ask)

def quote(t, inventory, mid, params):
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            # Start async computation
            threading.Thread(target=_compute_regime_async, args=(key, params), daemon=True).start()
            # Return a temporary fallback while computing
            # Use AS heuristic for immediate response
            Q = params['Q']
            sigma = params['sigma']
            gamma = params['gamma']
            k = params['k']
            T = params['T']
            steps = params['steps']
            
            q = np.asarray(inventory)
            tau = (steps - n) * dt
            
            bid_h = 1.0/k + gamma * sigma**2 * q * tau
            ask_h = 1.0/k - gamma * sigma**2 * q * tau
            
            # Clip
            min_allowed = -3.0 / k
            bid_h = np.clip(bid_h, min_allowed, 10.0/k)
            ask_h = np.clip(ask_h, min_allowed, 10.0/k)
            
            # Handle boundaries
            # q == Q -> no bid
            bid_h[q == Q] = np.inf
            # q == -Q -> no ask
            ask_h[q == -Q] = np.inf
            
            return bid_h, ask_h

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    min_allowed = -3.0 / params['k']
    bid_depths = np.where(np.isnan(bid_depths) | (bid_depths == -np.inf) | (bid_depths < min_allowed), np.inf, bid_depths)
    ask_depths = np.where(np.isnan(ask_depths) | (ask_depths == -np.inf) | (ask_depths < min_allowed), np.inf, ask_depths)
    
    return bid_depths, ask_depths
