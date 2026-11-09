import numpy as np
import time
import threading

# Global deadline tracking
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill

# Cache for precomputed tables
_TABLE_CACHE = {}
_TABLE_LOCK = threading.Lock()

def _check_deadline():
    return time.time() >= _DEADLINE

def _precompute_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Precompute optimal bid and ask depths for all steps and inventories using
    vectorized Newton iteration for the coupled bid/ask optimization.
    """
    dt = T / steps
    N = steps
    max_q = Q
    min_q = -Q
    n_states = 2 * Q + 1
    
    V_next = np.zeros(n_states)
    opt_bid = np.zeros((N, n_states))
    opt_ask = np.zeros((N, n_states))
    
    # Precompute constants
    p_0 = 1.0 - np.exp(-A * dt)
    p_0_prime = -k * A * dt * (1 - p_0)
    
    # Initial guess for depths
    b_guess = np.full(n_states, 1.0 / k)
    a_guess = np.full(n_states, 1.0 / k)
    
    for n in range(N - 1, -1, -1):
        if _check_deadline():
            # Return what we have so far, fill remaining with a reasonable default
            depth_default = 1.0 / k
            for n_fill in range(n, N):
                opt_bid[n_fill, :] = depth_default
                opt_ask[n_fill, :] = depth_default
            return opt_bid, opt_ask
        
        V_curr = np.zeros(n_states)
        
        # Compute deltas for all states
        delta_plus = np.zeros(n_states)
        delta_minus = np.zeros(n_states)
        
        # q < max_q means q_idx < n_states - 1
        if n_states > 1:
            delta_plus[:-1] = V_next[1:] - V_next[:-1]
            # q > min_q means q_idx > 0
            delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # Initialize b and a with previous step's values or guess
        b = b_guess.copy()
        a = a_guess.copy()
        
        # Newton iteration for coupled b and a
        for _iter in range(40):
            # Compute probabilities and derivatives for current b
            u_b = np.exp(-k * b)
            p_b = 1.0 - np.exp(-A * u_b * dt)
            dp_db = -k * A * dt * u_b * (1.0 - p_b)
            
            # Compute probabilities and derivatives for current a
            u_a = np.exp(-k * a)
            p_a = 1.0 - np.exp(-A * u_a * dt)
            dp_da = -k * A * dt * u_a * (1.0 - p_a)
            
            # Coupling terms
            C_b = delta_plus * (1.0 - p_a)
            C_a = delta_minus * (1.0 - p_b)
            
            # Objective for bid: f_b(x) = dp_db * (x + C_b) + p_b
            # Derivative: f_b'(x) = dp_db' * (x + C_b) + dp_db
            # dp_db = -k*A*dt*u*(1-p)
            # dp_db' = -k*A*dt * du * (1-p) + u * (-dp)
            # du/dx = -k*u
            # dp/dx = A*dt*u*(-1) * ... wait, let's be more careful
            # p = 1 - exp(-A*u*dt)
            # dp/dx = A*dt*u * exp(-A*u*dt) * (-k) = -k*A*dt*u*(1-p) = dp_db
            # d2p/dx2 = -k*A*dt * [du*(1-p) + u*(-dp/dx)]
            # = -k*A*dt * [-k*u*(1-p) + u*k*A*dt*u*(1-p)]
            # = k^2*A*dt*u*(1-p) - k^2*A^2*dt^2*u^2*(1-p)
            # = k^2*A*dt*u*(1-p)*(1 - A*dt*u)
            
            u_b2 = u_b * u_b
            d2p_db2 = k**2 * A * dt * u_b * (1.0 - p_b) * (1.0 - A * dt * u_b)
            f_b_prime = d2p_db2 * (b + C_b) + dp_db
            
            # Newton update for b
            f_b = dp_db * (b + C_b) + p_b
            # Handle edge cases: if f_b_prime is near zero, skip
            mask_b = np.abs(f_b_prime) > 1e-12
            b_new = b.copy()
            b_new[mask_b] = b[mask_b] - f_b[mask_b] / f_b_prime[mask_b]
            b_new = np.clip(b_new, 0.0, 10.0 / k)
            
            # Recompute with updated b for coupling
            u_b_new = np.exp(-k * b_new)
            p_b_new = 1.0 - np.exp(-A * u_b_new * dt)
            
            C_a_new = delta_minus * (1.0 - p_b_new)
            
            # Objective for ask: f_a(x) = dp_da * (x + C_a_new) + p_a
            d2p_da2 = k**2 * A * dt * u_a * (1.0 - p_a) * (1.0 - A * dt * u_a)
            f_a_prime = d2p_da2 * (a + C_a_new) + dp_da
            
            f_a = dp_da * (a + C_a_new) + p_a
            mask_a = np.abs(f_a_prime) > 1e-12
            a_new = a.copy()
            a_new[mask_a] = a[mask_a] - f_a[mask_a] / f_a_prime[mask_a]
            a_new = np.clip(a_new, 0.0, 10.0 / k)
            
            # Check convergence
            if np.max(np.abs(b_new - b)) < 1e-9 and np.max(np.abs(a_new - a)) < 1e-9:
                b = b_new
                a = a_new
                break
            b = b_new
            a = a_new
        
        opt_bid[n, :] = b
        opt_ask[n, :] = a
        
        # Update V_curr
        p_b_final = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
        p_a_final = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
        
        # For q = max_q (last index): no bid fill possible
        # For q = min_q (first index): no ask fill possible
        
        prob_q_plus = np.zeros(n_states)
        prob_q_minus = np.zeros(n_states)
        prob_q = np.zeros(n_states)
        reward = np.zeros(n_states)
        
        # Interior states: 1 to n_states-2
        if n_states > 2:
            prob_q_plus[1:-1] = p_b_final[1:-1] * (1.0 - p_a_final[1:-1])
            prob_q_minus[1:-1] = (1.0 - p_b_final[1:-1]) * p_a_final[1:-1]
            prob_q[1:-1] = 1.0 - prob_q_plus[1:-1] - prob_q_minus[1:-1]
            reward[1:-1] = p_b_final[1:-1] * b[1:-1] + p_a_final[1:-1] * a[1:-1]
        
        # q = max_q (last index)
        prob_q_minus[-1] = p_a_final[-1]
        prob_q[-1] = 1.0 - p_a_final[-1]
        reward[-1] = p_a_final[-1] * a[-1]
        
        # q = min_q (first index)
        prob_q_plus[0] = p_b_final[0]
        prob_q[0] = 1.0 - p_b_final[0]
        reward[0] = p_b_final[0] * b[0]
        
        # Expected value of next state
        V_q_plus = np.zeros(n_states)
        V_q_minus = np.zeros(n_states)
        if n_states > 1:
            V_q_plus[:-1] = V_next[1:]
            V_q_minus[1:] = V_next[:-1]
        
        exp_val = prob_q_plus * V_q_plus + prob_q_minus * V_q_minus + prob_q * V_next
        
        # Inventory: q = q_idx - Q
        q_vals = np.arange(n_states) - Q
        risk = (gamma / 2.0) * sigma**2 * q_vals**2 * dt
        
        V_curr = reward - risk + exp_val
        V_next = V_curr
        
        # Update guesses for next iteration
        b_guess = b.copy()
        a_guess = a.copy()
    
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    # Determine step index
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    if n < 0: n = 0
    if n >= params['steps']: n = params['steps'] - 1
    
    key = (params['sigma'], params['A'], params['k'], params['gamma'])
    
    with _TABLE_LOCK:
        if key not in _TABLE_CACHE:
            # Precompute
            start_time = time.time()
            opt_bid, opt_ask = _precompute_regime(
                params['sigma'], params['A'], params['k'], params['gamma'], 
                params['T'], params['steps'], params['Q']
            )
            _TABLE_CACHE[key] = (opt_bid, opt_ask)
            
            # Time guard (safety)
            if time.time() - start_time > 280:
                 depth = 1.0 / params["k"]
                 return np.full(len(inventory), depth), np.full(len(inventory), depth)

    opt_bid, opt_ask = _TABLE_CACHE[key]
    
    q_indices = np.asarray(inventory) + params['Q']
    
    bid_depths = opt_bid[n, q_indices]
    ask_depths = opt_ask[n, q_indices]
    
    return bid_depths, ask_depths
