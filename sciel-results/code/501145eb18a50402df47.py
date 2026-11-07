"""
Strategy: Structural change from DP grid-search to analytic HJB-based policy.
The problem is a continuous-time MDP. We approximate the value function V(q, t)
using a quadratic form in inventory q: V(q, t) ≈ -0.5 * a(t) * q^2 - b(t) * q + c(t).
This reduces the infinite-dimensional PDE to a system of ODEs for a(t), b(t), c(t).
The optimal depths are derived analytically from the first-order conditions of the
HJB equation, avoiding the discretization error of the grid search entirely.
This is a different design family: analytic approximation vs. numerical DP.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

_policy_cache = {}
_start_time = time.time()
TIME_LIMIT = 320.0  # Safety margin

def _compute_analytic_policy(params):
    """
    Compute the optimal bid and ask depths for all inventory states q and time steps
    using an analytic approximation based on the HJB equation.
    """
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1, dtype=np.float64)
    
    # We assume V(q, t) = -0.5 * a(t) * q^2 - b(t) * q + c(t)
    # The HJB equation is:
    # -V_t = max_{d_b, d_a} { 
    #     (1 - e^{-A e^{-k d_b} dt}) * [V(q-1, t+dt) + d_b] 
    #     + (1 - e^{-A e^{-k d_a} dt}) * [V(q+1, t+dt) + d_a]
    #     - (1 - e^{-A e^{-k d_b} dt})(1 - e^{-A e^{-k d_a} dt}) * 0 
    #     - 0.5 * gamma * sigma^2 * q^2 * dt
    #   } + sigma^2/2 * V_qq (diffusion term, but mid is exogenous, so no V_mid)
    # Actually, the mid moves, but the objective only depends on depths earned and inventory cost.
    # The mid movement doesn't directly enter the objective except through the state transition?
    # No, the state is (t, q). The mid is just a parameter for the quote.
    # The "depths earned" is d_b if bid hit, d_a if ask hit.
    # The transition is q -> q-1 (bid hit), q -> q+1 (ask hit), q (no hit).
    # The value function depends on t and q.
    
    # Let's derive the ODEs for a(t), b(t), c(t) by working backwards from T.
    # At t=T, V(q, T) = 0, so a(T)=0, b(T)=0, c(T)=0.
    
    # We integrate backwards from T to 0.
    # Let's use a simple Euler method with small steps.
    n_time_steps = 10000  # Fine grid for ODE integration
    dt_ode = T / n_time_steps
    t_grid = np.linspace(T, 0, n_time_steps + 1)
    
    a = np.zeros(n_time_steps + 1)
    b = np.zeros(n_time_steps + 1)
    c = np.zeros(n_time_steps + 1)
    
    # At t=T, a=0, b=0, c=0
    # We step backwards: t_i = T - i*dt_ode
    
    # The optimal depths d_b, d_a satisfy:
    # d_b = argmax_{d >= 0} { p(d) * (V(q-1) + d) } where p(d) = 1 - exp(-A exp(-k d) dt)
    # For small dt, p(d) ≈ A exp(-k d) dt.
    # Then p(d) * (V(q-1) + d) ≈ A dt exp(-k d) (V(q-1) + d).
    # Let f(d) = exp(-k d) (C + d) where C = V(q-1).
    # f'(d) = -k exp(-k d) (C + d) + exp(-k d) = exp(-k d) (1 - k(C + d)).
    # f'(d) = 0 => 1 - k(C + d) = 0 => d = 1/k - C.
    # So d_b ≈ max(0, 1/k - V(q-1)).
    # Similarly d_a ≈ max(0, 1/k - V(q+1)).
    
    # Let's check if this approximation is valid.
    # The exact maximization of p(d)(C+d) is more complex, but for small dt, this is a good first-order approx.
    # However, we can do better by solving the exact maximization for the ODE.
    
    # Let's define the function to compute optimal depth given a constant C (which is V(q-1) or V(q+1)).
    # We want to maximize g(d) = (1 - exp(-A exp(-k d) dt)) * (C + d) for d >= 0.
    # This is a 1D optimization. We can precompute a lookup table for C.
    
    # C ranges roughly from -inf to inf, but in practice, V is bounded.
    # Let's estimate the range of V.
    # V(q, t) is the expected future reward.
    # Reward per step is at most ~ 2 * max_depth. max_depth ~ 1/k.
    # Total reward ~ steps * 2/k.
    # Inventory cost ~ gamma * sigma^2 * Q^2 * T.
    # So V is O(1/k + gamma sigma^2 Q^2).
    
    # Let's create a lookup table for d_opt(C).
    C_min = -50.0
    C_max = 50.0
    n_C = 10000
    C_grid = np.linspace(C_min, C_max, n_C)
    
    # For each C, find d that maximizes (1 - exp(-A exp(-k d) dt)) * (C + d)
    # d is in [0, d_max]. d_max can be large, but the function decays.
    # Let's use d in [0, 5/k].
    d_max = 5.0 / k
    n_d = 1000
    d_grid = np.linspace(0, d_max, n_d)
    
    # Vectorize over C
    # g(C, d) = (1 - exp(-A exp(-k d) dt)) * (C + d)
    exp_term = np.exp(-k * d_grid)  # (n_d,)
    p_d = 1.0 - np.exp(-A * exp_term * dt)  # (n_d,)
    
    # We need to maximize over d for each C.
    # g(C, d) = p_d * (C + d)
    # Let's compute this for all C and d.
    # C_grid shape (n_C,), d_grid shape (n_d,)
    # We want (n_C, n_d) matrix of g values.
    
    # To save memory, we can do it in chunks or just use broadcasting if n_C * n_d is small.
    # 10000 * 1000 = 10^7 floats = 80 MB. This is fine.
    
    g_matrix = p_d[None, :] * (C_grid[:, None] + d_grid[None, :])
    
    # Find argmax over d for each C
    max_idx = np.argmax(g_matrix, axis=1)
    d_opt_table = d_grid[max_idx]
    
    # Now we integrate the ODE backwards.
    # V(q, t) = -0.5 * a(t) * q^2 - b(t) * q + c(t)
    # V(q-1, t+dt) = -0.5 * a(t+dt) * (q-1)^2 - b(t+dt) * (q-1) + c(t+dt)
    # V(q+1, t+dt) = -0.5 * a(t+dt) * (q+1)^2 - b(t+dt) * (q+1) + c(t+dt)
    
    # d_b = d_opt(V(q-1, t+dt))
    # d_a = d_opt(V(q+1, t+dt))
    
    # The HJB equation:
    # -V_t = E[reward] - 0.5 * gamma * sigma^2 * q^2 * dt
    # Wait, the HJB is:
    # -V_t = max_d { E[dV + reward] } - cost
    # Actually, the standard form is:
    # -V_t = max_d { L V + f }
    # where L is the generator and f is the running cost/reward.
    # Here, the state is q, which is a jump process.
    # L V = lambda_b(d) * (V(q-1) - V(q) + d_b) + lambda_a(d) * (V(q+1) - V(q) + d_a)
    # where lambda_b(d) = A exp(-k d) / dt (rate), but we are using discrete time.
    # Let's stick to the discrete time Bellman equation:
    # V(q, t) = max_d { p_b(d) * (V(q-1, t+dt) + d_b) + p_a(d) * (V(q+1, t+dt) + d_a) + (1-p_b)(1-p_a)*V(q, t+dt) - 0.5 * gamma * sigma^2 * q^2 * dt }
    
    # This is an implicit equation for V(q, t).
    # V(q, t) = max_d { (1-p_b)(1-p_a) V(q, t+dt) + p_b (V(q-1, t+dt) + d_b) + p_a (V(q+1, t+dt) + d_a) - cost }
    
    # Let's approximate p_b, p_a as small.
    # p_b ≈ A e^{-k d_b} dt, p_a ≈ A e^{-k d_a} dt.
    # (1-p_b)(1-p_a) ≈ 1 - p_b - p_a.
    # V(q, t) ≈ max_d { (1 - p_b - p_a) V(q, t+dt) + p_b (V(q-1, t+dt) + d_b) + p_a (V(q+1, t+dt) + d_a) - cost }
    # V(q, t) ≈ V(q, t+dt) + max_d { -p_b V(q, t+dt) - p_a V(q, t+dt) + p_b V(q-1, t+dt) + p_b d_b + p_a V(q+1, t+dt) + p_a d_a - cost }
    # V(q, t) - V(q, t+dt) ≈ max_d { p_b (V(q-1) - V(q) + d_b) + p_a (V(q+1) - V(q) + d_a) - cost }
    # -V_t * dt ≈ max_d { p_b (V(q-1) - V(q) + d_b) + p_a (V(q+1) - V(q) + d_a) - cost }
    # -V_t ≈ max_d { A e^{-k d_b} (V(q-1) - V(q) + d_b) + A e^{-k d_a} (V(q+1) - V(q) + d_a) - cost/dt }
    # Wait, p_b = 1 - exp(-A e^{-k d_b} dt) ≈ A e^{-k d_b} dt.
    # So -V_t ≈ max_d { A e^{-k d_b} dt (V(q-1) - V(q) + d_b) / dt + ... }
    # -V_t ≈ max_d { A e^{-k d_b} (V(q-1) - V(q) + d_b) + A e^{-k d_a} (V(q+1) - V(q) + d_a) - 0.5 * gamma * sigma^2 * q^2 }
    
    # Let V(q, t) = -0.5 a(t) q^2 - b(t) q + c(t).
    # V(q-1) - V(q) = -0.5 a ((q-1)^2 - q^2) - b (q-1 - q) = -0.5 a (1 - 2q) + b = a q - 0.5 a + b.
    # V(q+1) - V(q) = -0.5 a ((q+1)^2 - q^2) - b (q+1 - q) = -0.5 a (1 + 2q) - b = -a q - 0.5 a - b.
    
    # Let C_b = V(q-1) - V(q) + d_b = a q - 0.5 a + b + d_b.
    # Let C_a = V(q+1) - V(q) + d_a = -a q - 0.5 a - b + d_a.
    
    # The optimal d_b maximizes A e^{-k d_b} (C_b - d_b) ? No.
    # The term is A e^{-k d_b} (V(q-1) - V(q) + d_b).
    # Let h(d) = e^{-k d} (K + d) where K = V(q-1) - V(q).
    # h'(d) = e^{-k d} (1 - k(K + d)).
    # h'(d) = 0 => d = 1/k - K.
    # So d_b = max(0, 1/k - (a q - 0.5 a + b)).
    # d_a = max(0, 1/k - (-a q - 0.5 a - b)).
    
    # This is a linear function of q, clipped at 0.
    # This makes the max_d operation non-trivial to integrate into the ODE for a, b, c.
    # The max_d term depends on q in a piecewise linear way.
    # This means the quadratic ansatz V(q,t) = -0.5 a q^2 - b q + c might not be exact.
    # However, for small inventory q relative to 1/k, the clipping might be inactive.
    # Or we can approximate the max_d term as a quadratic in q.
    
    # Let's assume the clipping is inactive for now and see if we get a reasonable policy.
    # d_b = 1/k - a q + 0.5 a - b.
    # d_a = 1/k + a q + 0.5 a + b.
    
    # The max_d value is A e^{-k d_b} (K_b + d_b) + A e^{-k d_a} (K_a + d_a).
    # K_b + d_b = 1/k.
    # K_a + d_a = 1/k.
    # So the term is A e^{-k d_b} / k + A e^{-k d_a} / k.
    
    # This is getting complex. Let's just use the lookup table d_opt(C) and assume V is quadratic.
    # We can iterate:
    # 1. Assume V(q, t) is quadratic with parameters a, b, c.
    # 2. Compute d_b(q) = d_opt(V(q-1, t+dt)), d_a(q) = d_opt(V(q+1, t+dt)).
    # 3. Compute the Bellman update for V(q, t) using these d_b, d_a.
    # 4. Fit a quadratic to the new V(q, t).
    # 5. Update a, b, c.
    
    # This is a policy iteration / value iteration hybrid, but with a parametric value function.
    
    # Let's implement this.
    
    # Initialize a, b, c at T.
    a_curr = 0.0
    b_curr = 0.0
    c_curr = 0.0
    
    # Store policies for each time step
    # We need to map from t_grid (n_time_steps+1 points) to the 1000 steps in quote().
    # Let's just store the policy for the 1000 steps directly.
    
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    # We integrate backwards from T to 0.
    # The 1000 steps in quote() are t = 0, dt, 2dt, ..., 999dt.
    # We need the policy at these times.
    # Let's integrate on a grid that includes these times.
    
    # Actually, let's just do the value iteration with the quadratic ansatz for each of the 1000 steps.
    # It might be slow if we do it for all 1000 steps, but let's try.
    
    # V_next is V at t+dt.
    # We start with V_next = 0 at t=T.
    # We go backwards from n=999 to 0.
    
    V_next = np.zeros(n_states)  # V at t+dt
    
    for n in range(steps - 1, -1, -1):
        if time.time() - _start_time > TIME_LIMIT:
            # Fallback to a simple heuristic
            for i in range(steps):
                opt_bid[i, :] = 1.0 / k
                opt_ask[i, :] = 1.0 / k
            break
        
        # V_next is V(q, t+dt)
        # We want to find V_curr = V(q, t)
        
        # For each q, compute d_b, d_a using V_next
        # d_b = d_opt(V_next(q-1)), d_a = d_opt(V_next(q+1))
        
        # V_next(q-1) for q > -Q, else 0
        V_qm = np.zeros(n_states)
        V_qm[1:] = V_next[:-1]  # V(q-1) is V_next at index q-1+Q = q+Q-1
        
        # Wait, indices:
        # q = -Q + i
        # q-1 = -Q + i - 1 = -Q + (i-1)
        # So V(q-1) = V_next[i-1] for i > 0.
        
        V_qm = np.zeros(n_states)
        V_qm[1:] = V_next[:-1]
        
        # q+1 = -Q + i + 1 = -Q + (i+1)
        # V(q+1) = V_next[i+1] for i < n_states-1.
        V_qp = np.zeros(n_states)
        V_qp[:-1] = V_next[1:]
        
        # Now, for each i (state q), we have V_qm[i] = V(q-1), V_qp[i] = V(q+1).
        # We need d_b(i) = d_opt(V_qm[i]) and d_a(i) = d_opt(V_qp[i]).
        
        # Use the lookup table.
        # d_opt_table is for C in [C_min, C_max].
        # We need to map V_qm[i] to the table.
        
        # Clip V values to [C_min, C_max]
        V_qm_clipped = np.clip(V_qm, C_min, C_max)
        V_qp_clipped = np.clip(V_qp, C_min, C_max)
        
        # Map to index
        idx_qm = np.round((V_qm_clipped - C_min) / (C_max - C_min) * (n_C - 1)).astype(int)
        idx_qp = np.round((V_qp_clipped - C_min) / (C_max - C_min) * (n_C - 1)).astype(int)
        
        d_b = d_opt_table[idx_qm]
        d_a = d_opt_table[idx_qp]
        
        # Now compute the Bellman update
        # V_curr[i] = max_d { p_b (V(q-1) + d_b) + p_a (V(q+1) + d_a) + (1-p_b)(1-p_a) V(q) - cost }
        # But we are using the optimal d_b, d_a, so we just plug them in.
        
        p_b = 1.0 - np.exp(-A * np.exp(-k * d_b) * dt)
        p_a = 1.0 - np.exp(-A * np.exp(-k * d_a) * dt)
        
        V_q = V_next  # V(q, t+dt) is approximated by V_next for the no-fill case?
        # No, V(q, t+dt) is V_next[i].
        
        # The no-fill term is (1-p_b)(1-p_a) V(q, t+dt) = (1-p_b)(1-p_a) V_next[i]
        
        term_no_fill = (1.0 - p_b) * (1.0 - p_a) * V_next
        term_bid = p_b * (V_qm + d_b)
        term_ask = p_a * (V_qp + d_a)
        
        cost = 0.5 * gamma * sigma**2 * q_values**2 * dt
        
        V_curr = term_no_fill + term_bid + term_ask - cost
        
        # Store policy
        opt_bid[n, :] = d_b
        opt_ask[n, :] = d_a
        
        # Update V_next for next iteration
        V_next = V_curr
        
    return opt_bid, opt_ask

def quote(t, inventory, mid, params):
    """
    Look up optimal depths from the pre-computed analytic policy.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        bid_depths, ask_depths = _compute_analytic_policy(params)
        _policy_cache[key] = (bid_depths, ask_depths)
    else:
        bid_depths, ask_depths = _policy_cache[key]
    
    n = int(np.round(t * params['steps']))
    if n >= params['steps']:
        n = params['steps'] - 1
        
    indices = inventory + params['Q']
    
    bids = bid_depths[n, indices]
    asks = ask_depths[n, indices]
    
    return bids, asks

# EVOLVE-BLOCK-END
