"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_POLICY_CACHE = {}
_GLOBAL_START = time.time()
_DEADLINE = _GLOBAL_START + 310.0  # Safety margin before 330s kill


def _solve_optimal_depth(delta, A, k, dt):
    """
    Solve for the optimal depth d >= 0 maximizing:
        f(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + delta)
    
    The first-order condition gives:
        A*exp(-k*d)*dt * (d + delta) * exp(-A*exp(-k*d)*dt) = 1 - exp(-A*exp(-k*d)*dt)
    
    Let x = A*exp(-k*d)*dt, then d = (1/k)*ln(A*dt/x)
    
    f(d) = (1 - exp(-x)) * (d + delta)
    df/dd = exp(-k*d)*A*dt*exp(-x)*(d+delta) + (1-exp(-x))*(-1) = 0
    (in the limit, the derivative of (1-exp(-x)) w.r.t. d is k*x*exp(-x))
    
    Actually let me be more careful. Let u = A*exp(-k*d)*dt.
    Then du/dd = -k*u, and d = (1/k)*ln(A*dt/u).
    
    f(d) = (1 - exp(-u)) * (d + delta)
    df/dd = du/dd * d/du [(1-exp(-u))(d+delta)] + (1-exp(-u)) * dd/dd
    = -k*u * d/du[(1-exp(-u))(d+delta)] + (1-exp(-u))
    
    d/du[(1-exp(-u))(d+delta)] where d = (1/k)*ln(A*dt/u)
    = exp(-u) * (d+delta) + (1-exp(-u)) * d/du[d]
    = exp(-u)*(d+delta) + (1-exp(-u)) * (-1/(k*u))
    
    So df/dd = -k*u * [exp(-u)*(d+delta) - (1-exp(-u))/(k*u)] + (1-exp(-u))
    = -k*u*exp(-u)*(d+delta) + (1-exp(-u)) + (1-exp(-u))
    = -k*u*exp(-u)*(d+delta) + 2*(1-exp(-u))
    
    Wait, that doesn't look right. Let me redo.
    
    f(d) = (1 - exp(-u(d))) * (d + delta), where u(d) = A*exp(-k*d)*dt
    
    df/dd = -exp(-u) * u'(d) * (d+delta) + (1-exp(-u)) * 1
    = -exp(-u) * (-k*u) * (d+delta) + (1-exp(-u))
    = k*u*exp(-u)*(d+delta) + (1-exp(-u))
    
    Setting to zero:
    k*u*exp(-u)*(d+delta) = -(1-exp(-u))
    
    But both sides... wait, k*u*exp(-u)*(d+delta) is positive when d+delta > 0, and -(1-exp(-u)) is negative. So there's no solution when d+delta > 0?
    
    Hmm, let me reconsider. The derivative should be:
    df/dd = k*u*exp(-u)*(d+delta) - (1-exp(-u))*(-1)  ... no.
    
    Let me be very careful:
    f(d) = (1 - e^{-u}) (d + delta)
    where u = A e^{-kd} dt
    
    du/dd = -k A e^{-kd} dt = -k u
    
    df/dd = (d/d dd (1-e^{-u})) (d+delta) + (1-e^{-u}) (d/dd (d+delta))
    = (e^{-u} * du/dd) (d+delta) + (1-e^{-u}) * 1
    = (e^{-u} * (-ku)) (d+delta) + (1-e^{-u})
    = -k u e^{-u} (d+delta) + (1-e^{-u})
    
    Setting df/dd = 0:
    k u e^{-u} (d+delta) = 1 - e^{-u}
    
    Now, if d+delta > 0, LHS > 0 and RHS > 0 (for u > 0), so there can be a solution.
    If d+delta < 0, LHS < 0 and RHS > 0, so no solution (f is decreasing).
    If d+delta = 0, LHS = 0 and RHS > 0, so f is increasing at d=0.
    
    So the optimal d is either 0 (if f is decreasing everywhere for d>0) or the solution to:
    k u e^{-u} (d+delta) = 1 - e^{-u}
    where u = A e^{-kd} dt and d = (1/k) ln(A dt / u).
    
    Substituting d:
    k u e^{-u} ((1/k) ln(A dt / u) + delta) = 1 - e^{-u}
    u e^{-u} (ln(A dt / u) + k*delta) = 1 - e^{-u}
    
    Let's define g(u) = u e^{-u} (ln(A dt / u) + k*delta) - (1 - e^{-u}) = 0
    
    This is a transcendental equation in u. We can solve it numerically for each state.
    """
    # We need to find d >= 0 maximizing f(d) = (1-exp(-A*exp(-k*d)*dt))*(d+delta)
    
    # Check if d=0 is optimal: f'(0) = -k*u0*e^{-u0}*delta + (1-e^{-u0}) where u0 = A*dt
    # If f'(0) <= 0, then d=0 is optimal (since f is concave-ish near 0)
    
    u0 = A * dt
    if u0 < 1e-15:
        return 0.0
    
    # f'(0) = -k*u0*exp(-u0)*delta + (1-exp(-u0))
    fp0 = -k * u0 * np.exp(-u0) * delta + (1.0 - np.exp(-u0))
    
    if fp0 <= 0:
        return 0.0
    
    # Otherwise, solve for the interior optimum.
    # The equation is: k*u*exp(-u)*(d+delta) = 1 - exp(-u)
    # with d = (1/k)*ln(A*dt/u)
    
    # Let's use Newton's method on h(d) = df/dd = -k*u*exp(-u)*(d+delta) + (1-exp(-u))
    # where u = A*exp(-k*d)*dt
    
    # Start with an initial guess
    # For large delta, the optimum is small; for small delta, it can be larger
    # A reasonable initial guess: d_init = max(0, (1/k)*log(max(A*dt, 1e-10)))
    
    # Actually, let's just do a few Newton iterations
    d = max(0.0, 0.5 / k)  # initial guess
    
    for _ in range(30):
        u = A * np.exp(-k * d) * dt
        if u < 1e-15:
            break
        # h(d) = -k*u*exp(-u)*(d+delta) + (1-exp(-u))
        h = -k * u * np.exp(-u) * (d + delta) + (1.0 - np.exp(-u))
        # h'(d) = derivative of h w.r.t. d
        # h = -k*u*exp(-u)*(d+delta) + (1-exp(-u))
        # du/dd = -k*u
        # d/du[-k*u*exp(-u)*(d+delta)] = -k*(d+delta)*exp(-u)*(1-u)  [d/du of u*exp(-u) = exp(-u)(1-u)]
        # d/du[(1-exp(-u))] = exp(-u)
        # h'(d) = [-k*(d+delta)*exp(-u)*(1-u) + exp(-u)] * du/dd
        #       = [-k*(d+delta)*exp(-u)*(1-u) + exp(-u)] * (-k*u)
        #       = k*u*exp(-u)*[k*(d+delta)*(1-u) - 1]
        
        hp = k * u * np.exp(-u) * (k * (d + delta) * (1.0 - u) - 1.0)
        
        if abs(hp) < 1e-15:
            break
        
        d_new = d - h / hp
        if d_new < 0:
            d_new = 0.0
        if abs(d_new - d) < 1e-12:
            d = d_new
            break
        d = d_new
    
    return max(0.0, d)


def _compute_policy(sigma, A, k, gamma, T, steps, Q):
    """
    Computes the optimal bid and ask depths for each (time, inventory) state
    using the separable 1D approximation with analytical depth optimization.
    """
    dt = T / steps
    n_states = 2 * Q + 1
    
    V = np.zeros((steps + 1, n_states))
    opt_bid = np.zeros((steps, n_states))
    opt_ask = np.zeros((steps, n_states))
    
    q_vals = np.arange(-Q, Q + 1)
    C_inv = 0.5 * gamma * sigma**2 * q_vals**2 * dt
    
    for n in range(steps - 1, -1, -1):
        if time.time() > _DEADLINE:
            break
            
        V_next = V[n + 1]
        
        # delta_plus[q] = V_next[q+1] - V_next[q] for q < Q, 0 for q = Q
        delta_plus = np.zeros(n_states)
        delta_plus[:-1] = V_next[1:] - V_next[:-1]
        
        # delta_minus[q] = V_next[q-1] - V_next[q] for q > -Q, 0 for q = -Q
        delta_minus = np.zeros(n_states)
        delta_minus[1:] = V_next[:-1] - V_next[1:]
        
        # For each state, solve for optimal bid and ask depth analytically
        bid_depths = np.zeros(n_states)
        ask_depths = np.zeros(n_states)
        
        for q in range(n_states):
            bid_depths[q] = _solve_optimal_depth(delta_plus[q], A, k, dt)
            ask_depths[q] = _solve_optimal_depth(delta_minus[q], A, k, dt)
        
        opt_bid[n, :] = bid_depths
        opt_ask[n, :] = ask_depths
        
        # Compute the value added by optimal actions
        u_bid = A * np.exp(-k * bid_depths) * dt
        u_ask = A * np.exp(-k * ask_depths) * dt
        
        p_bid = 1.0 - np.exp(-u_bid)
        p_ask = 1.0 - np.exp(-u_ask)
        
        bid_reward = p_bid * (bid_depths + delta_plus)
        ask_reward = p_ask * (ask_depths + delta_minus)
        
        V[n, :] = V_next - C_inv + bid_reward + ask_reward
    
    return opt_bid, opt_ask


def _get_policy(params):
    """
    Retrieves or computes the optimal policy for the given parameters.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
    
    opt_bid, opt_ask = _compute_policy(
        params['sigma'], params['A'], params['k'], params['gamma'],
        params['T'], params['steps'], params['Q']
    )
    
    _POLICY_CACHE[key] = (opt_bid, opt_ask)
    return opt_bid, opt_ask


def quote(t, inventory, mid, params):
    """
    Returns optimal bid and ask depths based on pre-computed DP policy.
    """
    opt_bid, opt_ask = _get_policy(params)
    
    dt = params['T'] / params['steps']
    n = int(round(t / dt))
    
    if n < 0:
        n = 0
    if n >= params['steps']:
        n = params['steps'] - 1
    
    Q = params['Q']
    q_idx = inventory + Q
    q_idx = np.clip(q_idx, 0, 2 * Q)
    
    bid_depths = opt_bid[n, q_idx]
    ask_depths = opt_ask[n, q_idx]
    
    # Ensure depths are valid: >= 0, finite
    bid_depths = np.maximum(bid_depths, 0.0)
    ask_depths = np.maximum(ask_depths, 0.0)
    
    # Clip to reasonable range to avoid issues
    max_depth = 10.0 / params['k']
    bid_depths = np.minimum(bid_depths, max_depth)
    ask_depths = np.minimum(ask_depths, max_depth)
    
    return bid_depths.astype(float), ask_depths.astype(float)

# EVOLVE-BLOCK-END
