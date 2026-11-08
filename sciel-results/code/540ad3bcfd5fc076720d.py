"""Optimal market making with inventory risk – analytical first-order policy."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for policy tables keyed by (sigma, A, k, gamma, T, steps, Q)
_policy_cache = {}


def _build_policy_table(sigma, A, k, gamma, T, steps, Q):
    """
    Build a policy table via backward induction on the discrete inventory state space.
    For each time step and inventory level, compute optimal bid and ask depths.
    
    The value function V_n(q) satisfies:
    V_n(q) = max_{b,a >= 0} [
        (1 - exp(-A exp(-k b) dt)) * (b + V_{n+1}(q+1))
      + (1 - exp(-A exp(-k a) dt)) * (-a + V_{n+1}(q-1))
      - (1 - exp(-A exp(-k b) dt)) * (1 - exp(-A exp(-k a) dt)) * V_{n+1}(q)
      - gamma/2 * sigma^2 * q^2 * dt
    ]
    
    For interior q (not at ±Q), the first-order condition for bid depth b:
    A exp(-k b) dt * [exp(-A exp(-k b) dt) * (V(q+1) - V(q)) + b] = 0
    => exp(-A exp(-k b) dt) * (V(q+1) - V(q) + b) = 0
    
    Wait, let me re-derive. The objective for bid depth b (holding ask fixed):
    f(b) = p_b * (b + V_{n+1}(q+1)) + (1-p_b) * V_{n+1}(q)
    where p_b = 1 - exp(-A exp(-k b) dt)
    
    df/db = p_b' * (b + V(q+1) - V(q)) + p_b
    where p_b' = A k exp(-k b) exp(-A exp(-k b) dt)
    
    Setting df/db = 0:
    A k exp(-k b) exp(-A exp(-k b) dt) * (b + V(q+1) - V(q)) + (1 - exp(-A exp(-k b) dt)) = 0
    
    Let x = exp(-k b), so b = -ln(x)/k, and exp(-A x dt) = exp(-A x dt).
    A k x exp(-A x dt) * (-ln(x)/k + V(q+1) - V(q)) + 1 - exp(-A x dt) = 0
    A x exp(-A x dt) * (V(q+1) - V(q) - ln(x)/k) + 1 - exp(-A x dt) = 0
    
    This is a 1D root-finding problem in x ∈ (0, 1].
    """
    dt = T / steps
    q_values = np.arange(-Q, Q + 1)
    n_states = len(q_values)
    q_index = {q: i for i, q in enumerate(q_values)}
    
    # V_next[q_index] for the next time step
    V_next = np.zeros(n_states)
    
    # Store optimal depths: shape (steps, n_states, 2) -> [step, state, {bid, ask}]
    # But we only need the policy for the current step at evaluation time.
    # Actually, we need to precompute for all steps since quote() is called at each step.
    # Store bid_depths[step, q_idx] and ask_depths[step, q_idx]
    bid_table = np.zeros((steps, n_states))
    ask_table = np.zeros((steps, n_states))
    
    def solve_depth(V_plus, V_same, side_sign):
        """
        Solve for optimal depth given V(q+side_sign) and V(q).
        side_sign = +1 for bid (inventory increases), -1 for ask (inventory decreases).
        For bid: V_plus = V(q+1), V_same = V(q)
        For ask: V_plus = V(q-1), V_same = V(q), and the earned amount is -a.
        
        For bid: f(b) = p_b*(b + V(q+1)) + (1-p_b)*V(q)
                 df/db = A*k*x*exp(-A*x*dt)*(b + V(q+1)-V(q)) + p_b = 0
                 where x = exp(-k*b)
        
        For ask: f(a) = p_a*(-a + V(q-1)) + (1-p_a)*V(q)
                 df/da = A*k*x*exp(-A*x*dt)*(-a + V(q-1)-V(q)) + p_a = 0
                 where x = exp(-k*a)
        
        General form: A*k*x*exp(-A*x*dt)*(c - ln(x)/k) + 1 - exp(-A*x*dt) = 0
        where c = V(q+1)-V(q) for bid, c = V(q-1)-V(q) for ask.
        """
        c = V_plus - V_same
        # Solve: A*k*x*exp(-A*x*dt)*(c - ln(x)/k) + 1 - exp(-A*x*dt) = 0
        # Let g(x) = A*k*x*exp(-A*x*dt)*(c - ln(x)/k) + 1 - exp(-A*x*dt)
        # x in (0, 1]. At x=1 (b=0): g(1) = A*k*1*exp(-A*dt)*(c) + 1 - exp(-A*dt)
        #   = exp(-A*dt)*(A*k*c + 1) - 1
        # At x->0: exp(-A*x*dt)->1, ln(x)->-inf, so -ln(x)/k -> +inf
        #   g(x) ~ A*k*x*(c - ln(x)/k) + 1 - 1 = A*k*x*c - A*x*ln(x) -> 0 (since x*ln(x)->0)
        # Actually as x->0+, A*x*ln(x) -> 0, so g(x) -> 0 from... let's check sign.
        # A*k*x*(c - ln(x)/k) = A*k*c*x - A*x*ln(x). As x->0+, -A*x*ln(x) -> 0+.
        # So g(x) -> 0.
        
        # We want g(x) = 0. Let's use bisection or Newton.
        # g(1) = exp(-A*dt)*(A*k*c + 1) - 1
        # If g(1) >= 0, then optimal b = 0 (x = 1).
        # If g(1) < 0, there's a root in (0, 1).
        
        g1 = np.exp(-A * dt) * (A * k * c + 1.0) - 1.0
        if g1 >= 0:
            return 0.0  # depth = 0
        
        # Bisection in x in (eps, 1)
        eps = 1e-12
        lo, hi = eps, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            exp_term = np.exp(-A * mid * dt)
            g_mid = A * k * mid * exp_term * (c - np.log(mid) / k) + 1.0 - exp_term
            if g_mid > 0:
                lo = mid
            else:
                hi = mid
        x_opt = (lo + hi) / 2.0
        depth = -np.log(x_opt) / k
        return depth
    
    for n in range(steps - 1, -1, -1):
        V_curr = np.zeros(n_states)
        for i, q in enumerate(q_values):
            # Inventory holding cost
            hold_cost = 0.5 * gamma * sigma**2 * q**2 * dt
            
            if q == Q:
                # Can only quote ask
                # f(a) = p_a*(-a + V_next[q_idx(q-1)]) + (1-p_a)*V_next[q_idx(q)] - hold_cost
                # But at q=Q, no bid fill. Ask fill moves to Q-1.
                idx_q = i
                idx_qm1 = i - 1  # q-1
                a_opt = solve_depth(V_next[idx_qm1], V_next[idx_q], -1)
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                V_curr[i] = p_a * (-a_opt + V_next[idx_qm1]) + (1.0 - p_a) * V_next[idx_q] - hold_cost
                bid_table[n, i] = np.inf
                ask_table[n, i] = a_opt
            elif q == -Q:
                # Can only quote bid
                idx_q = i
                idx_qp1 = i + 1  # q+1
                b_opt = solve_depth(V_next[idx_qp1], V_next[idx_q], 1)
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                V_curr[i] = p_b * (b_opt + V_next[idx_qp1]) + (1.0 - p_b) * V_next[idx_q] - hold_cost
                bid_table[n, i] = b_opt
                ask_table[n, i] = np.inf
            else:
                idx_q = i
                idx_qp1 = i + 1
                idx_qm1 = i - 1
                
                # Joint optimization of b and a. For simplicity, alternate or use 
                # the fact that in the linear-quadratic case, bid and ask decouple 
                # approximately. Let's do one round of alternating optimization.
                b_opt = solve_depth(V_next[idx_qp1], V_next[idx_q], 1)
                a_opt = solve_depth(V_next[idx_qm1], V_next[idx_q], -1)
                
                # One more iteration with updated values
                p_b = 1.0 - np.exp(-A * np.exp(-k * b_opt) * dt)
                p_a = 1.0 - np.exp(-A * np.exp(-k * a_opt) * dt)
                
                # Expected V_next
                V_expected = (p_b * p_a * V_next[idx_q] +
                              p_b * (1 - p_a) * V_next[idx_qp1] +
                              (1 - p_b) * p_a * V_next[idx_qm1] +
                              (1 - p_b) * (1 - p_a) * V_next[idx_q])
                
                # Actually the correct expansion:
                # P(bid only) = p_b*(1-p_a): earn b, go to q+1
                # P(ask only) = (1-p_b)*p_a: earn -a, go to q-1
                # P(both) = p_b*p_a: earn b-a, go to q
                # P(neither) = (1-p_b)*(1-p_a): earn 0, go to q
                V_expected = (p_b * (1 - p_a) * (b_opt + V_next[idx_qp1]) +
                              (1 - p_b) * p_a * (-a_opt + V_next[idx_qm1]) +
                              p_b * p_a * (b_opt - a_opt + V_next[idx_q]) +
                              (1 - p_b) * (1 - p_a) * V_next[idx_q])
                
                V_curr[i] = V_expected - hold_cost
                bid_table[n, i] = b_opt
                ask_table[n, i] = a_opt
        
        V_next = V_curr
    
    return bid_table, ask_table, q_index


def _get_policy_table(params):
    key = (params["sigma"], params["A"], params["k"], params["gamma"],
           params["T"], params["steps"], params["Q"])
    if key not in _policy_cache:
        _policy_cache[key] = _build_policy_table(
            params["sigma"], params["A"], params["k"], params["gamma"],
            params["T"], params["steps"], params["Q"]
        )
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    n_paths = len(inventory)
    dt = T / steps
    
    # Determine current step index
    step_idx = int(round(t / dt))
    if step_idx >= steps:
        step_idx = steps - 1
    if step_idx < 0:
        step_idx = 0
    
    bid_table, ask_table, q_index = _get_policy_table(params)
    
    bid_depths = np.full(n_paths, np.inf)
    ask_depths = np.full(n_paths, np.inf)
    
    for i in range(n_paths):
        q = int(inventory[i])
        # Clamp to valid range
        if q > Q:
            q = Q
        elif q < -Q:
            q = -Q
        
        idx = q - (-Q)  # index in q_values = arange(-Q, Q+1)
        
        b = bid_table[step_idx, idx]
        a = ask_table[step_idx, idx]
        
        # Ensure validity: depths must be >= 0 and finite or inf
        if np.isfinite(b) and b < 0:
            b = 0.0
        if np.isfinite(a) and a < 0:
            a = 0.0
        
        # Check minimum depth constraint: values below -3/k are invalid
        # Depth should be >= 0. If b or a is negative, set to 0.
        
        bid_depths[i] = b
        ask_depths[i] = a
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
