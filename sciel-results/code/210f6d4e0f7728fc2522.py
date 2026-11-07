import numpy as np
import time

# Cache for precomputed policies
_POLICY_CACHE = {}

def _get_policy(params):
    """
    Compute and cache the optimal policy table for a given set of parameters.
    Returns a tuple (bid_table, ask_table) where tables are of shape (T, 2*Q+1).
    bid_table[tau, q_idx] is the optimal bid depth.
    ask_table[tau, q_idx] is the optimal ask depth.
    """
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key in _POLICY_CACHE:
        return _POLICY_CACHE[key]
        
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    dt = T / steps
    # Inventory indices: 0 corresponds to -Q, 2*Q corresponds to Q
    n_states = 2 * Q + 1
    q_values = np.arange(-Q, Q + 1)
    
    # V_prev[q_idx] holds V(q, tau-1)
    V_prev = np.zeros(n_states)
    
    # We store the optimal actions for each tau and q
    # Note: quote is called at t = n*dt. 
    # The decision is made at t, affecting the transition to t+dt.
    # So we need the policy for tau = T - t.
    # At t=0, tau=T. At t=T, tau=0.
    # The DP computes V(q, tau) based on V(q, tau-1).
    # We need to store the action taken when the remaining time is tau.
    
    bid_table = np.zeros((steps, n_states))
    ask_table = np.zeros((steps, n_states))
    
    # We iterate tau from 1 to steps
    # V(q, tau) is computed using V(q, tau-1)
    # The action chosen at time t (where remaining time is tau) is based on V(q, tau-1).
    
    for tau in range(1, steps + 1):
        V_curr = np.zeros(n_states)
        for i, q in enumerate(q_values):
            # Penalty term
            penalty = -0.5 * gamma * sigma**2 * q**2 * dt
            
            # Get V for neighbors at tau-1
            V_q = V_prev[i]
            
            # Determine valid neighbors
            # Bid hit: q -> q-1. Valid if q < Q.
            # Ask hit: q -> q+1. Valid if q > -Q.
            
            # Approximation for optimal bid depth b
            # b* = max(0, (V(q-1, tau-1) - V(q, tau-1) + 1/k) / k)
            # Note: If q == Q, no bid fill allowed. So b is effectively inf (or we just don't care, but we must return inf).
            # The formula assumes we can choose b. If q==Q, we must set b=inf.
            
            if q < Q:
                V_qm1 = V_prev[i-1]
                delta_minus = V_qm1 - V_q
                b_opt = (delta_minus + 1.0/k) / k
                if b_opt < 0:
                    b_opt = np.inf
                else:
                    b_opt = float(b_opt)
            else:
                b_opt = np.inf
                
            # Approximation for optimal ask depth a
            # a* = max(0, (1/k - (V(q+1, tau-1) - V(q, tau-1))) / k)
            if q > -Q:
                V_qp1 = V_prev[i+1]
                delta_plus = V_qp1 - V_q
                a_opt = (1.0/k - delta_plus) / k
                if a_opt < 0:
                    a_opt = np.inf
                else:
                    a_opt = float(a_opt)
            else:
                a_opt = np.inf
                
            # We need to calculate the value V_curr[i] using these optimal b and a.
            # Although we used the approximation to FIND b and a, we should ideally plug them back into the exact expectation
            # or use the approximated value. For consistency in the DP recursion, using the approximated value function
            # derived from the first-order expansion is standard and stable.
            
            # Let's use the first-order approximation for the value update as well to be consistent with the action derivation.
            # Value contribution from bid: p_b * (b_opt - V_q + V_qm1)  (Wait, reward is -b, next V is V_qm1)
            # Reward = -b. Next Value = V_qm1.
            # Term = p_b * (-b + V_qm1 - V_q)
            
            # Value contribution from ask: p_a * (a_opt + V_qp1 - V_q)
            # Reward = +a. Next Value = V_qp1.
            # Term = p_a * (a + V_qp1 - V_q)
            
            # Base term: (1 - p_b - p_a) * V_q  (Approximating p_b p_a as 0)
            # Actually, the exact term is V_q + p_b(V_qm1 - V_q - b) + p_a(V_qp1 - V_q + a) - p_b p_a (...)
            # With p_b p_a ~ 0:
            # V_curr = V_q + p_b(V_qm1 - V_q - b_opt) + p_a(V_qp1 - V_q + a_opt)
            
            p_b = 0.0
            p_a = 0.0
            
            if q < Q and not np.isinf(b_opt):
                # p = 1 - exp(-A exp(-k b) dt)
                # For small dt, p ~ A exp(-k b) dt
                # Using exact formula is safer if b is small
                exponent = -A * np.exp(-k * b_opt) * dt
                p_b = 1.0 - np.exp(exponent)
            # else p_b = 0 (since b=inf => prob=0)
            
            if q > -Q and not np.isinf(a_opt):
                exponent = -A * np.exp(-k * a_opt) * dt
                p_a = 1.0 - np.exp(exponent)
            # else p_a = 0
            
            # Calculate V_curr
            val = V_q
            if q < Q:
                val += p_b * (V_prev[i-1] - V_q - b_opt)
            if q > -Q:
                val += p_a * (V_prev[i+1] - V_q + a_opt)
                
            V_curr[i] = penalty + val
            
            # Store actions for this tau (remaining time)
            # The quote function is called at t. 
            # t = n * dt. 
            # Remaining time tau = T - t = (steps - n) * dt.
            # So if we are at step n, we need the action computed for tau = steps - n.
            # In our loop, `tau` goes from 1 to steps.
            # We store it at index `tau-1` in the table? 
            # Let's define the table index such that table[tau_idx] corresponds to remaining time tau.
            # We will store bid_table[tau-1, i] = b_opt.
            
            bid_table[tau-1, i] = b_opt
            ask_table[tau-1, i] = a_opt
            
        V_prev = V_curr
        
    _POLICY_CACHE[key] = (bid_table, ask_table)
    return _POLICY_CACHE[key]


def quote(t, inventory, mid, params):
    """
    Returns (bid_depths, ask_depths) for the given state.
    """
    steps = params['steps']
    Q = params['Q']
    
    # Get the precomputed policy
    bid_table, ask_table = _get_policy(params)
    
    # Determine the remaining time index
    # t is the current time.
    # The step n is floor(t / dt). But t is passed as n*dt.
    # We need the policy for the decision made at this step.
    # The decision at step n uses the value function for remaining time tau = T - n*dt.
    # In our DP, we computed actions for tau = 1, ..., steps.
    # tau = steps - n.
    # Since t = n * dt, n = t / dt.
    # dt = T / steps.
    # n = t * steps / T.
    
    n = int(round(t * params['steps'] / params['T']))
    
    # Clamp n to [0, steps-1] to be safe, though t should be in [0, T-dt]
    if n < 0: n = 0
    if n >= steps: n = steps - 1
    
    tau_idx = steps - 1 - n
    
    # Map inventory to indices
    # inventory is an array of ints in [-Q, Q]
    q_idx = inventory + Q
    
    # Retrieve depths
    bid_depths = bid_table[tau_idx, q_idx]
    ask_depths = ask_table[tau_idx, q_idx]
    
    return bid_depths, ask_depths
