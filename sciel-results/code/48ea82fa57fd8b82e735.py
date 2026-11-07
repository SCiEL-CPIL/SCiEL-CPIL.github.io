"""
Optimal market making with inventory risk via exact Dynamic Programming.

The state space is (step, inventory) = (1000, 61). For each state, we solve
the Bellman equation to find the optimal bid and ask depths. The policy is
pre-computed per regime and cached.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for pre-computed optimal policies
# Key: (sigma, A, k, gamma, T, steps, Q)
# Value: (bid_depths, ask_depths) arrays of shape (steps, 2*Q+1)
_policy_cache = {}

# Deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 320.0  # 10s safety margin before 330s kill


def _compute_optimal_policy(params):
    """
    Compute the optimal bid and ask depths for all (t, q) states using DP.
    
    The objective per path is:
        sum of depths earned (from fills) - sum over steps of (gamma/2) * sigma^2 * q^2 * dt
    
    At each step, given inventory q:
    - Bid depth b: fill prob = 1 - exp(-A * exp(-k*b) * dt), earns b, inventory becomes q+1
    - Ask depth a: fill prob = 1 - exp(-A * exp(-k*a) * dt), earns a, inventory becomes q-1
    - Both can fill independently
    - No bid fill at q = Q, no ask fill at q = -Q
    - Mid moves by sigma*sqrt(dt)*N(0,1) but this doesn't affect the depth decision
      (depths are relative to mid, and the mid movement is additive to both sides equally
      in terms of the depth earned)
    
    Wait, let me re-read the model more carefully.
    
    "A fill earns its depth and moves inventory one unit"
    
    So if bid is hit, we earn bid_depth (the distance below mid) and inventory increases by 1.
    If ask is hit, we earn ask_depth (the distance above mid) and inventory decreases by 1.
    
    The mid moves by sigma*sqrt(dt)*N(0,1), but this is a random walk of the mid price.
    The depths are quoted relative to the current mid. The next step's mid will be different,
    but the depth we quote this step is relative to THIS step's mid.
    
    So the decision at step n depends only on (n, q), not on the actual mid value,
    because the depth earned is the distance from mid, and the mid movement is
    independent of our depth choice.
    
    This means the optimal policy is a function of (step, inventory) only.
    
    Bellman equation:
    V(n, q) = max_{b, a} { 
        (1 - exp(-A*exp(-k*b)*dt)) * [b + V(n+1, q+1)] 
        + (1 - exp(-A*exp(-k*a)*dt)) * [a + V(n+1, q-1)]
        - (1 - exp(-A*exp(-k*b)*dt)) * (1 - exp(-A*exp(-k*a)*dt)) * 0  # no double-count
        - (gamma/2) * sigma^2 * q^2 * dt
    }
    
    Wait, both fills can happen independently. Let me reconsider.
    
    If bid fills with prob p_b and ask fills with prob p_a, independently:
    - Both fill: prob p_b * p_a, earn b + a, inventory becomes q (q+1-1 = q)
    - Only bid: prob p_b * (1-p_a), earn b, inventory becomes q+1
    - Only ask: prob (1-p_b) * p_a, earn a, inventory becomes q-1
    - Neither: prob (1-p_b) * (1-p_a), earn 0, inventory stays q
    
    So expected value:
    E = p_b*(1-p_a) * [b + V(n+1, q+1)] 
      + (1-p_b)*p_a * [a + V(n+1, q-1)]
      + p_b*p_a * [b + a + V(n+1, q)]
      + (1-p_b)*(1-p_a) * V(n+1, q)
    
    Simplify:
    E = p_b * b + p_a * a 
      + p_b*(1-p_a) * V(n+1, q+1)
      + (1-p_b)*p_a * V(n+1, q-1)
      + (p_b*p_a + (1-p_b)*(1-p_a)) * V(n+1, q)
    
    And we subtract the holding cost: (gamma/2) * sigma^2 * q^2 * dt
    
    So:
    V(n, q) = max_{b, a} { 
        p_b * b + p_a * a 
        + p_b*(1-p_a) * V(n+1, q+1)
        + (1-p_b)*p_a * V(n+1, q-1)
        + (p_b*p_a + (1-p_b)*(1-p_a)) * V(n+1, q)
        - (gamma/2) * sigma^2 * q^2 * dt
    }
    
    where p_b = 1 - exp(-A*exp(-k*b)*dt) if q < Q, else 0
          p_a = 1 - exp(-A*exp(-k*a)*dt) if q > -Q, else 0
    
    At terminal step (n = steps), V(steps, q) = 0 (no terminal charge).
    
    We need to find optimal b and a for each (n, q). Since b and a appear in separate
    terms (p_b*b and p_a*a, and the V terms), we can potentially separate the optimization.
    
    Actually, looking at the expression:
    p_b * b + p_a * a 
    + p_b*(1-p_a) * V(n+1, q+1)
    + (1-p_b)*p_a * V(n+1, q-1)
    + (p_b*p_a + (1-p_b)*(1-p_a)) * V(n+1, q)
    
    Let me expand the V terms:
    = p_b * b + p_a * a 
    + p_b*V(n+1,q+1) - p_b*p_a*V(n+1,q+1)
    + p_a*V(n+1,q-1) - p_b*p_a*V(n+1,q-1)
    + p_b*p_a*V(n+1,q) + V(n+1,q) - p_b*V(n+1,q) - p_a*V(n+1,q) + p_b*p_a*V(n+1,q)
    
    Wait, let me redo this more carefully.
    
    p_b*p_a + (1-p_b)*(1-p_a) = p_b*p_a + 1 - p_b - p_a + p_b*p_a = 1 - p_b - p_a + 2*p_b*p_a
    
    So the V(n+1,q) term coefficient is: 1 - p_b - p_a + 2*p_b*p_a
    
    Full expression:
    E = p_b*b + p_a*a 
      + p_b*(1-p_a)*V(q+1) 
      + (1-p_b)*p_a*V(q-1)
      + (1 - p_b - p_a + 2*p_b*p_a)*V(q)
    
    where V(q±1) = V(n+1, q±1) and V(q) = V(n+1, q) for brevity.
    
    This is coupled in p_b and p_a through the p_b*p_a terms. So we can't fully separate.
    
    However, for a given p_a, we can optimize over p_b, and vice versa. Or we can do
    a 2D search. Since the depth space is continuous, we need to find the optimal
    depth analytically or via search.
    
    For a fixed p_a, the expression as a function of p_b is:
    p_b*b + p_b*(1-p_a)*V(q+1) + (1-p_b)*p_a*V(q-1) + (1-p_b-p_a+2*p_b*p_a)*V(q) + const
    
    But b is related to p_b: p_b = 1 - exp(-A*exp(-k*b)*dt), so b = -ln(-ln(1-p_b)/A)/k ... 
    Actually: p_b = 1 - exp(-A*exp(-k*b)*dt)
    => 1 - p_b = exp(-A*exp(-k*b)*dt)
    => ln(1-p_b) = -A*exp(-k*b)*dt
    => exp(-k*b) = -ln(1-p_b)/(A*dt)
    => -k*b = ln(-ln(1-p_b)/(A*dt))
    => b = -ln(-ln(1-p_b)/(A*dt))/k
    
    So b is a function of p_b. Similarly for a and p_a.
    
    This makes the optimization a 2D problem in (p_b, p_a) or equivalently (b, a).
    
    Given the small state space, I'll do a grid search over depths for each state.
    But that might be slow. Let me think of a better approach.
    
    Actually, since the coupling is only through the p_b*p_a term, and the optimal
    depths are likely to be in a reasonable range, I can use a coarse-to-fine grid
    search or use scipy's minimize.
    
    Given the time budget, let me use a grid search with a reasonable number of points.
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
    # V has shape (steps+1, n_states)
    V = np.zeros((steps + 1, n_states))
    bid_opt = np.zeros((steps, n_states))
    ask_opt = np.zeros((steps, n_states))
    
    # Grid of depths to search over
    # Depths should be positive. Let's use a logarithmic grid.
    # The risk-neutral depth is 1/k. Let's search from very small to a few times 1/k.
    depth_grid = np.geomspace(1e-4, 5.0 / k, 50)
    
    for n in range(steps - 1, -1, -1):
        for qi in range(n_states):
            q = qi - Q  # inventory level, from -Q to Q
            
            # Holding cost
            holding_cost = 0.5 * gamma * sigma * sigma * q * q * dt
            
            # V(n+1, q-1), V(n+1, q), V(n+1, q+1)
            V_qm1 = V[n + 1, max(0, qi - 1)] if qi > 0 else V[n + 1, 0]
            V_q = V[n + 1, qi]
            V_qp1 = V[n + 1, min(n_states - 1, qi + 1)] if qi < n_states - 1 else V[n + 1, n_states - 1]
            
            # Check if bid/ask fills are allowed
            can_bid = q < Q
            can_ask = q > -Q
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            if not can_bid and not can_ask:
                V[n, qi] = -holding_cost
                bid_opt[n, qi] = 0.0
                ask_opt[n, qi] = 0.0
                continue
            
            # Search over depth grid
            for b in depth_grid:
                if can_bid:
                    p_b = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                else:
                    p_b = 0.0
                
                for a in depth_grid:
                    if can_ask:
                        p_a = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    else:
                        p_a = 0.0
                    
                    # Expected value
                    val = p_b * b + p_a * a
                    val += p_b * (1.0 - p_a) * V_qp1
                    val += (1.0 - p_b) * p_a * V_qm1
                    coeff_q = 1.0 - p_b - p_a + 2.0 * p_b * p_a
                    val += coeff_q * V_q
                    val -= holding_cost
                    
                    if val > best_val:
                        best_val = val
                        best_b = b
                        best_a = a
            
            V[n, qi] = best_val
            bid_opt[n, qi] = best_b
            ask_opt[n, qi] = best_a
    
    return bid_opt, ask_opt


def _get_policy(params):
    """Get or compute the optimal policy for the given regime."""
    key = (params['sigma'], params['A'], params['k'], params['gamma'], 
           params['T'], params['steps'], params['Q'])
    
    if key not in _policy_cache:
        if time.time() > _DEADLINE - 5.0:
            # Not enough time to compute; return a simple heuristic
            Q = params['Q']
            k = params['k']
            n_states = 2 * Q + 1
            steps = params['steps']
            depth = 1.0 / k
            bid_opt = np.full((steps, n_states), depth)
            ask_opt = np.full((steps, n_states), depth)
            _policy_cache[key] = (bid_opt, ask_opt)
        else:
            bid_opt, ask_opt = _compute_optimal_policy(params)
            _policy_cache[key] = (bid_opt, ask_opt)
    
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """
    Return optimal bid and ask depths for each path.
    """
    Q = params['Q']
    steps = params['steps']
    
    # Get step index from t
    dt = params['T'] / steps
    n = int(round(t / dt))
    n = min(n, steps - 1)  # clamp to valid range
    
    # Get policy
    bid_opt, ask_opt = _get_policy(params)
    
    # Map inventory to index: inventory in [-Q, Q] -> index in [0, 2Q]
    inv_idx = np.clip(inventory + Q, 0, 2 * Q)
    
    # Look up depths
    bid_depths = bid_opt[n, inv_idx]
    ask_depths = ask_opt[n, inv_idx]
    
    # Ensure valid depths (positive, finite)
    bid_depths = np.maximum(bid_depths, 1e-6)
    ask_depths = np.maximum(ask_depths, 1e-6)
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
