# EVOLVE-BLOCK-START
import numpy as np
import time
import math

# Precompute tables for fast evaluation
# We'll compute the optimal policy using dynamic programming for each regime
# The state is (inventory, time_step). Since mid is a random walk independent of inventory,
# and the objective depends on mid only through the depth earned (which is the distance from mid),
# the optimal depth at each step depends only on inventory and remaining time (or current step).
# Actually, let's think more carefully.

# The model:
# - At each step n, we have inventory q_n (before fills), mid m_n
# - We quote bid_depth b_n and ask_depth a_n (distances below/above mid)
# - Bid hit prob: 1 - exp(-A * exp(-k * b_n) * dt)
# - Ask hit prob: 1 - exp(-A * exp(-k * a_n) * dt)
# - If bid hit: earn b_n, inventory increases by 1 (q_{n+1} = q_n + 1)
# - If ask hit: earn a_n, inventory decreases by 1 (q_{n+1} = q_n - 1)
# - No bid fill at q = Q, no ask fill at q = -Q
# - Mid moves: m_{n+1} = m_n + sigma * sqrt(dt) * Z, Z ~ N(0,1)
# - Objective: sum of depths earned - sum over steps of (gamma/2) * sigma^2 * q_n^2 * dt

# Key insight: The mid movement is independent of inventory and doesn't affect the fill probabilities
# (fill probs depend only on depth, not on mid level). The depth earned is just the depth value itself.
# So the optimal policy for depths depends only on (n, q), not on the actual mid value.
# We can precompute the optimal bid_depth and ask_depth for each (n, q) pair.

# State space: n in [0, 999], q in [-30, 30]
# That's 1000 * 61 = 61000 states. Very manageable.

# For each state (n, q), we choose b >= 0 and a >= 0 to maximize:
#   b * P_bid + a * P_ask + E[V(n+1, q')] - (gamma/2) * sigma^2 * q^2 * dt
# where:
#   P_bid = 1 - exp(-A * exp(-k*b) * dt) if q < Q else 0
#   P_ask = 1 - exp(-A * exp(-k*a) * dt) if q > -Q else 0
#   q' = q + 1 if bid hit, q - 1 if ask hit, q if no hit
#   (bid and ask hits are independent)

# So:
#   E[V(n+1, q')] = P_bid * P_ask * V(n+1, q) + P_bid * (1-P_ask) * V(n+1, q+1) 
#                  + (1-P_bid) * P_ask * V(n+1, q-1) + (1-P_bid)*(1-P_ask) * V(n+1, q)
#   = P_bid * (P_ask * V(n+1, q) + (1-P_ask) * V(n+1, q+1))
#     + (1-P_bid) * (P_ask * V(n+1, q-1) + (1-P_ask) * V(n+1, q))

# And the immediate reward is: b * P_bid + a * P_ask - (gamma/2) * sigma^2 * q^2 * dt

# We need to optimize over b and a. Since b and a appear in separate terms (b only in P_bid and b*P_bid,
# a only in P_ask and a*P_ask), we can optimize them separately!

# For bid: maximize over b >= 0:
#   f(b) = b * P_bid(b) + P_bid(b) * (P_ask * V(n+1, q) + (1-P_ask) * V(n+1, q+1))
#        - (1-P_bid(b)) * (P_ask * V(n+1, q-1) + (1-P_ask) * V(n+1, q))
#   where P_bid(b) = 1 - exp(-A * exp(-k*b) * dt)

# Let's define:
#   Vq = V(n+1, q), Vq1 = V(n+1, q+1), Vqm1 = V(n+1, q-1)
#   The term involving P_bid:
#   P_bid * [b + Vq + Vq1 - Vq] ... let me redo this.

# E[V(n+1, q')] = P_bid * P_ask * Vq + P_bid * (1-P_ask) * Vq1 + (1-P_bid) * P_ask * Vqm1 + (1-P_bid)*(1-P_ask) * Vq
# = P_bid * [P_ask * Vq + (1-P_ask) * Vq1] + (1-P_bid) * [P_ask * Vqm1 + (1-P_ask) * Vq]
# = P_bid * [P_ask * Vq + Vq1 - P_ask * Vq1] + (1-P_bid) * [P_ask * Vqm1 + Vq - P_ask * Vq]
# = P_bid * [Vq1 - P_ask * (Vq1 - Vq)] + (1-P_bid) * [Vq - P_ask * (Vq - Vqm1)]

# Let me just expand:
# = P_bid * P_ask * Vq + P_bid * Vq1 - P_bid * P_ask * Vq1 + P_ask * Vqm1 - P_ask * P_bid * Vqm1 + Vq - P_ask * Vq - P_bid * Vq + P_bid * P_ask * Vq
# This is getting messy. Let me just use the original form.

# Objective for state (n,q):
# R(b,a) = b*Pb + a*Pa + Pb*Pa*Vq + Pb*(1-Pa)*Vq1 + (1-Pb)*Pa*Vqm1 + (1-Pb)*(1-Pa)*Vq - (gamma/2)*sigma^2*q^2*dt

# = b*Pb + a*Pa + Pb*(Pa*Vq + (1-Pa)*Vq1) + (1-Pb)*(Pa*Vqm1 + (1-Pa)*Vq) - cost
# = b*Pb + Pb*(Pa*Vq + (1-Pa)*Vq1 - Pa*Vqm1 - (1-Pa)*Vq) + (Pa*Vqm1 + (1-Pa)*Vq) + a*Pa - cost
# = b*Pb + Pb*(Pa*(Vq - Vqm1) + (1-Pa)*(Vq1 - Vq)) + (Pa*Vqm1 + (1-Pa)*Vq) + a*Pa - cost

# So the terms involving Pb (and thus b) are:
#   b*Pb + Pb*(Pa*(Vq - Vqm1) + (1-Pa)*(Vq1 - Vq))
# And the terms involving Pa (and thus a) are:
#   a*Pa + (Pa*Vqm1 + (1-Pa)*Vq)
# Plus the constant: -cost

# Wait, I think I made an error. Let me re-derive more carefully.

# R = b*Pb + a*Pa + Pb*Pa*Vq + Pb*(1-Pa)*Vq1 + (1-Pb)*Pa*Vqm1 + (1-Pb)*(1-Pa)*Vq - cost

# Group by Pb:
# = Pb * [b + Pa*Vq + (1-Pa)*Vq1] + (1-Pb) * [Pa*Vqm1 + (1-Pa)*Vq] + a*Pa - cost
# = Pb * [b + Pa*Vq + (1-Pa)*Vq1 - Pa*Vqm1 - (1-Pa)*Vq] + [Pa*Vqm1 + (1-Pa)*Vq] + a*Pa - cost
# = Pb * [b + Pa*(Vq - Vqm1) + (1-Pa)*(Vq1 - Vq)] + [Pa*Vqm1 + (1-Pa)*Vq] + a*Pa - cost

# So for fixed Pa, the optimal b maximizes:
#   Pb * [b + C1] where C1 = Pa*(Vq - Vqm1) + (1-Pa)*(Vq1 - Vq)
# And Pb = 1 - exp(-A*exp(-k*b)*dt)

# Similarly for fixed Pb, the optimal a maximizes:
#   a*Pa + [Pa*Vqm1 + (1-Pa)*Vq]
# But wait, the second bracket doesn't depend on a, so we just maximize a*Pa + constant.
# That means we want to maximize a*Pa where Pa = 1 - exp(-A*exp(-k*a)*dt).

# Hmm, but b and a are coupled through the fact that Pb appears in C1 and Pa appears in the a-term.
# So we can't fully separate them. But we can use value iteration: for a given (Vq, Vq1, Vqm1),
# we can find the optimal (b, a) by solving a 2D optimization problem.

# Since the state space is small (61000 states), and for each state we need to solve a 2D optimization,
# we can precompute this offline and store the optimal depths in a table.

# Let me implement this with a coarse grid search for b and a, then refine.

def _precompute_policy(params):
    """Precompute optimal bid and ask depths for each (step, inventory) state."""
    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    dt = T / steps
    
    # V[n, q] for n in [0, steps], q in [-Q, Q]
    # V[steps, q] = 0 for all q (no terminal charge)
    V = np.zeros((steps + 1, 2 * Q + 1))
    bid_depth = np.zeros((steps, 2 * Q + 1))
    ask_depth = np.zeros((steps, 2 * Q + 1))
    
    # Grid search for b and a
    # b and a are distances. Reasonable range: [0, 1/k] or so.
    # 1/k for k=1.5 is ~0.67, for k=0.75 is ~1.33
    max_depth = 2.0 / k
    n_grid = 50
    b_grid = np.linspace(0, max_depth, n_grid)
    a_grid = np.linspace(0, max_depth, n_grid)
    
    for n in range(steps - 1, -1, -1):
        for qi in range(2 * Q + 1):
            q = qi - Q
            Vq = V[n + 1, qi]
            Vq1 = V[n + 1, min(qi + 1, 2 * Q)] if q < Q else 0.0
            Vqm1 = V[n + 1, max(qi - 1, 0)] if q > -Q else 0.0
            
            best_val = -np.inf
            best_b = 0.0
            best_a = 0.0
            
            for b in b_grid:
                if q >= Q:
                    Pb = 0.0
                else:
                    Pb = 1.0 - np.exp(-A * np.exp(-k * b) * dt)
                
                for a in a_grid:
                    if q <= -Q:
                        Pa = 0.0
                    else:
                        Pa = 1.0 - np.exp(-A * np.exp(-k * a) * dt)
                    
                    # R = b*Pb + a*Pa + Pb*Pa*Vq + Pb*(1-Pa)*Vq1 + (1-Pb)*Pa*Vqm1 + (1-Pb)*(1-Pa)*Vq
                    R = b * Pb + a * Pa + Pb * Pa * Vq + Pb * (1 - Pa) * Vq1 \
                        + (1 - Pb) * Pa * Vqm1 + (1 - Pb) * (1 - Pa) * Vq \
                        - (gamma / 2.0) * sigma * sigma * q * q * dt
                    
                    if R > best_val:
                        best_val = R
                        best_b = b
                        best_a = a
            
            V[n, qi] = best_val
            bid_depth[n, qi] = best_b
            ask_depth[n, qi] = best_a
    
    return bid_depth, ask_depth


# Cache precomputed policies
_policy_cache = {}

def _get_policy(params):
    key = (params['sigma'], params['A'], params['k'], params['gamma'], params['T'], params['steps'], params['Q'])
    if key not in _policy_cache:
        _policy_cache[key] = _precompute_policy(params)
    return _policy_cache[key]


def quote(t, inventory, mid, params):
    """Quote using precomputed optimal policy based on (step, inventory)."""
    start_time = time.time()
    
    n = int(round(t / (params['T'] / params['steps'])))
    n = min(n, params['steps'] - 1)
    Q = params['Q']
    
    bid_depth, ask_depth = _get_policy(params)
    
    # inventory is in [-Q, Q], map to index
    qi = np.clip(inventory + Q, 0, 2 * Q)
    
    b = bid_depth[n, qi]
    a = ask_depth[n, qi]
    
    # Ensure valid values
    b = np.maximum(b, 0.0)
    a = np.maximum(a, 0.0)
    
    # Check deadline
    if time.time() - start_time > 250:
        # Fallback: simple heuristic
        k = params['k']
        b = np.full(len(inventory), 1.0 / k)
        a = np.full(len(inventory), 1.0 / k)
    
    return b, a
# EVOLVE-BLOCK-END
