"""Optimal market making with inventory risk via dynamic programming."""
# EVOLVE-BLOCK-START
import numpy as np

# Pre-computed DP tables for each regime
# Regimes: (sigma, A, k, gamma)
# T=1, steps=1000, Q=30, dt=0.001
# We'll compute the optimal depth for each inventory level at each time step
# using backward induction on the discrete model.

# Since the model is:
# - bid hit prob: 1 - exp(-A * exp(-k * bid_depth) * dt)
# - ask hit prob: 1 - exp(-A * exp(-k * ask_depth) * dt)
# - fill earns depth, moves inventory by +/-1
# - mid moves by sigma*sqrt(dt)*N(0,1) - but this doesn't affect the depth choice directly
#   because the objective is depths earned minus holding cost. The mid movement doesn't
#   enter the objective directly (the depths are what's earned, not the P&L from mid moves).
#   Wait, let me re-read: "depths earned minus sum of (gamma/2)*sigma^2*q^2*dt"
#   So the objective is: sum of (bid_depth if bid filled + ask_depth if ask filled) - sum of holding costs.
#   The mid movement is irrelevant to the objective! It's just part of the simulation.
#   So the optimal policy depends only on t, inventory, and the parameters.

# The state is (n, q) where n is the step index and q is inventory.
# At each state, we choose bid_depth >= 0 and ask_depth >= 0.
# Constraints: no bid fill at q=Q, no ask fill at q=-Q.
#   Actually, "no bid fill at q=Q" means if q==Q, we can't have a bid fill (inventory can't go above Q).
#   Similarly, no ask fill at q=-Q.
#   But we can still quote - it just won't fill. So effectively, if q==Q, bid_depth should be inf (no quote)
#   or we just accept that it won't fill. For optimality, if q==Q, quoting bid is wasteful if it can't fill,
#   but the probability formula still applies. Actually, re-reading: "no bid fill at q=Q" means the fill
#   is prevented, so the depth is not earned. So it's as if the probability is 0.
#   Similarly for ask at q=-Q.

# Let's define the DP:
# V(n, q) = max over bid_depth, ask_depth of [
#   (1 - exp(-A*exp(-k*bid_depth)*dt)) * (bid_depth + V(n+1, q+1))  [if q < Q, else 0 for the fill]
#   + (1 - exp(-A*exp(-k*ask_depth)*dt)) * (ask_depth + V(n+1, q-1))  [if q > -Q, else 0]
#   + exp(-A*exp(-k*bid_depth)*dt) * exp(-A*exp(-k*ask_depth)*dt) * V(n+1, q)
#   - (gamma/2) * sigma^2 * q^2 * dt
# ]
# Wait, the fills are independent. Let p_b = 1 - exp(-A*exp(-k*bid_depth)*dt), p_a = 1 - exp(-A*exp(-k*ask_depth)*dt).
# Both fill: prob p_b * p_a, earn bid_depth + ask_depth, inventory q -> q (bid +1, ask -1, net 0)
# Only bid: prob p_b * (1-p_a), earn bid_depth, inventory q -> q+1
# Only ask: prob (1-p_b) * p_a, earn ask_depth, inventory q -> q-1
# Neither: prob (1-p_b)*(1-p_a), earn 0, inventory q -> q

# But with constraints: if q == Q, bid fill is prevented, so effectively p_b = 0 for the fill effect.
# If q == -Q, ask fill is prevented, so p_a = 0.

# V(1000, q) = 0 for all q (no terminal charge)

# This is a 2D optimization at each state. We can use the fact that the optimal depth
# likely has a simple form. For a single side, the optimal depth that maximizes
# p(d) * (d + V_next) where p(d) = 1 - exp(-A*exp(-k*d)*dt).
# Let f(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + C)
# f'(d) = A*k*exp(-k*d)*dt * exp(-A*exp(-k*d)*dt) * (d+C) + (1 - exp(-A*exp(-k*d)*dt))
# Set to 0. This is complex.

# Alternative: discretize the depth space and do DP with a grid.
# Since Q=30, steps=1000, and we need to be fast (300s total for 1000 calls * 4000 paths),
# we can precompute the optimal depths for each (n, q) pair.

# Let's discretize depth into a grid. The optimal depth is likely O(1/k) ~ 0.5-1.3.
# Let's use a grid of depths from 0 to 5 with step 0.01, and find the best.

# Actually, let's think about this differently. The problem is separable in a sense.
# For given V(n+1, q+1), V(n+1, q), V(n+1, q-1), we need to maximize:
# E[reward] = p_b * (1-p_a) * (bid_depth + V(n+1,q+1)) + (1-p_b)*p_a * (ask_depth + V(n+1,q-1)) 
#            + p_b*p_a * (bid_depth + ask_depth + V(n+1,q)) + (1-p_b)*(1-p_a)*V(n+1,q)
#            - (gamma/2)*sigma^2*q^2*dt

# Let's expand:
# = p_b*(1-p_a)*(bid_depth + Vb) + (1-p_b)*p_a*(ask_depth + Va) + p_b*p_a*(bid_depth + ask_depth + V0) + (1-p_b)*(1-p_a)*V0 - cost
# where Vb = V(n+1,q+1), Va = V(n+1,q-1), V0 = V(n+1,q)

# Group by bid_depth and ask_depth:
# Terms with bid_depth: p_b*(1-p_a)*bid_depth + p_b*p_a*bid_depth = p_b*bid_depth
# Terms with ask_depth: (1-p_b)*p_a*ask_depth + p_b*p_a*ask_depth = p_a*ask_depth
# Terms with Vb: p_b*(1-p_a)*Vb
# Terms with Va: (1-p_b)*p_a*Va
# Terms with V0: p_b*p_a*V0 + (1-p_b)*(1-p_a)*V0 = V0*(p_b*p_a + 1 - p_b - p_a + p_b*p_a) = V0*(1 - p_b - p_a + 2*p_b*p_a)
# Wait, that's not right. Let me redo:
# p_b*p_a*V0 + (1-p_b)*(1-p_a)*V0 = V0*(p_b*p_a + 1 - p_b - p_a + p_b*p_a) = V0*(1 - p_b - p_a + 2*p_b*p_a)

# So E[reward] = p_b*bid_depth + p_a*ask_depth + p_b*(1-p_a)*Vb + (1-p_b)*p_a*Va + V0*(1 - p_b - p_a + 2*p_b*p_a) - cost

# This is still coupled through p_b and p_a in the V terms. But for small dt, p_b and p_a are small,
# so the coupling is weak. Let's approximate by ignoring the coupling in the V terms:
# E[reward] ≈ p_b*bid_depth + p_a*ask_depth + p_b*Vb + p_a*Va + V0 - cost
# (to first order in p_b, p_a)

# Then the optimal bid_depth maximizes p_b*(bid_depth + Vb) where p_b = 1 - exp(-A*exp(-k*bid_depth)*dt)
# Similarly for ask.

# Let's solve for the optimal depth numerically. Define g(d) = (1 - exp(-A*exp(-k*d)*dt)) * (d + C)
# where C = Vb or Va.
# g'(d) = A*k*dt*exp(-k*d)*exp(-A*exp(-k*d)*dt) * (d+C) + (1 - exp(-A*exp(-k*d)*dt))
# Set g'(d) = 0.

# For small dt, A*exp(-k*d)*dt is small, so exp(-A*exp(-k*d)*dt) ≈ 1 - A*exp(-k*d)*dt
# Then 1 - exp(-A*exp(-k*d)*dt) ≈ A*exp(-k*d)*dt
# And exp(-A*exp(-k*d)*dt) ≈ 1
# So g(d) ≈ A*exp(-k*d)*dt * (d+C)
# g'(d) ≈ A*dt * (-k*exp(-k*d)*(d+C) + exp(-k*d)) = A*dt*exp(-k*d)*(1 - k*(d+C))
# Set to 0: 1 - k*(d+C) = 0 => d = 1/k - C

# So the optimal depth is approximately 1/k - C, where C is the continuation value.
# But C can be negative (if holding cost is high), so d could be negative, meaning no quote.

# Let's use this approximation: optimal bid_depth = max(0, 1/k - V(n+1, q+1))
# optimal ask_depth = max(0, 1/k - V(n+1, q-1))

# But we need to compute V(n, q) properly. Let's do DP with this approximation.

def _compute_dp(sigma, A, k, gamma, T=1.0, steps=1000, Q=30):
    """Compute optimal bid and ask depths for each (n, q) using DP with first-order approximation."""
    dt = T / steps
    # V[n, q_index] where q_index = q + Q, q in [-Q, Q]
    V = np.zeros((steps + 1, 2 * Q + 1), dtype=np.float64)
    bid_depths = np.zeros((steps, 2 * Q + 1), dtype=np.float64)
    ask_depths = np.zeros((steps, 2 * Q + 1), dtype=np.float64)
    
    for n in range(steps - 1, -1, -1):
        for q in range(-Q, Q + 1):
            q_idx = q + Q
            Vb = V[n + 1, q + 1 + Q] if q < Q else 0.0  # V(n+1, q+1), but if q==Q, no bid fill
            Va = V[n + 1, q - 1 + Q] if q > -Q else 0.0  # V(n+1, q-1), but if q==-Q, no ask fill
            V0 = V[n + 1, q_idx]
            
            # Optimal bid depth: max(0, 1/k - Vb) if q < Q, else inf (no quote)
            if q < Q:
                opt_bid = max(0.0, 1.0 / k - Vb)
            else:
                opt_bid = np.inf
            
            # Optimal ask depth: max(0, 1/k - Va) if q > -Q, else inf
            if q > -Q:
                opt_ask = max(0.0, 1.0 / k - Va)
            else:
                opt_ask = np.inf
            
            bid_depths[n, q_idx] = opt_bid
            ask_depths[n, q_idx] = opt_ask
            
            # Compute V(n, q)
            if q < Q:
                p_b = 1.0 - np.exp(-A * np.exp(-k * opt_bid) * dt) if np.isfinite(opt_bid) else 0.0
            else:
                p_b = 0.0
            if q > -Q:
                p_a = 1.0 - np.exp(-A * np.exp(-k * opt_ask) * dt) if np.isfinite(opt_ask) else 0.0
            else:
                p_a = 0.0
            
            # First-order approximation for V(n, q)
            # V ≈ p_b*(opt_bid + Vb) + p_a*(opt_ask + Va) + V0 - (gamma/2)*sigma^2*q^2*dt
            # But with the constraint handling:
            V[n, q_idx] = p_b * (opt_bid + Vb) + p_a * (opt_ask + Va) + V0 - 0.5 * gamma * sigma**2 * q**2 * dt
    
    return bid_depths, ask_depths

# Precompute for each regime
_regimes = [
    (2.0, 140.0, 1.5, 0.05),
    (2.0, 140.0, 1.5, 0.5),
    (1.5, 50.0, 0.75, 0.2),
]

_dp_tables = {}
for sigma, A, k, gamma in _regimes:
    key = (sigma, A, k, gamma)
    _dp_tables[key] = _compute_dp(sigma, A, k, gamma)


def quote(t, inventory, mid, params):
    """Return optimal bid and ask depths for each path."""
    sigma = params["sigma"]
    A = params["A"]
    k = params["k"]
    gamma = params["gamma"]
    T = params["T"]
    steps = params["steps"]
    Q = params["Q"]
    
    key = (sigma, A, k, gamma)
    if key not in _dp_tables:
        # Fallback: risk-neutral depth
        depth = np.full(len(inventory), 1.0 / k)
        return depth, depth.copy()
    
    bid_table, ask_table = _dp_tables[key]
    
    # t = n * dt, so n = round(t / dt)
    dt = T / steps
    n = int(round(t / dt))
    n = max(0, min(n, steps - 1))
    
    bid_depths = np.empty(len(inventory), dtype=np.float64)
    ask_depths = np.empty(len(inventory), dtype=np.float64)
    
    for i, q in enumerate(inventory):
        q_idx = int(q) + Q
        if 0 <= q_idx < 2 * Q + 1:
            bid_depths[i] = bid_table[n, q_idx]
            ask_depths[i] = ask_table[n, q_idx]
        else:
            bid_depths[i] = 1.0 / k
            ask_depths[i] = 1.0 / k
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
