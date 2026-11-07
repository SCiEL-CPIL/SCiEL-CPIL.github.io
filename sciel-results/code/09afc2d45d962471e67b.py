"""Market-making with inventory risk: Analytical LQ Control + Vectorized Grid Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_T0 = time.time()
_DEADLINE = 295.0  # Safety margin before 300s limit

# Cache for optimized parameters per regime
# Key: (sigma, A, k, gamma, T, steps, Q)
_OPT_CACHE = {}

def _optimize_regime(sigma, A, k, gamma, T, steps, Q):
    """
    Pre-compute the optimal depth parameters for a given regime.
    We assume the optimal policy is affine in inventory:
    bid_depth = d_bid + m_bid * q
    ask_depth = d_ask + m_ask * q
    
    Due to symmetry, we expect d_bid ≈ d_ask = d0 and m_bid ≈ -m_ask = -m.
    So:
    bid_depth = d0 + m*q
    ask_depth = d0 - m*q
    
    We search for (d0, m) that maximizes the expected objective.
    To be fast, we use a vectorized grid search.
    """
    key = (sigma, A, k, gamma, T, steps, Q)
    if key in _OPT_CACHE:
        return _OPT_CACHE[key]

    dt = T / steps
    
    # Define grid for d0 and m
    # d0 is the base depth. Optimal is usually around 1/k.
    # Range: 0.5/k to 3.0/k
    # m is the inventory skew.
    # Range: -0.2 to 0.2 (relative to 1/k scale? No, absolute)
    # Let's check magnitudes.
    # If q=30, m=0.01, shift=0.3. 1/k is approx 0.66 (k=1.5).
    # So m should be small.
    
    d0_grid = np.linspace(0.5 / k, 2.5 / k, 40)
    m_grid = np.linspace(-0.05, 0.05, 40)
    
    # Create meshgrid
    D0, M = np.meshgrid(d0_grid, m_grid, indexing='ij')
    
    # We need to evaluate the expected value of this policy.
    # Since the problem is stochastic, we can't just plug in numbers.
    # However, for a fixed policy, we can simulate the expected trajectory.
    # But doing a full 1000-step simulation for 1600 grid points is too slow if done naively.
    
    # Alternative: Use the Bellman equation structure or a shorter horizon approximation 
    # that is more accurate than the previous one.
    # Or, since the dynamics are linear in probabilities (which are non-linear in depth), 
    # we can approximate the expected inventory change.
    
    # Let's use a "Rollout" approach with a longer horizon (H=20) but vectorized over the grid.
    # We simulate the *mean* inventory evolution for the population? 
    # No, the policy depends on the current inventory q.
    # The value function V(t, q) is quadratic: V(t, q) = -0.5 * gamma * sigma^2 * q^2 * (T-t) + ...?
    # Actually, the terminal cost is 0, but the running cost is quadratic.
    # The optimal control for LQ systems is linear feedback.
    
    # Let's refine the grid search to be faster by using a coarser grid first, 
    # then a finer grid around the best point.
    
    # Coarse Grid
    d0_coarse = np.linspace(0.5 / k, 2.5 / k, 10)
    m_coarse = np.linspace(-0.05, 0.05, 10)
    
    best_score = -np.inf
    best_d0 = 1.0 / k
    best_m = 0.0
    
    # We will evaluate the policy on a representative set of initial inventories.
    # Since the policy is applied to 4000 paths, and the cost is convex in q,
    # evaluating on the mean and variance is a good approximation for the *average* objective.
    # However, the reward from fills is non-linear in depth.
    
    # Let's simulate H=10 steps for a few representative q values: -Q, -Q/2, 0, Q/2, Q.
    # This captures the boundary effects and the skew effect.
    rep_qs = np.array([-Q, -Q/2, 0, Q/2, Q])
    
    H = 10
    
    for d0 in d0_coarse:
        for m in m_coarse:
            # Simulate H steps for representative qs
            q_sim = rep_qs.copy().astype(float)
            total_reward = 0.0
            
            for h in range(H):
                bid_d = d0 + m * q_sim
                ask_d = d0 - m * q_sim
                
                # Clip depths
                bid_d = np.clip(bid_d, 0.0, 5.0 / k)
                ask_d = np.clip(ask_d, 0.0, 5.0 / k)
                
                # Probabilities
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                
                # Expected reward
                # Note: We must handle the "no fill at boundary" constraint.
                # If q == Q, no bid fill. If q == -Q, no ask fill.
                # In our simulation, if q_sim reaches Q, p_bid should be 0.
                # But q_sim is just an estimate. 
                # For the purpose of parameter selection, the boundary constraint 
                # mainly means we shouldn't quote bid if q=Q.
                # Our formula bid_d = d0 + m*Q might be large or small.
                # If we quote, and it fills, we get reward. But model says no fill.
                # So we set p_bid = 0 if q_sim >= Q - 0.5?
                # Actually, the model says "no bid fill at q=Q".
                # So if q_sim is close to Q, the effective p_bid is 0.
                
                # Mask for boundaries
                mask_bid = q_sim < Q
                mask_ask = q_sim > -Q
                
                p_bid_eff = np.where(mask_bid, p_bid, 0.0)
                p_ask_eff = np.where(mask_ask, p_ask, 0.0)
                
                step_reward = np.mean(p_bid_eff * bid_d + p_ask_eff * ask_d - 0.5 * gamma * sigma**2 * q_sim**2 * dt)
                total_reward += step_reward
                
                # Update inventory
                q_sim = q_sim + p_ask_eff - p_bid_eff
                q_sim = np.clip(q_sim, -Q, Q)
            
            if total_reward > best_score:
                best_score = total_reward
                best_d0 = d0
                best_m = m

    # Fine Grid around best coarse point
    d0_fine = np.linspace(best_d0 - 0.2/k, best_d0 + 0.2/k, 20)
    m_fine = np.linspace(best_m - 0.01, best_m + 0.01, 20)
    
    for d0 in d0_fine:
        for m in m_fine:
            q_sim = rep_qs.copy().astype(float)
            total_reward = 0.0
            
            for h in range(H):
                bid_d = d0 + m * q_sim
                ask_d = d0 - m * q_sim
                
                bid_d = np.clip(bid_d, 0.0, 5.0 / k)
                ask_d = np.clip(ask_d, 0.0, 5.0 / k)
                
                p_bid = 1.0 - np.exp(-A * np.exp(-k * bid_d) * dt)
                p_ask = 1.0 - np.exp(-A * np.exp(-k * ask_d) * dt)
                
                mask_bid = q_sim < Q
                mask_ask = q_sim > -Q
                
                p_bid_eff = np.where(mask_bid, p_bid, 0.0)
                p_ask_eff = np.where(mask_ask, p_ask, 0.0)
                
                step_reward = np.mean(p_bid_eff * bid_d + p_ask_eff * ask_d - 0.5 * gamma * sigma**2 * q_sim**2 * dt)
                total_reward += step_reward
                
                q_sim = q_sim + p_ask_eff - p_bid_eff
                q_sim = np.clip(q_sim, -Q, Q)
            
            if total_reward > best_score:
                best_score = total_reward
                best_d0 = d0
                best_m = m

    _OPT_CACHE[key] = (best_d0, best_m)
    return _OPT_CACHE[key]

def quote(t, inventory, mid, params):
    """
    Quote policy using pre-computed optimal affine parameters.
    """
    # Deadline check
    if time.time() > _DEADLINE:
        depth = np.full(len(inventory), 1.0 / params["k"])
        return depth, depth.copy()

    sigma = params['sigma']
    A = params['A']
    k = params['k']
    gamma = params['gamma']
    T = params['T']
    steps = params['steps']
    Q = params['Q']
    
    # Get optimized parameters
    d0, m = _optimize_regime(sigma, A, k, gamma, T, steps, Q)
    
    # Apply affine policy
    q = inventory.astype(float)
    
    bid_depths = d0 + m * q
    ask_depths = d0 - m * q
    
    # Ensure valid depths
    # Min depth 0, max depth 5/k (arbitrary large bound)
    bid_depths = np.clip(bid_depths, 0.0, 5.0 / k)
    ask_depths = np.clip(ask_depths, 0.0, 5.0 / k)
    
    # Handle boundaries explicitly to be safe and efficient
    # If q >= Q, no bid fill is possible. Setting depth to 0 or inf?
    # If we set depth to 0, we quote at mid. Fill prob is high, but model says no fill.
    # So reward is 0, inventory change is 0.
    # If we set depth to inf, no quote. Reward 0, inventory change 0.
    # It doesn't matter for the objective, but inf is cleaner for "no quote".
    # However, the problem says "np.inf means no quote".
    # Let's set to 0.0 if at boundary to avoid any potential issues with inf in calculations 
    # if the evaluator does something weird, though it should handle inf.
    # Actually, if we quote at depth 0, p_fill = 1-exp(-A*dt). 
    # But the model overrides this: "no bid fill at q=Q".
    # So the evaluator will set p_fill=0 regardless of depth.
    # So depth value doesn't matter for the outcome at boundary.
    # We just need to ensure the depth is valid (>= 0).
    
    # One edge case: if m is large, bid_depth might become negative for high q?
    # We clipped to 0.0.
    
    return bid_depths, ask_depths
# EVOLVE-BLOCK-END
