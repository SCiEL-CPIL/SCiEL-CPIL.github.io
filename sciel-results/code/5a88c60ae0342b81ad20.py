"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Parameters
    ----------
    mu : np.ndarray, shape (2000,)
        GP mean predictions
    sigma : np.ndarray, shape (2000,)
        GP standard deviation predictions
    f_best : float
        Best observed value so far
    cost : np.ndarray, shape (2000,)
        Cost of evaluating each candidate
    
    Returns
    -------
    np.ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next)
    """
    # Ensure finite inputs and proper types
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases: avoid division by zero and negative sigma
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Calculate Expected Improvement (EI) for minimization
    # Improvement is positive if mu < f_best
    improvement = f_best - mu
    z = improvement / sigma
    ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate Upper Confidence Bound (UCB) for minimization
    # We want low values, so UCB_min = mu - kappa * sigma
    # However, standard UCB is for maximization. 
    # For minimization, we can use Lower Confidence Bound (LCB) = mu - kappa * sigma
    # Or equivalently, maximize (f_best - mu) + kappa * sigma ? 
    # Let's stick to a hybrid: Cost-aware EI is often good, but can be too greedy.
    # Let's use a "Probability of Improvement" (PI) or a modified UCB.
    
    # Strategy: Adaptive UCB with Cost Awareness
    # The parameter kappa controls exploration. 
    # Early in the budget, we might want more exploration.
    # However, we don't know the step count directly, but we can infer from f_best or cost?
    # Actually, the contract says "While total cost is below 30".
    # We don't have the current total cost passed in, only the cost of the *candidate*.
    # But f_best changes. If f_best is high, we are early. If low, we are late.
    # Ackley min is 0. 
    # Let's use a dynamic kappa based on the normalized improvement.
    
    # A robust alternative: Stochastic Selection (Thompson Sampling approximation)
    # We cannot sample 2000 x N samples easily in 300s if N is large, but we can do a single vectorized "sample" 
    # or use the analytical expectation of a utility function.
    
    # Let's use a "Cost-Adjusted UCB" where the exploration term is scaled by the inverse of the cost.
    # Score = (f_best - mu) * w1 + sigma * w2 / cost
    
    # To make it structurally different from simple EI/cost:
    # Use a "Hyper-volume" like proxy or a "Pareto" filter?
    # Inspiration 3 mentioned Pareto-frontier filtering.
    
    # Let's try a "Normalized Improvement" approach.
    # Scale EI and UCB to comparable ranges.
    
    # UCB term for minimization: mu - kappa*sigma
    # We want to maximize the chance of beating f_best.
    # Let's compute the "Expected Regret" (ER) which is E[max(0, f(x) - f_best)]
    # ER = -EI (for minimization, if EI is defined as E[max(0, f_best - f(x))])
    # So minimizing ER is same as maximizing EI.
    
    # The current program uses EI/cost. This is a "Greedy" approach.
    # A different approach: **Probability of Improvement (PI) with Cost Penalty**
    # PI = P(f(x) < f_best) = Phi(z)
    # Score = PI / cost? This favors high probability over high magnitude improvement.
    
    # Or: **Lower Confidence Bound (LCB) / Cost**
    # LCB = mu - kappa * sigma
    # We want to minimize LCB. So Score = -LCB / cost = (kappa*sigma - mu) / cost
    # This is similar to EI but linear in sigma instead of the non-linear EI function.
    
    # Let's combine EI and PI with a dynamic weight.
    # If f_best is close to 0 (min), we should exploit (high EI weight).
    # If f_best is far from 0, we should explore (high PI/UCB weight).
    
    # Normalized f_best: Ackley range is roughly [0, ~27].
    # Let's assume f_best starts around 10-20 and drops to 0.
    # progress = 1 - (f_best / 27) approx.
    
    # Let's use a simpler structural change: **Vectorized Thompson Sampling Approximation**
    # Instead of deterministic EI, we add a small random noise to the scores to break ties and encourage diversity in selection over iterations?
    # No, the function is called once per iteration for 2000 candidates. The selection is deterministic (argmax).
    # Adding noise to the *score* inside the acquisition function effectively makes the selection stochastic.
    
    # Let's try: **Stochastic UCB**
    # Score = (mu - kappa * sigma) / cost + noise
    # But we must return deterministic scores? "Return 2000 finite scores".
    # If I return random scores, it's still valid, but might be unstable.
    
    # Better Structural Change: **Pareto Front Selection**
    # 1. Compute EI and Cost.
    # 2. Identify non-dominated points in (EI, 1/Cost) space? 
    # 3. Assign scores based on hypervolume contribution?
    # This is computationally heavier.
    
    # Let's try a **Weighted Sum of PI and EI** with adaptive weights.
    # This is a different "design family" (Hybrid Probabilistic vs Pure Expected Value).
    
    pi = norm.cdf(z)  # Probability of Improvement
    ei = np.maximum(ei, 0.0)
    
    # Normalize EI and PI to [0, 1] roughly to make them comparable
    # Max EI is roughly f_best. Max PI is 1.
    # Let's scale EI by f_best to get a relative improvement.
    if f_best > 1e-6:
        ei_rel = ei / f_best
    else:
        ei_rel = ei
        
    # We want to balance "likely to improve" (PI) and "how much it improves" (EI).
    # In early stages, PI is useful to find any improvement.
    # In late stages, EI is useful to squeeze out the last bit.
    
    # Heuristic: 
    # If f_best is high (> 5), we are far from min. Explore more (PI).
    # If f_best is low (< 1), we are close. Exploit more (EI).
    
    # Let's use a sigmoid switch.
    # switch = sigmoid( (f_best - 2) / 1 )
    # weight_ei = switch
    # weight_pi = 1 - switch
    
    # But we need to be careful with the cost.
    # Score = (weight_ei * ei_rel + weight_pi * pi) / cost
    
    # Let's refine the cost awareness.
    # Dividing by cost is linear. Maybe use 1/sqrt(cost) to be less penalizing?
    # Or cost^0.5?
    
    # Let's stick to the "Different Design Family": **Adaptive Weighted PI/EI Hybrid**
    
    # Calculate adaptive weight
    # f_best for Ackley 2D can be up to ~20-25 initially.
    # Let's define a threshold.
    threshold = 5.0
    scale = 2.0
    
    # Use a smoothstep or sigmoid for the weight
    # t = (f_best - threshold) / scale
    # w_exploit = 1 / (1 + exp(-t))  # High when f_best is high? 
    # Wait, if f_best is high, we want to EXPLORE (find the basin). So weight on PI (which is broader) or UCB?
    # Actually, PI is also local. UCB is more global.
    # Let's use UCB for exploration.
    
    # UCB_min = mu - kappa * sigma
    # We want to maximize (f_best - UCB_min) ? No.
    # We want to find low mu - kappa*sigma.
    # Let's define Score_UCB = (kappa * sigma - mu) / cost
    # This is high when sigma is high (uncertain) or mu is low (good).
    
    # Let's compute both EI/cost and UCB/cost.
    
    # Dynamic Kappa:
    # Start with high kappa (explore), decrease as f_best improves.
    # kappa = 1.0 + 1.0 * exp(-f_best / 2.0)
    # If f_best=20, kappa ~ 1.0
    # If f_best=0, kappa ~ 2.0 ? No, we want less exploration at the end.
    # kappa = 2.0 * exp(-f_best / 5.0) + 0.1
    # If f_best=20, kappa ~ 0.1 + small = 0.1 (Exploit)
    # If f_best=0, kappa ~ 2.1 (Explore)
    # This is backwards. We want to explore when f_best is BAD (high).
    # So:
    # kappa = 0.1 + 2.0 * (f_best / 20.0)
    # If f_best=20, kappa=2.1 (Explore)
    # If f_best=0, kappa=0.1 (Exploit)
    
    # Let's cap f_best for the scaling to avoid huge values if f_best is weird.
    f_best_scaled = np.clip(f_best, 0, 20.0)
    kappa = 0.5 + 1.5 * (f_best_scaled / 20.0)
    
    # Calculate UCB-based score for minimization
    # We want to maximize the potential for low value.
    # Score_UCB = (kappa * sigma - mu) / cost
    
    # Calculate EI-based score
    # Score_EI = ei / cost
    
    # Blend them.
    # If f_best is high (bad), trust UCB more.
    # If f_best is low (good), trust EI more.
    
    # Weight for UCB:
    w_ucb = np.clip(f_best_scaled / 10.0, 0.2, 1.0)
    w_ei = 1.0 - w_ucb
    
    scores = w_ei * (ei / cost) + w_ucb * (kappa * sigma / cost - mu / cost)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
