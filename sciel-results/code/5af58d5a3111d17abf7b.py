import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety check for empty inputs
    if mu.size == 0:
        return np.array([])
    
    # Handle zero/negative sigma to avoid division by zero and log issues
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)
    
    # Calculate z-score
    # z = (f_best - mu) / sigma
    # If mu < f_best, z is positive (potential improvement)
    z = (f_best - mu) / sigma_safe
    
    # Calculate CDF and PDF
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement (PoI)
    poi = cdf_z
    
    # --- Structural Change: Adaptive Exploration-Exploitation Balance ---
    # The current approach uses a fixed linear cost penalty and a fixed PoI weight.
    # Shekel 4d is multimodal. A fixed balance may trap the search in local optima
    # or explore too broadly.
    #
    # 1. **Cost Awareness via Log-Penalty**: 
    #    Linear cost penalty (1/cost) might be too aggressive or too weak.
    #    Using a logarithmic penalty or a power less than 1 can soften the cost
    #    impact, allowing slightly more expensive points if they offer high EI.
    #    However, cost is bounded [1, 2]. 
    #    Let's try a softer penalty: cost^0.5 or simply 1/sqrt(cost).
    #    Actually, let's look at the "Knowledge" hint: "Adaptive PoI/EI mix with uncertainty bonus".
    #    
    # 2. **Adaptive PoI Weight**:
    #    The weight on PoI should depend on the uncertainty (sigma) or the current
    #    improvement potential. 
    #    If sigma is high, we should explore more (higher PoI weight).
    #    If sigma is low, we should exploit (higher EI weight).
    #    
    #    Let's define an exploration weight that scales with normalized sigma.
    #    We need a reference scale for sigma. Since targets are standardized in the GP,
    #    sigma is in objective units but relative to the function's scale.
    #    However, we don't know the global scale. 
    #    We can use the current EI magnitude or a fixed heuristic.
    #    
    #    Alternative: Use a "Thompson Sampling" like approach or a modified UCB.
    #    UCB = mu - kappa * sigma. 
    #    Cost-aware UCB: (mu - kappa * sigma) / cost.
    #    But UCB can be negative. We need scores where higher is better.
    #    So, Score = -(mu - kappa * sigma) / cost = (kappa * sigma - mu) / cost.
    #    This is essentially a linear combination of mu and sigma.
    #    
    #    Let's try a hybrid: 
    #    Score = (EI + alpha * sigma * PoI) / cost^beta
    #    
    #    Let's analyze the "Adaptive PoI/EI mix" hint more closely.
    #    It suggests the mix itself is adaptive.
    #    
    #    Let's try an approach that rewards high uncertainty more strongly when
    #    the expected improvement is low (to escape local minima), and rewards
    #    high expected improvement when uncertainty is low.
    #    
    #    Let's define a "confidence" term.
    #    z is the number of standard deviations f_best is above mu.
    #    If z is large, we are confident mu is better.
    #    If z is small/negative, we are uncertain.
    #    
    #    Let's use a dynamic alpha for PoI.
    #    alpha = 1.0 / (1.0 + z^2) ? No, z can be negative.
    #    alpha = 1.0 / (1.0 + |z|) ?
    #    
    #    Let's try a simpler structural change: **Log-Posterior Sampling Approximation**
    #    or **Knowledge Gradient** approximation? Too complex.
    #    
    #    Let's go with **Adaptive UCB with Cost Normalization**.
    #    The UCB criterion is often more robust for multimodal functions than EI
    #    because EI can be "greedy" and ignore regions with high potential but
    #    low immediate expected gain due to high variance.
    #    
    #    UCB_score = -mu + kappa * sigma
    #    We want to maximize this, penalized by cost.
    #    Score = (-mu + kappa * sigma) / cost
    #    
    #    How to choose kappa?
    #    A common adaptive kappa is based on the number of evaluations or time.
    #    But we don't have explicit step count. We have cost.
    #    Total cost budget is 30.
    #    We can estimate progress based on cost? No, cost is per point.
    #    
    #    Let's use a fixed kappa that balances exploration and exploitation well for
    #    4D Shekel. 
    #    However, the hint says "Adaptive PoI/EI mix".
    #    
    #    Let's try: Score = (EI * w_ei + PoI * sigma * w_poi) / cost
    #    Where w_poi increases as we explore more?
    #    
    #    Actually, let's look at the failure mode of the current score 1.2255.
    #    It likely converges to a local minimum.
    #    To fix this, we need to force exploration of unexplored regions.
    #    Regions with high sigma are unexplored.
    #    
    #    Let's try a **Penalized EI** where the penalty is stronger for points
    #    that are "close" to already evaluated points? We don't have coordinates.
    #    But high sigma indicates distance from data.
    #    
    #    Let's try: 
    #    Score = EI / cost + lambda * sigma / cost
    #    This is equivalent to (EI + lambda * sigma) / cost.
    #    Since EI already contains a sigma term (sigma * pdf(z)), this is similar to UCB.
    #    
    #    Let's try a **Non-linear Cost Penalty**.
    #    Instead of 1/cost, use 1/cost^2. This heavily penalizes expensive points.
    #    Or 1/cost^0.5. This lightly penalizes.
    #    Given the budget is tight (30), we want to be efficient.
    #    The current program uses 1/cost.
    #    
    #    Let's try a **Hybrid EI-UCB** with an adaptive exploration term.
    #    
    #    Idea: 
    #    Base Score = EI / cost
    #    Exploration Bonus = sigma * (1 - cdf_z) / cost  # Probability of NOT improving * uncertainty?
    #    Or just sigma / cost.
    #    
    #    Let's try: Score = (EI + 0.5 * sigma) / cost
    #    This is a standard UCB-like formulation.
    #    
    #    To make it "adaptive", let's scale the exploration term by the current best
    #    improvement potential.
    #    
    #    Let's try a **Logarithmic EI** to dampen the influence of very high EI values
    #    which might occur in far-away unexplored regions with high variance, potentially
    #    leading to erratic jumps. 
    #    Score = log(1 + EI) / cost + log(1 + sigma) / cost
    #    
    #    Let's stick to a robust **Adaptive UCB**:
    #    kappa = 1.0
    #    ucb = -mu + kappa * sigma
    #    score = ucb / cost
    #    
    #    But EI is generally better than UCB for Bayesian Optimization.
    #    Let's blend them.
    #    
    #    Score = (0.5 * EI + 0.5 * (-mu + sigma)) / cost
    #    
    #    Let's try a different angle: **Probability of Significant Improvement (PSI)**
    #    PSI = P(f(x) < f_best - delta)
    #    This is more demanding than PoI.
    #    z_psi = (f_best - delta - mu) / sigma
    #    psi = norm.cdf(z_psi)
    #    
    #    Let's try **EI with a dynamic kappa for the sigma term**.
    #    Standard EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #    This is equivalent to integrating the improvement.
    #    
    #    Let's try **Lower Confidence Bound (LCB)** for minimization?
    #    LCB = mu - kappa * sigma.
    #    We want to minimize f, so we want low LCB.
    #    Score = -LCB / cost = (kappa * sigma - mu) / cost.
    #    This is the same as UCB for maximization of -f.
    #    
    #    Let's go with **LCB with Adaptive Kappa**.
    #    kappa starts high (exploration) and decreases?
    #    We don't have a step counter.
    #    However, we can use the average cost or something? No.
    #    
    #    Let's assume the GP is well-calibrated.
    #    Let's try **Thompson Sampling** approximation.
    #    Sample y ~ N(mu, sigma).
    #    Score = -y / cost.
    #    But we need to return deterministic scores for 2000 points.
    #    We can approximate the expected value of -y/cost? That's just LCB with kappa=1.
    #    Or we can sample? No, we must return a single score per point.
    #    
    #    Let's try **Optimistic Posterior Sampling (OPS)** style.
    #    Score = max(-mu, -mu + sigma) / cost? No.
    #    
    #    Let's try **Knowledge Gradient (KG)** approximation.
    #    KG is hard to compute efficiently for 2000 points without vectorization tricks.
    #    
    #    Let's go back to **Adaptive EI/PoI**.
    #    The hint says "Adaptive PoI/EI mix with uncertainty bonus".
    #    
    #    Let's define:
    #    Score = (EI + alpha * PoI * sigma) / cost
    #    Where alpha is adaptive.
    #    
    #    If we are early in the process (low total cost used?), alpha should be high.
    #    We don't know total cost used.
    #    But we can infer it? No.
    #    
    #    However, the "cost" input is the cost of the candidate point.
    #    The loop runs until total cost > 30.
    #    The evaluator picks the best score.
    #    
    #    Let's try a **Fixed but Tuned UCB**:
    #    Score = ( -mu + 1.5 * sigma ) / cost
    #    
    #    Let's try **EI with a stronger exploration term**:
    #    Score = ( EI + 0.5 * sigma ) / cost
    #    
    #    Let's compare EI vs UCB.
    #    EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    #    If mu is very low, z is large, Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    #    UCB = -mu + kappa * sigma.
    #    If mu is very low, UCB is very low (high score).
    #    
    #    EI is more "cautious" because it weights the probability.
    #    
    #    Let's try a **Log-UCB**:
    #    Score = log(1 + max(0, -mu + kappa * sigma)) / cost
    #    
    #    Let's try a **Hybrid**:
    #    Score = (0.8 * EI + 0.2 * (-mu + sigma)) / cost
    #    
    #    Let's try to implement a **Simple Regret Minimization** proxy.
    #    The goal is to minimize simple regret: f_best - min(f(x)).
    #    
    #    Let's try **Probability of Improvement (PoI) weighted by Sigma**:
    #    Score = PoI * sigma / cost
    #    This favors points that are likely to improve AND have high uncertainty.
    #    
    #    Let's try **Expected Hypervolume Improvement (EHI)**? No, single objective.
    #    
    #    Let's try **Entropy Search**? No.
    #    
    #    Let's go with **Adaptive LCB (Lower Confidence Bound)**:
    #    The key is the kappa value.
    #    A common choice is kappa = 1.0 or 2.0.
    #    Let's use kappa = 1.5.
    #    
    #    Score = (1.5 * sigma - mu) / cost
    #    
    #    But we must ensure scores are finite and positive? No, "higher = evaluate next".
    #    If mu is very high (bad), -mu is very negative. Score is negative.
    #    If mu is very low (good), -mu is positive. Score is positive.
    #    This is fine.
    #    
    #    However, if we mix with EI, EI is always non-negative.
    #    LCB can be negative.
    #    Mixing them might be tricky.
    #    
    #    Let's try **Normalized EI**:
    #    Normalize EI by the range of EI?
    #    
    #    Let's try a **Two-Phase Strategy**:
    #    We can't detect phase.
    #    
    #    Let's try **Cost-Effective EI**:
    #    Score = EI / (cost * sigma)
    #    This rewards high EI relative to uncertainty.
    #    If sigma is high, the score is lower, encouraging exploitation of certain improvements.
    #    This might reduce exploration too much.
    #    
    #    Let's try **Cost-Effective UCB**:
    #    Score = (-mu + kappa * sigma) / (cost * sigma)
    #    = (-mu/sigma + kappa) / cost
    #    = (kappa - z) / cost  (since z = (f_best - mu)/sigma, so mu = f_best - z*sigma)
    #    Wait, z = (f_best - mu) / sigma => mu = f_best - z * sigma.
    #    -mu + kappa * sigma = -f_best + z * sigma + kappa * sigma = sigma * (z + kappa) - f_best.
    #    So Score = (sigma * (z + kappa) - f_best) / cost.
    #    
    #    Let's try **z-score based scoring**:
    #    Score = (z + kappa) / cost
    #    This ignores the magnitude of sigma and f_best.
    #    This is effectively comparing the number of standard deviations we are above f_best.
    #    This is a common heuristic.
    #    
    #    Let's try **Score = (z + 1.0) / cost**
    #    
    #    Let's combine EI and z-score.
    #    Score = (EI + z) / cost
    #    
    #    Let's try **Score = (EI + 0.5 * z) / cost**
    #    
    #    I will implement a **Hybrid EI-UCB** with a **Log-Cost Penalty** to soften the cost impact slightly, allowing for more exploration of slightly more expensive regions if they have high potential.
    #    
    #    Score = (EI + 0.5 * sigma) / (cost ** 0.8)
    #    
    #    Let's check the budget. 30 cost.
    #    If we use cost ** 0.8, the penalty is less severe.
    #    
    #    Let's try:
    #    Score = (EI + 0.3 * sigma) / cost
    #    
    #    I will use **EI + 0.5 * sigma** divided by **cost**.
    #    This is a standard UCB-like blend.
    
    # Structural Change: Use a softer cost penalty and a stronger exploration term
    # to escape local minima in Shekel 4d.
    
    # Exploration term: sigma
    # Exploitation term: EI
    # Cost penalty: 1 / cost
    
    # Let's try a weighted sum with adaptive weights based on z.
    # If z is high, we are confident, so weight EI more.
    # If z is low, we are uncertain, so weight sigma more.
    
    # Weights:
    # w_ei = 1.0
    # w_sig = 0.5 + 0.5 * (1.0 / (1.0 + np.abs(z)))
    # This makes w_sig between 0.5 and 1.0.
    
    # Actually, let's keep it simple and robust.
    # Score = (EI + 0.5 * sigma) / cost
    
    exploration_term = 0.5 * sigma_safe
    combined = ei + exploration_term
    scores = combined / cost_safe
    
    # Ensure finite outputs
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return np.zeros_like(mu)
        
    return scores
