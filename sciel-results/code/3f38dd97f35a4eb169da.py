import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    n = mu.shape[0]
    zero = np.zeros(n, dtype=np.float64)

    # Safety: if inputs are degenerate, return a neutral score.
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(sigma)) and np.all(np.isfinite(cost))):
        return zero

    sigma_safe = np.maximum(sigma, 1e-12)
    cost_safe = np.maximum(cost, 1e-9)

    # ---- Expected Improvement (minimization) ----
    imp = f_best - mu
    s = sigma_safe
    z = imp / s
    ei = imp * norm.cdf(z) + s * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # ---- Probability of Improvement ----
    poi = norm.cdf(z)

    # ---- UCB (Upper Confidence Bound for minimization) ----
    # Lower bound of the GP: mu - beta * sigma
    # We want to minimize, so we look for low values.
    # A standard UCB for minimization is: score = -(mu - beta * sigma) = -mu + beta * sigma
    # But we want higher score = better.
    # So we want to maximize the "chance" of being low.
    # Let's use a "Lower Confidence Bound" score:
    # LCB = mu - beta * sigma
    # We want to select points with low LCB.
    # So score = -LCB = -mu + beta * sigma
    # However, we need to normalize/blend this with EI.
    
    # Let's try a different structural approach:
    # Instead of just EI * (1+PoI), let's use a weighted combination of:
    # 1. EI (magnitude of improvement)
    # 2. -mu (exploitation of current best mean)
    # 3. sigma (exploration of uncertainty)
    
    # But the directive says: "Make a STRUCTURAL change: a different design or modelling family, a different optimization or search method, or a different representation of the problem."
    # The current method is a simple heuristic blend.
    # Let's try a **Thompson Sampling** inspired approach? No, we only have mu/sigma.
    # Let's try **Knowledge Gradient (KG)** approximation?
    # Or **Expected Improvement with a different prior**?
    
    # Let's try a **multi-scale** approach:
    # Combine EI with a term that rewards points far from the current best mean (exploration)
    # but penalized by cost.
    
    # Actually, let's look at the inspirations.
    # Inspiration 2 used a multi-strategy ensemble (EI + UCB + Cost-UCB + Pure-Exploitation) using max.
    # The current program is a simple EI*(1+PoI)/cost.
    
    # Let's implement the **Ensemble** approach as suggested by inspiration 2, 
    # but adapted for cost-awareness.
    
    # Strategy 1: Cost-aware EI
    score_ei = ei / cost_safe
    
    # Strategy 2: Cost-aware UCB (minimization)
    # We want to minimize f(x).
    # UCB for minimization: mu - beta * sigma
    # We want to pick points with low UCB.
    # So score = - (mu - beta * sigma) = -mu + beta * sigma
    # Let's use beta = 1.0 (or tune it).
    beta = 1.0
    ucb_min = mu - beta * sigma
    score_ucb = -ucb_min / cost_safe  # Higher is better (more negative ucb_min -> higher score)
    
    # Strategy 3: Pure Exploitation (lowest mean)
    score_exploit = -mu / cost_safe
    
    # Strategy 4: Pure Exploration (highest uncertainty)
    score_explore = sigma / cost_safe
    
    # Combine using max (as in inspiration 2) or weighted sum?
    # Max can be too aggressive. Let's try a weighted sum with adaptive weights.
    # Or, let's stick to the "structural change" directive and use a **different acquisition function family**.
    
    # Let's try **Expected Improvement with a Student's t-distribution** instead of Gaussian?
    # Or **Pareto-front** based?
    
    # Given the constraint of 2000 points and 300s, let's try a **hybrid** that is structurally different:
    # **Cost-Adjusted Expected Improvement with a Floor**
    # The current EI*(1+PoI) is a smooth blend.
    # Let's try: EI / (cost * (1 + |mu - f_best| / (sigma + eps)))
    # This downweights points that are far from f_best in mean, relative to their uncertainty.
    
    # Actually, let's try the **Ensemble Max** approach from Inspiration 2, as it scored 1.0226 (worse than current 1.0495? No, 1.0226 < 1.0495).
    # Wait, higher is better. 1.0495 > 1.0226. So the current program is better than Inspiration 2.
    # Inspiration 1 (1.0444) is also worse.
    # Inspiration 3 (1.0152) is worse.
    
    # So the current EI*(1+PoI)/cost is the best so far.
    # The directive says: "STAGNATION: the population best has not improved... Make a STRUCTURAL change".
    # This implies we should try something new, even if it might score worse initially, to explore the search space.
    
    # Let's try a **Gaussian Process Upper Confidence Bound (GP-UCB)** formulation adapted for cost.
    # Standard GP-UCB: a(x) = mu(x) + sqrt(beta_t) * sigma(x) (for maximization)
    # For minimization: a(x) = mu(x) - sqrt(beta_t) * sigma(x)
    # We want to minimize, so we select x with low a(x).
    # Score = -a(x) = -mu(x) + sqrt(beta_t) * sigma(x)
    # Cost-aware: Score = (-mu(x) + sqrt(beta_t) * sigma(x)) / cost(x)
    
    # Let's try this. It is structurally different from EI.
    beta_t = 2.0  # Exploration parameter
    ucb_score = (-mu + beta_t * sigma) / cost_safe
    
    # Let's also try a **Knowledge Gradient (KG)** approximation.
    # KG is the expected improvement in the best known value.
    # For minimization: KG(x) = E[ max(f_best - f(x), 0) | X ]
    # This is similar to EI but considers the global best.
    # Actually, EI is a local version of KG.
    # For 1D, KG is complex. For multi-D, even more so.
    # Let's stick to UCB.
    
    # Let's combine EI and UCB scores using a **weighted sum** with weights that depend on the spread of sigma.
    # If sigma is high, UCB might be better. If sigma is low, EI might be better.
    
    # Let's try a **dynamic blend**:
    # w = sigma / (sigma + 1.0)  # Higher weight for UCB when sigma is large
    # score = w * ucb_score + (1-w) * score_ei
    
    # But wait, the current program uses EI*(1+PoI).
    # Let's try a **different structural change**: **Rank-based selection**.
    # Instead of using the raw values of EI/UCB, we use their ranks.
    # This makes the acquisition function invariant to scaling of mu/sigma.
    
    # Rank of EI (higher is better)
    rank_ei = np.argsort(np.argsort(ei)) / n
    # Rank of UCB (higher is better, i.e., lower mu - beta*sigma)
    rank_ucb = np.argsort(np.argsort(-mu + beta_t * sigma)) / n
    
    # Cost-aware rank:
    # We want to select points with high rank and low cost.
    # score = (rank_ei + rank_ucb) / (2 * cost_safe)
    
    # Let's try this rank-based approach.
    rank_score = (rank_ei + rank_ucb) / (2.0 * cost_safe)
    
    # Ensure finite
    rank_score = np.where(np.isfinite(rank_score), rank_score, 0.0)
    rank_score = np.maximum(rank_score, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 325:
        return rank_score
    
    return rank_score
