"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# We will use a hybrid acquisition:
# 1. Expected Improvement (EI) with a small exploration bonus.
# 2. Cost-awareness: we divide by cost, but we also consider that
#    later in the run (higher total cost), we should be more exploitative.
# 3. To handle the "stagnation" directive, we introduce a structural change:
#    We use a "Pessimistic Value Function" (PVF) or "Upper Confidence Bound" (UCB)
#    style acquisition that balances exploration and exploitation differently.
#    
#    Specifically, we will use:
#    Score = (f_best - mu) * Phi(z) + sigma * phi(z)  (Standard EI)
#    But we modify the exploitation term.
#    
#    Actually, the directive says "Make a STRUCTURAL change".
#    The current code is standard cost-aware EI.
#    A strong alternative is the "Lower Confidence Bound" (LCB) for minimization:
#    LCB = mu - kappa * sigma
#    But we need to return a score where higher is better.
#    So, Score = -LCB = -mu + kappa * sigma.
#    
#    However, LCB can be too exploratory or too exploitative depending on kappa.
#    EI is generally more robust.
#    
#    Let's look at the cost.
#    Cost = 1 + mean(x_scaled).
#    Points near the origin (if scaled such that 0 is center? No, x is scaled to [0,1]).
#    The problem says "x scaled to [0, 1]".
#    For Ackley on [-5, 5], x=0 (center) scales to 0.5. Cost = 1 + 0.5 = 1.5.
#    For Hartmann on [0, 1], x is already in [0,1]. Cost = 1 + mean(x).
#    Minimizers are moved to uniform points in the inner 80%.
#    
#    If we use standard EI / cost, we might be penalizing points that are "expensive"
#    even if they are very promising.
#    
#    Let's try a different structural approach: **Thompson Sampling** approximation?
#    No, we only have mu and sigma.
#    
#    How about **Expected Regret** with a modified cost model?
#    The cost is linear in the coordinates.
#    
#    Let's try **Cost-Aware UCB**:
#    Score = (mu - f_best) / cost  +  kappa * sigma / sqrt(cost)
#    Wait, higher score = evaluate next.
#    We want to minimize f. So we want low mu.
#    Standard EI: E[f_best - f] = (f_best - mu)Phi(z) + sigma phi(z).
#    This is always positive if f_best > mu.
#    
#    Let's try **Pessimistic Value Function (PVF)**:
#    PVF = mu + beta * sigma. (For minimization, we prefer low PVF).
#    Score = -PVF / cost?
#    Or Score = (f_best - (mu + beta*sigma)) / cost?
#    This is similar to EI but without the non-linearity of the normal CDF.
#    EI is known to be more sample efficient.
#    
#    The directive mentions "STAGNATION".
#    Maybe the standard EI gets stuck in local minima because the GP is too confident?
#    Or maybe the cost penalty is too strong?
#    
#    Let's try a **hybrid**:
#    If the run is early (low total cost), use more exploration (higher kappa in UCB or higher exploration in EI).
#    If the run is late, use more exploitation.
#    
#    But we don't know the total cost explicitly, only the cost of the 2000 candidates.
#    Wait, `cost` is the cost of evaluating each of the 2000 candidates.
#    The total cost is not passed.
#    However, `f_best` is passed.
#    
#    Let's look at the "FRESH DESIGN" directive.
#    "Write a new design from scratch".
#    
#    Let's use **Expected Improvement with a noise-free GP** assumption?
#    No, sigma is provided.
#    
#    Let's try **Maximin Infill Criterion** combined with EI?
#    No, we need a single score.
#    
#    How about **Knowledge Gradient (KG)**?
#    KG is computationally expensive (requires integration over the future).
#    With 2000 candidates and 300s limit, we might be able to do it if we approximate it.
#    But KG is usually for the *next* point, not 2000 at once.
#    We can approximate KG as EI.
#    
#    Let's try **Cost-Aware EI with a dynamic exploration parameter**.
#    The standard EI doesn't have a parameter.
#    We can use **UCB** which has a parameter `kappa`.
#    Score = - (mu - kappa * sigma) / cost
#    We need to choose kappa.
#    A common choice is kappa = 1 or 2.
#    
#    Let's try a **mixture** of EI and UCB.
#    Or simply **UCB** because it is structurally different from EI.
#    EI is non-linear in z. UCB is linear in mu and sigma.
#    
#    Let's try:
#    Score = (f_best - mu + kappa * sigma) / cost
#    This is equivalent to minimizing (mu - kappa*sigma) * cost / (f_best - mu + kappa*sigma)? No.
#    It's just (f_best - mu + kappa*sigma) / cost.
#    This rewards points with low mu and high sigma (exploration).
#    And penalizes high cost.
#    
#    Is this better than EI?
#    EI = (f_best - mu) Phi(z) + sigma phi(z).
#    For large z (very promising), EI ~ f_best - mu.
#    For small z (uncertain), EI ~ 0.5 * (f_best - mu) + ...
#    
#    UCB-like score:
#    If mu is very low, f_best - mu is large.
#    If sigma is large, kappa*sigma adds to the score.
#    
#    Let's try **kappa = 1.0** initially.
#    
#    Another structural change: **Use the cost to adjust the exploration**.
#    Expensive points should be explored less? Or more?
#    If a point is expensive, we should be very sure it's good.
#    So, for expensive points, we should rely more on mu (exploitation) and less on sigma (exploration).
#    For cheap points, we can afford to explore more.
#    
#    So, let kappa depend on cost.
#    kappa(cost) = base_kappa / cost?
#    Or kappa(cost) = base_kappa * (1 / (1 + cost))?
#    
#    Let's try:
#    Score = (f_best - mu + kappa(c) * sigma) / c
#    where kappa(c) = 1.0 / c.
#    Then Score = (f_best - mu) / c + sigma / c^2.
#    This heavily penalizes exploration for expensive points.
#    
#    Let's try this.
#    
#    Also, we need a deadline guard.
#    
# EVOLVE-BLOCK-END

def acquisition(mu, sigma, f_best, cost):
    start_time = time.time()
    deadline = start_time + 290.0  # 10s margin
    
    # Ensure inputs are arrays
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)
    cost = np.asarray(cost)
    
    # Handle edge cases
    if mu.shape != (2000,) or sigma.shape != (2000,) or cost.shape != (2000,):
        # Fallback to simple EI if shapes are wrong, though contract says shape is (2000,)
        pass
    
    # Clip sigma to avoid division by zero or negative values
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Structural Change: Cost-aware UCB with adaptive exploration
    # Exploration decreases with cost.
    # Score = (f_best - mu + kappa * sigma) / cost
    # Let kappa = 1.0 / cost.
    # Score = (f_best - mu) / cost + sigma / cost^2
    
    # Calculate the score
    # Term 1: Exploitation per cost
    term_exploit = (f_best - mu) / cost
    
    # Term 2: Exploration per cost^2 (penalized more for high cost)
    term_explore = sigma / (cost * cost)
    
    # Combine
    scores = term_exploit + term_explore
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # This shouldn't happen for vectorized numpy ops, but just in case
        pass
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
