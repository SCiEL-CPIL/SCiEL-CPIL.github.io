import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_DEADLINE = 300.0  # wall-clock safety margin below the 330s kill limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for minimizing Levy 2D.
    Combines a strong cost penalty (to bias towards the low-cost corner where
    shifted minima are likely to offer the best value/cost trade-off) with
    Expected Improvement (for local exploitation) and a mild exploration term.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Deadline guard
    if time.time() > _DEADLINE:
        return 1.0 / np.maximum(cost, 1.0)

    # 1. Compute Expected Improvement (EI) for minimization
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)

    # 2. Exploration term:
    #    We want to explore uncertain regions, but not excessively if EI is high.
    #    A simple sigma term is often too noisy. 
    #    Let's use a UCB-like term but keep it secondary to EI.
    #    beta = 1.0 is a moderate exploration constant.
    beta = 1.0
    ucb_part = beta * sigma_safe
    
    # Combine EI and Exploration.
    # EI is in objective units. UCB is in objective units (since sigma is std dev of objective).
    # We can sum them directly.
    base_score = ei + ucb_part

    # 3. Cost-aware scaling:
    #    The key insight from the "inspiration" (which actually scored poorly in isolation 
    #    but the "current" program's lesson says cost^2 helped) is that cost matters.
    #    However, simply dividing by cost can be too aggressive if EI is very small.
    #    
    #    Let's look at the "current" program's success: 
    #    "Stronger cost penalty (cost^2) biases search towards the low-cost corner... Measured: 1 -> 0.2431."
    #    Wait, the current program in the prompt uses `ucb / cost` with `beta=4.0` and scored 0.2431.
    #    The inspiration used `score / cost^0.5` and scored 0.0224.
    #    
    #    The lesson says: "Stronger cost penalty (cost^2) ... Measured: 1 -> 0.2431".
    #    This implies that the version that scored 0.2431 likely had a stronger cost penalty than the seed.
    #    The current program provided in the prompt *is* the one that scored 0.2431 (it says "CURRENT PROGRAM (score 0.2431)").
    #    It uses `ucb / cost`.
    #    
    #    The inspiration (0.0224) used EI + explore, divided by cost^0.5.
    #    
    #    So, the current strategy (UCB / cost) is working better.
    #    Why? UCB with high beta (4.0) is very exploratory. Dividing by cost biases that exploration towards cheap areas.
    #    EI is more exploitative. If we use EI, we might get stuck in local minima if the GP is inaccurate.
    #    
    #    Let's try to combine: 
    #    Use a blend of EI and UCB, but keep the strong cost penalty.
    #    
    #    Actually, the lesson says "Stronger cost penalty (cost^2)". 
    #    The current program uses `1/cost`. 
    #    Maybe `1/cost^2` is even better? Or maybe the "1 -> 0.2431" refers to a change from a previous state?
    #    The prompt says: "Measured: 1 -> 0.2431. (strategy of a program that improved island 3)".
    #    This suggests a program using cost^2 achieved 0.2431.
    #    But the CURRENT program provided also has score 0.2431 and uses `1/cost`.
    #    This is confusing. Let's assume the current program is the baseline to beat.
    #    
    #    Let's try a structural change as requested by STAGNATION directive:
    #    "Make a STRUCTURAL change: a different design or modelling family... A first try in a new direction may score worse; that is acceptable."
    #    
    #    Current: UCB / cost.
    #    Inspiration: (EI + Explore) / cost^0.5.
    #    
    #    New Direction: 
    #    Instead of linear UCB, let's use a **Posterior Sampling** approximation or a **Thompson Sampling** like score?
    #    No, we only have mu and sigma.
    #    
    #    How about **Cost-Effective Global Optimization (CEGO)** style?
    #    Score = (f_best - mu) / cost? (Expected improvement per unit cost)
    #    This is similar to EI/cost.
    #    
    #    Let's try **EI / cost**.
    #    EI is more focused than UCB. UCB with beta=4 is very broad.
    #    If the GP is somewhat accurate, EI might be better.
    #    If the GP is inaccurate (length scale 0.2 might be too small for Levy's complexity?), UCB might be better.
    #    The task says "GP: Matern-5/2, length scale 0.2". Levy is rugged. 0.2 is small.
    #    Small length scale means the GP thinks the function changes rapidly.
    #    This leads to high sigma everywhere except near observed points.
    #    UCB will pick points far from observations.
    #    EI will also pick points where mu is low and sigma is moderate.
    #    
    #    Let's try a hybrid: 
    #    score = (ei + alpha * sigma) / cost^gamma
    #    
    #    Let's stick to the "Stronger cost penalty" idea. 
    #    If `1/cost` gave 0.2431, maybe `1/cost^1.5` or `1/cost^2` is better?
    #    The lesson explicitly mentions `cost^2`.
    #    
    #    Let's try: `ei / cost^2`?
    #    Or `(ei + sigma) / cost^2`?
    #    
    #    Let's look at the inspiration again. It scored 0.0224. It used `cost^0.5`.
    #    The current program scores 0.2431. It uses `cost^1`.
    #    So increasing cost penalty from 0.5 to 1 improved score by ~10x.
    #    It is plausible that increasing to 2 improves it further.
    #    
    #    However, the directive says "Make a STRUCTURAL change".
    #    Changing the exponent is a tuning change, not structural.
    #    
    #    Structural change: **Use a different acquisition core**.
    #    Current: UCB (linear in sigma).
    #    Inspiration: EI (non-linear in sigma/mu).
    #    
    #    Let's switch to **EI-based** acquisition but with a **very strong cost penalty**.
    #    Score = EI / cost^2.
    #    
    #    Why? 
    #    1. EI is generally superior to UCB for BO when the GP is reasonably calibrated.
    #    2. The cost penalty is the key driver for this specific task (Levy shifted minima).
    #    3. The lesson suggests cost^2 is beneficial.
    #    4. This is structurally different from the current UCB-based approach.
    
    # Calculate EI
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Strong cost penalty
    # cost is in [1, 2] (since x in [0,1], mean in [0,1], cost = 1+mean in [1,2])
    # cost^2 is in [1, 4]
    cost_safe = np.maximum(cost, 1.0)
    penalty = cost_safe ** 2
    
    scores = ei / penalty
    
    # Add a tiny exploration term to avoid getting stuck if EI is 0 everywhere?
    # If EI is 0, scores are 0. 
    # Maybe add a small sigma/cost^2 term?
    # Let's try pure EI/cost^2 first.
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
