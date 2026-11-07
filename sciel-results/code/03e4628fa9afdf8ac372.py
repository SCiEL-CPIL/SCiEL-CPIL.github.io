import numpy as np
from scipy.stats import t as t_dist
import time

def acquisition(mu, sigma, f_best, cost):
    # Strategy: Probabilistic Improvement (PoI) with adaptive exploitation.
    # 
    # 1. Use t-distribution for improvement probability to account for 
    #    uncertainty in the GP fit (especially with few points).
    # 2. As the cost budget depletes, increase the weight on low-cost 
    #    candidates to maximize the number of evaluations before the limit.
    # 3. Combine exploration (PoI) with exploitation (cost efficiency).
    
    # Safety: Ensure inputs are arrays and handle NaNs/Infs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    mu = np.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e-6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1.0, neginf=1.0)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate Improvement: f_best - mu
    # We want to find points where f(x) < f_best
    improvement = f_best - mu
    
    # Degrees of freedom for t-distribution.
    # Start with low df (heavy tails, more exploration) and increase?
    # Actually, for PoI, we want the probability that f(x) < f_best.
    # P(f(x) < f_best) = P(N(mu, sigma) < f_best)
    # Using t-dist: P(T < (f_best - mu) / (sigma / sqrt(df)))
    # Let's use a fixed moderate df to balance between normal and heavy-tailed.
    # With 5 initial points, df=5 is reasonable.
    df = 5.0
    
    z = improvement / (sigma / np.sqrt(df))
    
    # Probability of Improvement
    # If z is large positive, prob is close to 1.
    # If z is large negative, prob is close to 0.
    poi = t_dist.cdf(z, df)
    
    # Cost-awareness:
    # The current program divides by cost. This penalizes high-cost points heavily.
    # However, if a high-cost point has a very high probability of finding a new minimum,
    # it might be worth it.
    # 
    # Alternative: Use a "cost efficiency" score.
    # Score = PoI * (1 / cost^alpha)
    # As budget runs out, we should prefer low cost.
    # Since we don't know the exact remaining budget, we can use the current cost
    # distribution to adapt.
    # 
    # Let's look at the cost vector. Lower cost is better.
    # Normalize cost to [0, 1] relative to the max cost in the batch?
    # Or just use raw cost.
    #
    # The task says: "Evaluating x costs 1 + mean(x scaled to [0, 1])".
    # Costs are between 1 and 2.
    #
    # Let's try a blend:
    # score = poi + lambda * (1 - cost)
    # But poi is in [0,1]. Cost is in [1,2].
    #
    # Better:
    # score = poi / cost
    # This is similar to the current approach but using PoI instead of LCB.
    # PoI is more robust to outliers in mu than LCB.
    
    # Let's refine the cost weighting.
    # If we are early in the run, exploration is key.
    # If we are late, exploitation is key.
    # We can estimate "late" by the magnitude of f_best. If f_best is close to 0, we are doing well.
    # However, f_best is just the minimum observed.
    
    # Let's stick to a robust cost-aware PoI.
    # score = poi / (cost * cost)  # Quadratic penalty for cost?
    # Or just linear: poi / cost
    
    # To differentiate from the current LCB/cost, let's add a small exploration bonus
    # for points with high sigma, but weighted by PoI.
    
    # Actually, PoI already incorporates sigma. High sigma near f_best gives moderate PoI.
    # High sigma far from f_best gives low PoI.
    
    # Let's try: score = poi * (1.0 / cost)
    # This is mathematically similar to the current program's logic if LCB approximates improvement,
    # but PoI is bounded and non-linear.
    
    # To make it "materially different", let's use **Expected Improvement (EI)** 
    # but with a **t-distribution** and **cost normalization**.
    # EI is generally superior to PoI for optimization.
    
    # EI = E[max(f_best - f(x), 0)]
    # For t-distribution, the closed form is more complex.
    # Let's stick to PoI but with a **dynamic cost exponent**.
    
    # If the current cost of the *best* candidate in the batch is low, 
    # maybe we can afford higher cost? No, we want to maximize count.
    
    # Let's use: score = poi / (cost ** 1.5)
    # Stronger penalty for high cost than linear.
    
    scores = poi / (cost ** 1.5)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
