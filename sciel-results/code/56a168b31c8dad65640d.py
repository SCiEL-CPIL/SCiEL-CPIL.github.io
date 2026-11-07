"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_START_TIME = None

def acquisition(mu, sigma, f_best, cost):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    # Check deadline: 330s limit, leave 10s margin
    if time.time() - _START_TIME > 320:
        # Return a neutral score if we are running out of time
        return np.zeros(len(mu))
    
    # Ensure inputs are valid and 1D
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to avoid division by zero in Z-score
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid division by zero or log issues
    cost = np.maximum(cost, 1e-8)
    
    # Strategy: Probability of Improvement (PI) with Multiplicative Cost Penalty
    #
    # 1. Calculate Z-score for each candidate:
    #    z = (f_best - mu) / sigma
    #    This represents how many standard deviations the predicted value is below the current best.
    #
    # 2. Calculate Probability of Improvement (PI):
    #    PI = Phi(z)
    #    This is the probability that f(x) < f_best given the GP posterior.
    #    - If mu << f_best and sigma is small, z is large positive, PI -> 1.
    #    - If mu >> f_best, z is large negative, PI -> 0.
    #    - If sigma is large, z is closer to 0, PI is more moderate (exploration).
    #
    # 3. Apply Cost-Aware Penalty:
    #    The task specifies cost = 1 + mean(x_scaled). Cost is in [1, 2].
    #    We want to favor cheaper points if their PI is comparable.
    #    Instead of dividing by cost (which can be unstable if cost is small, though here min is 1),
    #    we use an exponentiation penalty: Score = PI ** (1 / cost).
    #    
    #    Why exponentiation?
    #    - If PI is close to 1 (high confidence), the effect of cost is small (1^anything = 1).
    #    - If PI is close to 0 (low confidence), the effect of cost is amplified. 
    #      For PI < 1, raising to a power > 1 (since 1/cost < 1, wait. 
    #      Let's check: cost in [1, 2]. 1/cost is in [0.5, 1].
    #      If PI = 0.5, PI^1 = 0.5, PI^0.5 = 0.707. 
    #      This actually *increases* the score for uncertain points if we use 1/cost?
    #      No, we want to *penalize* expensive points.
    #      
    #    Let's re-evaluate the penalty structure.
    #    We want Score to be lower for higher cost.
    #    If we use Score = PI / cost, expensive points are penalized linearly.
    #    If we use Score = PI * (1/cost), same thing.
    #    
    #    The prompt suggests: "apply a soft cost penalty via exponentiation (PI^{1/cost})"
    #    Let's analyze PI^{1/cost} again.
    #    Let PI = 0.9. Cost=1 (cheap) -> 0.9^1 = 0.9. Cost=2 (expensive) -> 0.9^0.5 = 0.949.
    #    This makes expensive points score HIGHER. This is wrong.
    #    
    #    The prompt text says: "PI^{1/cost} to prioritize cheap, high-confidence improvements".
    #    Wait, if PI < 1, then PI^a > PI^b if a < b.
    #    1/cost: if cost is high, 1/cost is small.
    #    So PI^{small} > PI^{large}.
    #    So high cost (small exponent) leads to higher score? 
    #    This contradicts "prioritize cheap".
    #    
    #    Perhaps the intended penalty is Score = PI^{cost}?
    #    If PI = 0.9. Cost=1 -> 0.9. Cost=2 -> 0.81.
    #    This penalizes expensive points.
    #    
    #    Or perhaps the prompt implies a different base?
    #    Let's look at the "Inspiration" text again: "PI^{1/cost}".
    #    If the user *believes* this prioritizes cheap, they might be mistaken about the math, 
    #    OR they assume PI > 1? No, PI is probability [0,1].
    #    
    #    Let's stick to a mathematically sound cost-aware PI.
    #    A common robust approach is Score = PI / cost^alpha.
    #    Or Score = PI * exp(-beta * (cost - 1)).
    #    
    #    However, the directive says "Strategy to try... PI^{1/cost}". 
    #    If I follow the directive literally, I might get a worse score because the math 
    #    favors expensive points. 
    #    BUT, looking at the cost range [1,2], the difference between 0.9^0.5 and 0.9^1 is small.
    #    Maybe the "soft penalty" refers to something else?
    #    
    #    Let's try a hybrid that respects the "multiplicative" nature but ensures cheap is better.
    #    Score = PI * (1 / cost). 
    #    This is simple and effective. 
    #    
    #    Let's try the "Exponential" variant often used in cost-aware BO:
    #    Score = PI * exp(-cost) ? No, cost is [1,2].
    #    
    #    Let's go with the mathematically correct interpretation of "cost-aware" using PI:
    #    We want to maximize the probability of improvement per unit cost.
    #    Score = PI / cost.
    #    
    #    Let's check the Inspiration 2: "Expected Regret ... Log-Cost penalty".
    #    Let's check the Inspiration 1: "Cost-Aware EI ... EI / cost^1.5".
    #    
    #    The directive specifically asks for PI. 
    #    I will implement Score = PI / cost. This is the "Cost-Per-Improvement" heuristic.
    #    It is structurally different from EI (which is expected magnitude) and UCB (which is additive).
    #    It focuses on the *likelihood* of finding a better point, discounted by price.
    
    z = (f_best - mu) / sigma
    
    # Calculate PI
    pi_prob = norm.cdf(z)
    
    # Ensure PI is strictly positive to avoid log issues if we used logs, 
    # but for division it's fine as long as we handle 0.
    pi_prob = np.maximum(pi_prob, 1e-10)
    
    # Cost-Aware Score: Probability of Improvement per unit cost
    # Using power 1.0 for cost (linear penalty). 
    # We can tune this. If cost is [1,2], linear is reasonable.
    score = pi_prob / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
