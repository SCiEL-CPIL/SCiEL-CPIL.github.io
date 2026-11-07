"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
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
    
    # Clip sigma to avoid division by zero or extreme values in sampling
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Strategy: Cost-Aware Thompson Sampling
    #
    # 1. Sample from the GP posterior for each candidate:
    #    y ~ N(mu, sigma)
    #    This simulates the true function value under uncertainty.
    #
    # 2. Cost-Aware Selection:
    #    We want to minimize the sampled value y, but penalize high cost.
    #    We define a score to be maximized by the evaluator.
    #    Score = - (y / cost)
    #    
    #    Why this works:
    #    - If y is low (good), -y is high (good score).
    #    - If cost is low (cheap), y/cost is larger (more positive or less negative), 
    #      so -y/cost is more negative? 
    #      Wait. 
    #      Let y = 10, cost = 1. Score = -10.
    #      Let y = 10, cost = 2. Score = -5.
    #      -5 > -10. So the expensive point gets a HIGHER score. This is WRONG.
    #      We want the cheap point to be better.
    #      
    #    Correction:
    #    We want to maximize "Quality per Cost".
    #    Quality = -y (since lower y is better).
    #    Score = (-y) / cost.
    #    
    #    Check:
    #    y = 10, cost = 1. Score = -10.
    #    y = 10, cost = 2. Score = -5.
    #    -5 > -10. Still wrong? 
    #    Ah, if y is positive, -y is negative.
    #    If y is negative (good), -y is positive.
    #    
    #    Let's look at the sign.
    #    We want to pick the point with the lowest expected/cost value.
    #    Let's define the objective to minimize: J(x) = y(x) / cost(x).
    #    The evaluator picks argmax(Score).
    #    So we should return Score(x) = - J(x) = - y(x) / cost(x).
    #    
    #    Let's re-evaluate the comparison:
    #    Point A: y=10, cost=1. J_A = 10. Score_A = -10.
    #    Point B: y=10, cost=2. J_B = 5. Score_B = -5.
    #    Evaluator picks max(-10, -5) = -5 (Point B).
    #    Point B is more expensive but has the same y. 
    #    Is Point B better? No, Point A is cheaper.
    #    So minimizing J = y/cost is NOT the right objective if we just want cheap points.
    #    Usually, cost-aware BO minimizes y, subject to cost budget, or minimizes y/cost if we want "efficiency".
    #    But if y is the same, we prefer the cheaper one.
    #    So we want to minimize y, and break ties by cost.
    #    Or, we want to minimize y + lambda * cost.
    #    
    #    Let's use Score = - (y + lambda * cost).
    #    This penalizes cost directly.
    #    However, the "efficiency" metric (y/cost) is also common.
    #    Given the task "Cost-aware acquisition", and the fact that total cost is capped,
    #    it is often better to get more evaluations (cheaper) even if they are slightly worse,
    #    to allow the GP to update.
    #    
    #    Let's try Score = -y / cost. 
    #    If y is negative (improvement), -y is positive.
    #    y=-10, cost=1 -> Score=10.
    #    y=-10, cost=2 -> Score=5.
    #    10 > 5. So cheaper point is preferred. Correct.
    #    
    #    If y is positive (worse than best? No, y is absolute value).
    #    If y=10 (bad), cost=1 -> Score=-10.
    #    If y=10 (bad), cost=2 -> Score=-5.
    #    -5 > -10. Expensive point preferred. Incorrect.
    #    
    #    So -y/cost fails when y is positive.
    #    
    #    Let's use Score = -y - lambda * cost.
    #    y=10, cost=1, lambda=1 -> -11.
    #    y=10, cost=2, lambda=1 -> -12.
    #    -11 > -12. Cheaper point preferred. Correct.
    #    
    #    y=-10, cost=1, lambda=1 -> 9.
    #    y=-10, cost=2, lambda=1 -> 8.
    #    9 > 8. Cheaper point preferred. Correct.
    #    
    #    So Score = - (y + lambda * cost) is robust.
    #    We need to choose lambda.
    #    Cost is in [1, 2]. y is in objective units.
    #    Powell 4D values can be large.
    #    If y is ~100, lambda*cost is ~1-2. Cost penalty is negligible.
    #    If y is ~0, lambda*cost is ~1-2. Cost penalty is significant.
    #    This seems reasonable: when we are close to the optimum, we care more about cost.
    #    When we are far, we care more about function value.
    #    
    #    Let's set lambda = 1.0.

    # Sample from posterior
    # Use a fixed seed for reproducibility in tests? No, the task implies stochasticity is okay 
    # as long as it's finite. But for deterministic evaluation in the suite, 
    # maybe we should use a pseudo-random number generator seeded by the iteration?
    # We don't have the iteration count. 
    # Using np.random.randn is fine.
    
    z = np.random.randn(len(mu))
    y = mu + sigma * z
    
    # Cost-aware score: Maximize this
    # We want to minimize y + lambda * cost
    # So we maximize -(y + lambda * cost)
    lambda_cost = 1.0
    score = -(y + lambda_cost * cost)
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
