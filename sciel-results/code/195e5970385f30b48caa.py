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
    
    # Clip sigma to avoid division by zero
    sigma = np.maximum(sigma, 1e-8)
    # Clip cost to avoid log(0) or division by zero
    cost = np.maximum(cost, 1e-8)
    
    n_candidates = len(mu)
    
    # Strategy: Cost-Effective Expected Hypervolume Improvement (CE-EHI)
    # 
    # The standard EI focuses on improving the best observed value.
    # EHI focuses on improving the Pareto front of (cost, objective) pairs.
    # Since we have a fixed budget, we want to maximize the area under the 
    # cost-objective curve (hypervolume) relative to a reference point.
    #
    # Approximation:
    # 1. Estimate the current "front" using the best observed value f_best.
    #    In a single-objective BO with cost, the relevant metric is often 
    #    the best value achievable for a given cost, or simply the best value 
    #    and the cost spent to get it.
    #    However, the task gives us `f_best` (lowest observed value) and `cost` 
    #    (cost of each candidate). It does NOT give us the history of 
    #    (cost, value) pairs. 
    #    Wait, the task says: "f_best is the lowest observed value."
    #    It does not provide the cost of the point that achieved f_best.
    #    This makes a true Pareto front construction difficult without history.
    #    
    #    Alternative Interpretation of "Hypervolume" in this context:
    #    Often in cost-aware BO, the objective is to minimize f(x) subject to 
    #    budget. 
    #    If we treat this as a bi-objective problem (minimize f, minimize cost),
    #    and we don't have the full history, we can approximate the reference point.
    #    
    #    Let's look at the "Inspiration 2" which used Log-Cost penalty.
    #    The directive asks for CE-EHI.
    #    
    #    Since we only have `f_best` and the current candidate `cost`, we can
    #    approximate the "expected improvement in hypervolume" by considering
    #    how much a new point (c_new, f_new) would improve the area relative to
    #    a reference point (c_ref, f_ref).
    #    
    #    A common reference point in such settings is (max_cost, f_best) or 
    #    (min_cost, f_best) or a point slightly worse than the best known.
    #    Given we don't know the cost of f_best, let's assume the "current best"
    #    is effectively at some average cost, or we define the reference point
    #    based on the domain limits.
    #    
    #    Let's use a simpler proxy for EHI that captures the "cost-effectiveness":
    #    The expected improvement in the objective function, weighted by the 
    #    "marginal value" of the cost.
    #    
    #    Actually, a robust approximation for CE-EHI when history is limited is:
    #    Score(x) = E[ HV( { (c_x, f_x) } ∪ Front ) ] - HV( Front )
    #    
    #    If we assume the current Front is dominated by the point (c_best, f_best),
    #    and we don't know c_best, we can use a reference point (c_max, f_best).
    #    The hypervolume improvement of a new point (c, f) over the reference 
    #    rectangle [c, c_max] x [f, f_best] is (c_max - c) * (f_best - f) 
    #    if f < f_best and c < c_max.
    #    
    #    Expected Hypervolume Improvement (EHI) for a single point (c, f~N(mu, sigma)):
    #    E[ max(0, (c_max - c) * (f_best - f)) ]
    #    Since c is deterministic for a candidate, this is:
    #    (c_max - c) * E[ max(0, f_best - f) ]
    #    E[ max(0, f_best - f) ] is exactly Expected Improvement (EI).
    #    
    #    So, EHI ≈ (c_max - c) * EI(x)
    #    
    #    Cost-Effective EHI (CE-EHI) divides by the cost:
    #    CE-EHI = (c_max - c) * EI(x) / c
    #    
    #    This is structurally different from simple EI/cost because it includes
    #    the term (c_max - c), which penalizes points that are already expensive 
    #    relative to the max possible cost, and rewards points that are cheap 
    #    (high potential for "area" gain).
    #    
    #    Let's refine the reference point. 
    #    c_max = 2.0 (since cost = 1 + mean(x_scaled), max is 2).
    #    c_min = 1.0.
    #    
    #    Let's try: Score = EI(x) * (c_max - cost) / cost
    #    This effectively weights EI by the "unused budget potential" relative to 
    #    the cost.
    
    # Calculate EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Calculate CE-EHI approximation
    c_max = 2.0
    c_min = 1.0
    
    # (c_max - cost) is the "room" for improvement in cost dimension.
    # Dividing by cost makes it a rate of improvement per unit cost.
    
    # To avoid numerical issues if cost is very close to c_max, 
    # the term (c_max - cost) becomes small, reducing the score.
    # This encourages exploring cheaper regions if EI is comparable.
    
    score = ei * (c_max - cost) / cost
    
    # Add a small exploration term to break ties or encourage diversity
    # in case EI is 0 for all points (e.g., if mu is always > f_best)
    # But EI should be non-zero if there's uncertainty.
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
