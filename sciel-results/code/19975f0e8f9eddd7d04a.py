import numpy as np
from scipy.stats import norm
import time

# Global state to track time budget and potentially cache parameters if needed
# The function is called once per BO step. We need to ensure we don't exceed the 330s limit.
# Since the function is stateless in the contract but we can use module-level state,
# we will track the start time of the first call to enforce a global deadline.
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Computes a cost-aware acquisition score using UCB with a dynamic cost penalty.
    
    Parameters:
    mu (np.ndarray): Mean of GP posterior, shape (2000,)
    sigma (np.ndarray): Std dev of GP posterior, shape (2000,)
    f_best (float): Best observed value so far (minimum)
    cost (np.ndarray): Cost of evaluating each candidate, shape (2000,)
    
    Returns:
    np.ndarray: Acquisition scores, shape (2000,), higher is better.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: The evaluator kills after 330s. 
    # We should return immediately if we are close to the limit to avoid timeout penalty.
    # However, this function is called *during* the loop. If we are close to 330s, 
    # the loop might be in the final iterations. 
    # A safe margin is 5 seconds. If we exceed 325s, we return zeros (or best guess) 
    # to ensure we don't crash, though the loop logic usually handles termination.
    # Since we can't control the loop, we just ensure this function is fast.
    # 2000 points is very fast for numpy.
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Sanitize inputs
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # Strategy: Cost-Aware UCB
    # Standard UCB: mu - kappa * sigma (for minimization, we want low mu and high sigma exploration)
    # Wait, for minimization:
    # We want to find low values.
    # UCB for minimization is often: mu - kappa * sigma (explores where lower bound is low)
    # Or we can use EI which is robust.
    # The current program uses EI / cost.
    # The inspiration suggests "Reducing the cost penalty bias helps avoid the cheaper local minimum".
    # Actually, the global minimum of Three Hump Camel is at (0, 0) with f=0.
    # The "cheaper" local minimum?
    # Cost is 1 + mean(x_scaled). 
    # x in [-5, 5]. Scaled to [0, 1]: (x+5)/10.
    # At (0,0), scaled is (0.5, 0.5), mean is 0.5. Cost = 1.5.
    # At (-5, -5), scaled is (0, 0), mean is 0. Cost = 1.0.
    # At (5, 5), scaled is (1, 1), mean is 1. Cost = 2.0.
    # The function value at (-5, -5) is high (far from min).
    # The function value at (0,0) is 0.
    # The function value at (-1.0227, 0) is approx 0? No, min is at (0,0).
    # There is a local min? Three hump camel:
    # f(x,y) = 2x^2 - 1.05x^4 + x^6/6 + xy + y^2
    # Global min at (0,0) with f=0.
    # There are two local minima?
    # Actually, the "three humps" refer to the shape.
    # Let's check the cost bias.
    # If we divide by cost, we penalize high-cost points.
    # High cost = points near (5,5) or generally high x,y.
    # Low cost = points near (-5,-5).
    # If the global min is at (0,0), cost is 1.5.
    # If we penalize cost too heavily, we might avoid (0,0) if there's a point with lower cost and slightly higher f?
    # Or maybe the "cheaper local minimum" refers to a point with lower cost than the global min?
    # If we use EI / cost, and cost varies from 1 to 2, the impact is moderate.
    
    # Alternative: Use UCB with a cost term.
    # Score = (mu - kappa * sigma) / (cost^alpha)
    # Or Score = (mu - kappa * sigma) - beta * cost
    
    # Let's stick to the EI family but refine the cost handling.
    # The directive says: "KEEP this design and investigate improvements within its family."
    # Current design: EI / cost.
    # Improvement: 
    # 1. Use a softer cost penalty. Instead of dividing by cost, maybe subtract a scaled cost?
    #    But dividing is standard for "cost-aware".
    # 2. The inspiration says "Reducing the cost penalty bias helps...".
    #    This implies the current penalty (division) might be too strong or biased towards cheap (bad) points?
    #    If cheap points are bad (high f), and we divide by low cost (1.0), the score becomes EI / 1.0 = EI.
    #    If good points (low f) have cost 1.5, score is EI / 1.5.
    #    So cheap bad points get a higher score relative to their EI than expensive good points.
    #    To fix this, we should *increase* the penalty on cheap points? No, that would make them even more attractive if EI is high?
    #    Wait. If EI is low for bad points, dividing by 1.0 doesn't help.
    #    If EI is high for bad points (due to high sigma), dividing by 1.0 keeps it high.
    #    If EI is high for good points, dividing by 1.5 reduces it.
    #    So division *penalizes* good points if they are expensive.
    #    The inspiration says "Reducing the cost penalty bias helps avoid the cheaper local minimum in favor of the more expensive global minimum".
    #    This means the current program (EI/cost) is *picking* the cheaper local minimum?
    #    Why would it pick the cheaper one?
    #    Maybe the cheaper one has higher EI initially?
    #    If we reduce the penalty, we make the expensive global min more competitive.
    #    "Reduce cost penalty" -> Make cost matter less.
    #    E.g., EI / (cost^0.5) or EI - cost?
    #    If we use EI / cost^0.5, the difference between cost 1.0 and 1.5 is smaller.
    #    sqrt(1.0)=1, sqrt(1.5)=1.22. Ratio 1.22.
    #    Cost ratio 1.5.
    #    So sqrt reduces the bias.
    
    # Let's try EI / (cost^0.5)
    
    z = (f_best - mu) / sigma_safe
    z_clipped = np.clip(z, -50, 50)
    cdf_z = norm.cdf(z_clipped)
    pdf_z = norm.pdf(z_clipped)
    
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    ei = np.maximum(ei, 0.0)
    
    # Soften the cost penalty
    # Original: ei / cost
    # New: ei / (cost ** 0.5)
    cost_penalized = np.power(cost_safe, 0.5)
    
    scores = ei / cost_penalized
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
