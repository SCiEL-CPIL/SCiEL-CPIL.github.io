"""Acquisition function for Bayesian optimization of Goldstein-Price 2D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Pre-compute constants for speed
# The evaluator calls acquisition with mu and sigma of shape (2000,)
# We need to return 2000 finite scores.

def acquisition(mu, sigma, f_best):
    """
    Computes an acquisition score for each of the 2000 candidates.
    
    Parameters:
    mu: array of shape (2000,), posterior mean
    sigma: array of shape (2000,), posterior std
    f_best: float, best observed value so far
    
    Returns:
    scores: array of shape (2000,), higher is better
    """
    # Safety margin: ensure we don't crash or hang
    # The function is called 25 times per seed (30 total evals, 5 initial, so 25 steps).
    # 3 seeds * 25 steps = 75 calls.
    # Total time limit 330s.
    
    # Strategy: Use a hybrid of Expected Improvement (EI) and Upper Confidence Bound (UCB).
    # The "inspiration" mentioned Adaptive UCB with progress-based beta.
    # However, we don't know the progress (number of evaluations) directly from the signature.
    # We can infer progress from f_best. 
    # Minimum value of Goldstein-Price is 3.
    # Initial uniform random points on [-2, 2]^2 will likely have values around 10-30.
    # As f_best approaches 3, we should shift to exploitation (EI).
    # When f_best is high (far from 3), we should explore more (UCB).
    
    # Let's estimate the "progress" or "exploration budget".
    # Let range = f_best - 3.0
    # If range is large, explore. If range is small, exploit.
    
    # A robust approach is to use a weighted sum of EI and UCB.
    # Or simply use a UCB with an adaptive beta.
    
    # Let's try an adaptive UCB.
    # Score = mu - beta * sigma  (since we are minimizing, we want low mu and high sigma)
    # Wait, for minimization:
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z) where z = (f_best - mu) / sigma
    # UCB = mu - beta * sigma
    
    # The prompt says: "Adaptive UCB with progress-based beta increases exploration early... then shifts to exploitation"
    # This implies beta starts high and decreases.
    
    # How to estimate progress?
    # f_best starts high (e.g., 20) and goes to 3.
    # Let progress = 1 - (f_best - 3) / (max_initial - 3)
    # We don't know max_initial. But typical uniform on [-2,2]^2 for GP2:
    # GP2(x,y) = (1+2(x+y+1)^2(5(x+y+1)-x)^2 + 3(x+y+1)) * (30-10x+y)^2 * (2-3x+y^2/5)
    # At (0,0): (1+2(1)^2(5-0)^2+3(1)) * (30-0+0)^2 * (2-0+0) = (1+50+3)*900*2 = 54*1800 = 97200? No.
    # Let's check values.
    # x=0, y=0:
    # term1 = 1 + 2*(1)^2*(5-0)^2 + 3*(1) = 1 + 50 + 3 = 54
    # term2 = (30 - 0 + 0)^2 = 900
    # term3 = 2 - 0 + 0 = 2
    # val = 54 * 900 * 2 = 97200.
    
    # Wait, the standard Goldstein-Price function is:
    # f(x,y) = [1 + (x+y+1)^2 * (19 - 14x + 3x^2 - 14y + 6xy + 3y^2)] * [30 + (2x - 3y)^2 * (18 - 32x + 12x^2 + 48y - 36xy + 27y^2)]
    
    # Let's re-read the task description. "goldstein price 2d on [-2, 2]^2, minimum value 3".
    # The standard minimum is 3 at (0, -1).
    # Let's check f(0, -1):
    # x=0, y=-1
    # x+y+1 = 0
    # term1 = 1 + 0 = 1
    # 2x-3y = 3
    # 18 - 0 + 0 - 48 + 0 - 27 = 18 - 48 - 27 = -57
    # term2 = 30 + 9 * (-57) = 30 - 513 = -483.
    # This doesn't seem right. The standard function has a minimum of 3.
    # Let's assume the evaluator uses the standard definition where min is 3.
    
    # Regardless, f_best will start high and drop to 3.
    # Let's define a beta that depends on f_best.
    # Let span = f_best - 3.0
    # If span is large, beta is large.
    # Let beta = 1.0 + 2.0 * (span / 10.0)  # Cap the influence
    # Or simply use a fixed UCB that works well?
    # The current program uses EI. EI is often better than UCB for BO.
    # But the "inspiration" says Adaptive UCB improved island 1 to 1.308.
    # The current program is EI with score 1.0.
    # Let's try to combine EI and UCB or use a different acquisition.
    
    # Another option: Lower Confidence Bound (LCB) for minimization is effectively UCB for maximization.
    # LCB = mu - beta * sigma
    
    # Let's try a hybrid:
    # Score = EI + alpha * (mu - beta * sigma) ? No, they have different scales.
    
    # Let's stick to the insight: Adaptive UCB.
    # We need to estimate how many evaluations have been done or how close we are.
    # Since we don't have the count, we use f_best.
    # Assume initial f_best is around 10-50.
    # Let's map f_best to a beta.
    # beta = 1.5 when f_best is high, beta = 0.5 when f_best is close to 3.
    
    # Let's use a smooth transition.
    # Let t = (f_best - 3.0) / (max(f_best - 3.0, 1.0))  # Normalized progress?
    # Actually, just use f_best directly.
    
    # If f_best > 10, explore more.
    # beta = 1.0 + 0.5 * (f_best - 3.0) / 10.0  # Linear increase with distance
    # But if f_best is very large, beta could be huge.
    # Let's cap it.
    # beta = 1.5 + 0.5 * np.tanh((f_best - 3.0) / 5.0)
    # When f_best=3, beta=1.5.
    # When f_best=13, beta=1.5+0.5*0.99=2.0.
    # When f_best=33, beta=1.5+0.5*1=2.0.
    
    # Maybe a simpler approach:
    # Use EI but with a small noise addition to encourage exploration?
    # No, let's try the Adaptive UCB as suggested by the "inspiration".
    
    # Score = mu - beta * sigma
    # We want to minimize, so we pick the one with the lowest score?
    # Wait, the task says: "Return 2000 finite scores, higher meaning evaluate next".
    # So we must return scores where HIGHER is better.
    # For minimization, the "value" to maximize is the negative of the objective, or the "improvement".
    # EI returns positive values for improvement.
    # UCB (mu - beta*sigma) returns lower values for promising points (low mu, high sigma).
    # So to use UCB in a "higher is better" framework, we should return - (mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's verify this.
    # If we have two points:
    # A: mu=5, sigma=1
    # B: mu=4, sigma=2
    # f_best=3
    # EI_A: z=(3-5)/1 = -2. EI = (3-5)*Phi(-2) + 1*Phi'(-2) = -2*0.0228 + 0.054 = -0.0456+0.054 = 0.0084
    # EI_B: z=(3-4)/2 = -0.5. EI = (3-4)*Phi(-0.5) + 2*Phi'(-0.5) = -1*0.3085 + 2*0.352 = -0.3085 + 0.704 = 0.3955
    # EI prefers B (more uncertain, lower mean).
    
    # UCB (minimization): score = mu - beta*sigma. We want the MINIMUM of this.
    # To convert to "higher is better", we return -score = -mu + beta*sigma.
    # A: -5 + 1.5*1 = -3.5
    # B: -4 + 1.5*2 = -4 + 3 = -1
    # Higher is better -> B (-1) > A (-3.5).
    # So B is preferred.
    
    # So, if we use Adaptive UCB, we return:
    # scores = -mu + beta * sigma
    
    # Let's refine beta.
    # If we use a constant beta, say 1.5, it might be too exploratory or not enough.
    # The "inspiration" says "progress-based beta".
    # Let's estimate progress based on f_best.
    # Let's assume the initial f_best is around 15 (average of uniform on [-2,2]^2 for GP2 is likely high).
    # Let's define:
    # beta = 1.0 + 1.0 * np.exp(-(f_best - 3.0) / 5.0)
    # If f_best=3, beta=2.0 (high exploration? No, we want exploitation at the end).
    # Wait. "increases exploration early... then shifts to exploitation".
    # Exploration = High Beta. Exploitation = Low Beta.
    # So Early (High f_best) -> High Beta.
    # Late (Low f_best) -> Low Beta.
    
    # Let's try:
    # beta = 0.5 + 1.5 * np.exp(-(f_best - 3.0) / 10.0)
    # If f_best=3, beta=0.5+1.5=2.0? No, exp(0)=1.
    # We want beta to be HIGH when f_best is HIGH.
    # So beta should DECREASE as f_best decreases.
    # beta = 0.5 + 1.5 * (f_best - 3.0) / 10.0  # Linear?
    # If f_best=13, beta=2.0.
    # If f_best=3, beta=0.5.
    # This seems reasonable.
    # Let's cap it to avoid huge values if f_best is very large.
    # beta = 0.5 + 1.5 * np.tanh((f_best - 3.0) / 10.0)
    # f_best=3 -> 0.5
    # f_best=13 -> 0.5 + 1.5*0.76 = 1.64
    # f_best=33 -> 0.5 + 1.5*1.0 = 2.0
    
    # Let's try this Adaptive UCB.
    
    # One more consideration: The current program uses EI.
    # EI is known to be robust.
    # Maybe a mixture?
    # Score = 0.5 * EI + 0.5 * (-mu + beta * sigma) ?
    # Scales might differ. EI is in objective units. UCB is in objective units.
    # Let's just try the Adaptive UCB first, as it was an improvement in the other island.
    
    # Also, handle the case where sigma is very small to avoid division by zero in EI if we were using it.
    # For UCB, we just multiply.
    
    # Let's also consider that the GP is standardized.
    # "standardized targets" means the GP output is in standard deviation units?
    # "mu and sigma ... are in objective units."
    # So no scaling needed for the acquisition function itself.
    
    # Implementation:
    
    # 1. Calculate beta
    # 2. Calculate scores = -mu + beta * sigma
    # 3. Ensure finite
    
    # Let's add a small epsilon to sigma to prevent issues if sigma is 0?
    # If sigma is 0, UCB = -mu. This is pure exploitation.
    # This is fine.
    
    # Let's test the logic.
    # If f_best is very close to 3, beta is small (0.5).
    # Score = -mu + 0.5 * sigma.
    # This favors low mu.
    # If f_best is high, beta is large (2.0).
    # Score = -mu + 2.0 * sigma.
    # This favors high sigma (exploration).
    
    # This matches the "increases exploration early" description.
    
    # Let's write the code.
    
    # Calculate beta
    # Use a smooth function.
    # Let's use the tanh version for stability.
    beta = 0.5 + 1.5 * np.tanh((f_best - 3.0) / 10.0)
    
    # Calculate scores
    scores = -mu + beta * sigma
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
