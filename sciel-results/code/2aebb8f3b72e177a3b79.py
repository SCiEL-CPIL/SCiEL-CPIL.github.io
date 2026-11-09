import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s safety margin before 330s limit

def acquisition(mu, sigma, f_best, cost):
    # Check time budget
    if time.time() > _DEADLINE:
        # Fallback: return random-ish scores to satisfy interface, 
        # though this shouldn't happen if computation is fast.
        return np.random.rand(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Handle edge cases
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Probability of Improvement (POI)
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    poi = norm.cdf(z)
    
    # 2. Uncertainty Bonus
    # In high dimensions, pure exploitation (EI/POI) can get stuck.
    # We add a term proportional to sigma to encourage exploration.
    # The weight is adaptive: higher when sigma is large relative to the scale of f.
    # However, since f_best is the scale, we can normalize.
    # Let's use a simple heuristic: bonus = alpha * sigma / (f_best + eps)
    # Or simply alpha * sigma. 
    # Given the score is "higher = evaluate next", and we want to balance,
    # let's try a combined score: Score = POI * (f_best - mu + beta * sigma)
    # Actually, let's stick to the inspiration: POI is more conservative.
    # Let's try: Score = POI + gamma * sigma / (1 + sigma)
    # But cost-awareness is key.
    
    # Alternative Strategy: Cost-Adjusted POI with Exploration
    # The current program uses EI/cost.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # POI = Phi(z)
    
    # Let's try a hybrid: 
    # Improvement potential: (f_best - mu)
    # Confidence: POI
    # Exploration: sigma
    
    # A common cost-aware acquisition is qEI or similar, but here we have a single point.
    # Let's try: Score = (POI * (f_best - mu) + 0.1 * sigma) / cost
    # The 0.1 * sigma term ensures that points with high uncertainty get some chance,
    # even if POI is low (i.e., they are currently predicted to be worse).
    
    improvement = np.maximum(f_best - mu, 0.0)
    
    # Weight the exploration term. 
    # If we just use POI/cost, we might ignore areas with high sigma but low POI.
    # If we use EI/cost, we are balanced.
    # The inspiration says POI is better (1.073 vs 1.000).
    # Let's try to enhance POI with a small exploration bonus.
    
    # Normalized exploration: sigma / (1 + sigma) to keep it bounded
    exploration = sigma_safe / (1.0 + sigma_safe)
    
    # Combine: 
    # Base score: POI (probability that it's better)
    # Bonus: Exploration (potential to be much better)
    # Scale: Cost
    
    # Let's try: Score = (POI * (1 + exploration)) / cost
    # This boosts POI in uncertain regions.
    
    scores = (poi * (1.0 + 0.5 * exploration)) / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
