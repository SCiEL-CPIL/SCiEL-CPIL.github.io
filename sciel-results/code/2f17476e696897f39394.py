"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global deadline guard
_START_TIME = time.time()
_LIMIT = 285.0  # Safety margin below 300s


def acquisition(mu, sigma, f_best, cost):
    # Deadline check
    if time.time() - _START_TIME > _LIMIT:
        return np.zeros_like(mu)

    # Ensure finite inputs
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Sanitize inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Compute Expected Improvement for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Clip z to prevent numerical underflow/overflow in pdf/cdf
    z = np.clip(z, -10, 10)

    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Knowledge Gradient (KG) approximation
    # The KG is the expected reduction in the best observed value.
    # For a single point, KG(x) = E[max(f_best, f(x)) - f_best]
    # This is related to EI but emphasizes the probability of finding a new minimum.
    # A common approximation for KG in the context of GP is:
    # KG(x) ~ EI(x) + small term related to the probability that x is the new min.
    # However, a more direct structural change is to use a "Probability of Improvement" (PI)
    # component or a weighted combination that better handles flat regions.
    
    # Let's try a hybrid: EI + weighted PI
    # PI(x) = P(f(x) < f_best) = Phi(z)
    pi = norm.cdf(z)
    
    # The current program uses EI + exploitation_bias.
    # The inspiration 1 uses EI + exploitation_bias + uncertainty_bonus.
    # The directive asks for a structural change: KG approximation.
    
    # A simple KG approximation for minimization is:
    # KG(x) = E[ max(f_best, f(x)) - f_best ]
    # Note that EI(x) = E[ max(f_best - f(x), 0) ] = E[ f_best - min(f_best, f(x)) ]
    # So EI(x) = f_best - E[min(f_best, f(x))]
    # KG(x) = E[max(f_best, f(x))] - f_best = E[f(x)] - E[min(f_best, f(x))] = mu(x) - (f_best - EI(x))
    # KG(x) = mu(x) - f_best + EI(x)
    
    kg_approx = mu - f_best + ei
    
    # This KG approximation can be negative if mu > f_best significantly.
    # We want to maximize the expected reduction in f_best.
    # If mu is high, the chance of improvement is low, but if it is low, the reduction is high.
    # The term (mu - f_best) is negative when mu < f_best, which is good.
    # EI is always positive.
    
    # Let's check the behavior:
    # If mu << f_best, z is large positive, EI ~ f_best - mu, KG ~ mu - f_best + f_best - mu = 0?
    # Wait.
    # If mu << f_best, f(x) is likely < f_best.
    # E[min(f_best, f(x))] ~ E[f(x)] = mu.
    # EI = f_best - mu.
    # KG = mu - f_best + EI = mu - f_best + f_best - mu = 0.
    # This suggests my derivation is slightly off or I'm misinterpreting.
    
    # Let's re-evaluate.
    # KG(x) = E[ f_best - min(f_best, f(x)) ] = EI(x).
    # Actually, the standard Knowledge Gradient for a single point IS the Expected Improvement.
    # KG(x) = E[ max(f_best, f(x)) - f_best ] is NOT the standard definition.
    # Standard KG: Expected reduction in the best value.
    # If we evaluate x, the new best is min(f_best, f(x)).
    # Reduction = f_best - min(f_best, f(x)) = max(f_best - f(x), 0).
    # So KG(x) = E[max(f_best - f(x), 0)] = EI(x).
    
    # So EI IS the Knowledge Gradient for a single point.
    # Therefore, the "structural change" to KG doesn't change the core term if we stick to single-point evaluation.
    # However, the directive mentions "decoupling the discovery probability from the value magnitude".
    # This suggests using PI (Probability of Improvement) or a different weighting.
    
    # Let's look at Inspiration 1's "uncertainty bonus". It rewards moderate sigma.
    # Let's look at the "exploitation bias". It penalizes high mu.
    
    # Perhaps the structural change is to use a different acquisition function entirely, like UCB (Upper Confidence Bound) or PI.
    # UCB = mu - beta * sigma (for minimization, we want low mu and high uncertainty).
    # Wait, for minimization, UCB is mu - beta * sigma? No, that would favor high sigma and low mu.
    # Usually UCB for minimization is mu - beta * sigma.
    
    # Let's try a combination of EI and PI with a cost-aware scaling.
    # Or perhaps the "structural change" is to move away from EI and use a different metric.
    
    # Given the stagnation, let's try a different approach: 
    # Use a "Cost-Effective" acquisition that balances exploration and exploitation differently.
    # The current program uses (EI - 0.1*mu) / cost.
    
    # Let's try: Score = (EI * w1 + PI * w2 - 0.1 * mu * w3) / cost
    # Where w1, w2, w3 are weights.
    
    # Alternatively, let's implement the "Uncertainty Bonus" from Inspiration 1 but with a structural twist.
    # The uncertainty bonus in Insp 1 is:
    # mu_range = np.max(mu) - np.min(mu) + 1e-10
    # target_sigma = 0.1 * mu_range
    # uncertainty_bonus = 0.05 * target_sigma * np.exp(-0.5 * ((sigma - target_sigma) / (0.5 * target_sigma + 1e-10))**2)
    
    # This rewards sigma close to 0.1 * mu_range.
    
    # Let's combine the current program's exploitation bias with Inspiration 1's uncertainty bonus.
    # This is a "refinement" of the current program using the "structural idea" (uncertainty shaping) of Insp 1.
    
    # To make it more structural, let's also adjust the cost normalization.
    # Instead of dividing by cost, we might use a power law: cost ** -alpha.
    # But the task says "cost-aware", and division is standard.
    
    # Let's stick to the combination: EI + Exploitation Bias + Uncertainty Bonus, all divided by cost.
    # This takes the structural idea of "shaping the acquisition landscape with an uncertainty bonus" from Insp 1
    # and the "exploitation bias" from the current program.
    
    mu_range = np.max(mu) - np.min(mu) + 1e-10
    target_sigma = 0.1 * mu_range
    uncertainty_bonus = 0.05 * target_sigma * np.exp(-0.5 * ((sigma - target_sigma) / (0.5 * target_sigma + 1e-10))**2)
    
    exploitation_bias = -0.1 * mu
    
    score = (ei + exploitation_bias + uncertainty_bonus) / cost
    
    # Ensure finite output
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
