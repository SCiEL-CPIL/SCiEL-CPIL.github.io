"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard: 330s wall-clock limit, return early if close
    t0 = time.time()
    deadline = t0 + 325  # 5s safety margin
    
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure EI is non-negative (numerical noise can make it slightly negative)
    ei = np.maximum(ei, 1e-12)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-8)
    
    # Strategy: Log-Expected-Improvement (Log-EI)
    # This acquisition function maximizes log(EI) - lambda * log(cost)
    # It favors points with high efficiency (improvement per log-cost unit)
    # lambda controls the trade-off. Since cost is in [1, 2], log(cost) is small.
    # A standard choice is lambda=1.
    
    # Alternative: Cost-aware EI with a different normalization
    # The current program uses EI / cost.
    # Log-EI is: log(EI) - log(cost) = log(EI / cost)
    # This is monotonic with EI/cost, so it ranks points identically to EI/cost?
    # No, log is monotonic, so argmax(log(EI/cost)) == argmax(EI/cost).
    # So Log-EI with lambda=1 is equivalent to EI/cost in terms of ranking.
    # This is NOT a different strategy in terms of ranking.
    
    # To make it materially different, I should use a different functional form.
    # Perhaps: EI * (1 / (1 + cost))? Or EI - beta * cost?
    # Or use UCB: mu - beta * sigma, then divide by cost?
    # The inspiration says blending EI and UCB helps.
    
    # Let's try a hybrid:
    # Score = alpha * (EI / cost) + (1 - alpha) * ((f_best - mu - beta * sigma) / cost)
    # This combines exploitation (EI) and exploration (UCB-like) with cost awareness.
    
    alpha = 0.7  # Weight for EI (exploitation)
    beta = 2.0   # Weight for uncertainty in UCB term
    
    # UCB term (maximization of -f, so we use f_best - mu - beta*sigma)
    # Actually, for minimization, UCB is usually mu - beta * sigma (if maximizing).
    # For minimization, we want low mu and high sigma (uncertainty).
    # So "UCB" for minimization is: mu - beta * sigma? No, that's lower bound.
    # Upper Confidence Bound for minimization is often: mu + beta * sigma? No.
    # Let's think: We want to find the minimum.
    # If sigma is high, there's a chance f(x) is very low.
    # So we want to explore high sigma regions.
    # A common acquisition for minimization is: mu - beta * sigma (this is the lower bound, not UCB).
    # Wait, UCB is for maximization. For minimization, it's LCB (Lower Confidence Bound).
    # LCB = mu - beta * sigma.
    # We want to maximize the acquisition function.
    # So we want to maximize (f_best - LCB) ? Or just maximize -LCB?
    # If we use EI, it already accounts for this.
    # Let's use a simple exploration term: -mu + beta * sigma?
    # If mu is low (good), -mu is high. If sigma is high (uncertain), beta*sigma is high.
    # So Score_explore = -mu + beta * sigma.
    
    # Cost-aware version:
    # score_ei = ei / cost_safe
    # score_explore = (-mu + beta * sigma) / cost_safe
    
    # Normalize the terms to have similar scales?
    # mu and f_best are in objective units.
    # Let's just blend them.
    
    score_ei = ei / cost_safe
    score_explore = (-mu + beta * sigma) / cost_safe
    
    # Combine
    score = alpha * score_ei + (1 - alpha) * score_explore
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
