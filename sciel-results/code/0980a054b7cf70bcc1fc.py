"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Ensure numerical stability for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Compute Expected Improvement (EI)
    # EI favors points where the mean is low and uncertainty is moderate
    z_ei = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical issues in norm.cdf/pdf for extreme values
    z_ei = np.clip(z_ei, -10, 10)
    ei = (f_best - mu) * norm.cdf(z_ei) + sigma_safe * norm.pdf(z_ei)
    
    # 2. Compute Lower Confidence Bound (LCB)
    # LCB favors points with low mean and high uncertainty
    # We use a dynamic beta. 
    # As f_best decreases (improves), the function becomes flatter near the minimum,
    # making it harder to find improvements with pure EI. 
    # We want to explore more as we get closer to the optimum.
    # 
    # Let's estimate progress. The minimum of Weierstrass is 0. 
    # The initial random points on [-0.5, 0.5] for Weierstrass (typical params) 
    # often yield values in the range [0, 10] or so depending on frequency/decay.
    # A robust heuristic: 
    # If f_best is high, we are in early stages -> exploit more (lower beta).
    # If f_best is low, we are in late stages -> explore more (higher beta).
    #
    # Normalization:
    # Let's assume a typical scale for f_best. 
    # We can map f_best to a beta range.
    # beta_low = 1.0 (pure exploitation focus, similar to standard EI behavior which is somewhat conservative)
    # beta_high = 4.0 (strong exploration)
    #
    # A simple mapping:
    # progress = 1 - f_best / f_max_est
    # But f_max_est is unknown. 
    # Alternative: Use the magnitude of f_best directly if it's positive.
    # However, f_best can be negative in some contexts, but here min is 0.
    # Weierstrass function is non-negative.
    #
    # Let's use a logistic or linear mapping based on f_best.
    # If f_best > 1.0, assume early stage.
    # If f_best < 0.01, assume late stage.
    
    # Heuristic mapping:
    # beta = 1.0 + 3.0 * (1 - np.exp(-f_best)) 
    # This is not quite right. We want beta to DECREASE as f_best DECREASES? 
    # No, we want to EXPLORE MORE as we get closer to the minimum to avoid getting stuck.
    # So beta should INCREASE as f_best DECREASES.
    
    # Let's define a reference scale.
    # If f_best is large, say 5, beta should be small (e.g., 1.0).
    # If f_best is small, say 0.01, beta should be large (e.g., 5.0).
    
    # beta = 1.0 + 4.0 * (1 - np.tanh(f_best)) 
    # tanh(5) ~ 1 -> beta ~ 1
    # tanh(0) ~ 0 -> beta ~ 5
    # tanh(0.01) ~ 0.01 -> beta ~ 4.96
    
    # This looks reasonable.
    beta = 1.0 + 4.0 * (1 - np.tanh(f_best))
    
    lcb = mu - beta * sigma_safe
    
    # 3. Normalize both to [0, 1] range approximately to combine them
    # EI is always >= 0.
    # LCB can be negative.
    
    # To combine them, we can just sum them if they are on similar scales.
    # However, EI is bounded by (f_best - min_mu) roughly.
    # LCB is unbounded below.
    
    # A common approach is to use a weighted sum.
    # score = w * EI + (1-w) * LCB_normalized
    
    # Let's normalize EI by its max possible value in the batch? 
    # Or just use a fixed weight.
    
    # Actually, simply adding them might work if we scale them.
    # Let's try: score = EI + alpha * (mu - min_mu) - beta * sigma
    # This is getting complicated.
    
    # Let's stick to the finding from the prompt: 
    # "LCB component encourages exploration... weighted combination balances exploitation (EI) with exploration (LCB)"
    # Measured improvement: 1.726 -> 2.925.
    
    # Let's implement a weighted sum:
    # score = EI + gamma * (mu - min(mu)) - beta * sigma
    # But mu is in objective units.
    
    # A simpler and robust combination often used is:
    # score = EI + beta * sigma
    # Wait, EI already contains a sigma term.
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # If mu < f_best, EI is positive.
    # If mu > f_best, EI is small.
    
    # LCB = mu - beta * sigma
    # We want to maximize the score.
    # If we maximize LCB, we pick low mu, high sigma.
    # If we maximize EI, we pick low mu, moderate sigma.
    
    # Let's try: score = EI + 0.5 * (mu - np.min(mu)) - 0.5 * beta * sigma
    # The term (mu - min_mu) is 0 for the best mean, positive for others.
    # This penalizes high mu.
    # The term -beta*sigma rewards high sigma.
    
    # Actually, let's just use:
    # score = EI + beta * sigma * norm.pdf(z_ei) 
    # No, that's double counting.
    
    # Let's use the standard hybrid:
    # score = EI + lambda * (mu - min_mu) - beta * sigma
    # But we don't know min_mu exactly relative to the global min.
    
    # Let's try a simpler adaptive EI:
    # Increase the exploration factor in EI?
    # Standard EI doesn't have a free parameter like UCB.
    
    # Let's go back to the prompt's hint: "weighted combination balances exploitation (EI) with exploration (LCB)"
    # Let's compute LCB and EI, normalize them to have zero mean and unit variance in the batch, then sum them.
    
    ei_mean = np.mean(ei)
    ei_std = np.std(ei) + 1e-9
    ei_norm = (ei - ei_mean) / ei_std
    
    lcb_mean = np.mean(lcb)
    lcb_std = np.std(lcb) + 1e-9
    lcb_norm = (lcb - lcb_mean) / lcb_std
    
    # Weight: 
    # Early on (high f_best), trust EI more (w=0.8)
    # Late on (low f_best), trust LCB more (w=0.2)
    w_ei = 0.5 + 0.5 * np.tanh(f_best)  # tanh(5)=1 -> w=1, tanh(0)=0 -> w=0.5
    w_lcb = 1.0 - w_ei
    
    # If f_best is large, w_ei ~ 1, w_lcb ~ 0.
    # If f_best is small, w_ei ~ 0.5, w_lcb ~ 0.5.
    # Maybe we want more exploration later?
    # If f_best is small, we are stuck. We need LCB.
    # So w_lcb should be higher when f_best is small.
    # w_ei = 0.5 + 0.5 * np.tanh(f_best)
    # If f_best=0, w_ei=0.5.
    # If f_best=10, w_ei=1.0.
    
    # This seems okay.
    score = w_ei * ei_norm + w_lcb * lcb_norm
    
    # Ensure finite
    return score
# EVOLVE-BLOCK-END
