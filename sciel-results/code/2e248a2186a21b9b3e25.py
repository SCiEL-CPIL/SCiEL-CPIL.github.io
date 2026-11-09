# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Safety deadline guard
    t_start = time.time()
    deadline = t_start + 290.0  # Keep a 40s buffer before the 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)

    if n == 0:
        return np.array([], dtype=np.float64)

    # Prevent division by zero
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)

    # 1. Compute Expected Improvement (EI)
    # EI is a robust baseline for minimization.
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # 2. Compute Lower Confidence Bound (LCB)
    # LCB encourages exploration by subtracting a multiple of sigma.
    # We want to minimize mu - kappa*sigma.
    # So the "score" (higher is better) is -(mu - kappa*sigma) = kappa*sigma - mu.
    # Let's try a few values of kappa.
    # Early on, high kappa is good for exploration. Later, low kappa is good for exploitation.
    # However, we don't know the step number directly, only f_best.
    # A common heuristic is to use a fixed kappa, or one that decays.
    # Since we can't easily decay without step count, let's try a fixed moderate kappa.
    # The Beale function is relatively smooth, so excessive exploration might be wasteful.
    # Let's try kappa = 1.0 and 2.0.
    
    lcb_1 = 1.0 * sigma_safe - mu
    lcb_2 = 2.0 * sigma_safe - mu

    # 3. Strategy: Blend or Select
    # The current program (score 1.946) uses a normalized max of many candidates.
    # The seed program (score 1.0) used pure EI.
    # The "normalized max" approach effectively creates a union of exploration/exploitation regions.
    # However, it might be too greedy or noisy.
    
    # Let's try a different strategy: **Adaptive Blending based on f_best**.
    # If f_best is close to 0 (the global minimum), we should exploit (high EI weight).
    # If f_best is far from 0, we should explore (high LCB weight).
    # But we don't know the scale of "far".
    
    # Alternative Strategy: **Knowledge Gradient (KG) Approximation**
    # KG is theoretically optimal for finite horizons but computationally expensive.
    # A simple approximation is to weight EI by the probability of improvement.
    # This is essentially EI itself.
    
    # Let's try a **Weighted Sum** of EI and LCB.
    # We need to normalize them to comparable scales.
    # EI is always >= 0. LCB can be negative.
    
    # Normalize EI to [0, 1] range? Or just use raw values?
    # Raw EI values depend on the scale of the function.
    # Beale function values range from 0 to ~100+ (max at corners).
    # f_best starts high and goes to 0.
    
    # Let's try a simple heuristic:
    # Score = EI + w * (LCB - min(LCB))  # Shift LCB to be positive?
    # Or Score = EI + w * sigma?  # Add exploration bonus
    
    # Actually, let's look at the "normalized max" strategy again.
    # It takes the max of normalized EI, LCB_0.5, ..., LCB_5.0.
    # This means if ANY of these metrics thinks a point is "top 1%" (in relative terms), it gets a high score.
    # This is very aggressive exploration.
    
    # What if we use **Pareto Front** selection?
    # Points that are good in both EI and LCB.
    # But we just return a single score per point.
    
    # Let's try a **Decaying Exploration** approach using f_best as a proxy for progress.
    # As f_best decreases, we reduce exploration.
    # Let kappa = max(0.1, 10.0 / (f_best + 1.0))
    # If f_best is large (early), kappa is large (explore).
    # If f_best is small (late), kappa is small (exploit).
    
    # This is a material difference: adaptive kappa based on progress.
    
    # Estimate progress:
    # f_best starts around 50-100 and goes to 0.
    # Let's assume a typical initial f_best ~ 50.
    # If f_best > 10, we are still exploring.
    # If f_best < 1, we are exploiting.
    
    # Let's define a soft switch:
    # weight_explore = 1.0 / (1.0 + exp((f_best - 5.0) / 2.0))
    # If f_best >> 5, weight_explore ~ 1.
    # If f_best << 5, weight_explore ~ 0.
    
    # But wait, the Beale function minimum is 0.
    # So f_best will eventually be very close to 0.
    
    # Let's try a simpler adaptive kappa:
    # kappa = 2.0 * (f_best / (f_best + 1.0)) + 0.5
    # If f_best is large, kappa ~ 2.5.
    # If f_best is small, kappa ~ 0.5.
    
    # And we use LCB with this adaptive kappa.
    # But EI is also good.
    
    # Let's blend EI and Adaptive LCB.
    # Score = 0.5 * normalized_EI + 0.5 * normalized_LCB_adaptive
    
    # Normalization:
    # mean_ei = np.mean(ei)
    # std_ei = np.std(ei)
    # if std_ei > 1e-9:
    #     norm_ei = (ei - mean_ei) / std_ei
    # else:
    #     norm_ei = ei
    
    # lcb_adaptive = kappa * sigma_safe - mu
    # mean_lcb = np.mean(lcb_adaptive)
    # std_lcb = np.std(lcb_adaptive)
    # if std_lcb > 1e-9:
    #     norm_lcb = (lcb_adaptive - mean_lcb) / std_lcb
    # else:
    #     norm_lcb = lcb_adaptive
    
    # final_scores = 0.5 * norm_ei + 0.5 * norm_lcb
    
    # Let's implement this.
    
    # Adaptive kappa
    # f_best is the current best.
    # If f_best is very small, we want to exploit.
    # If f_best is large, we want to explore.
    # Let's use a sigmoid switch centered at f_best = 1.0 (since min is 0).
    # switch = 1.0 / (1.0 + np.exp((f_best - 1.0) / 0.5))
    # If f_best >> 1, switch ~ 1 (explore).
    # If f_best << 1, switch ~ 0 (exploit).
    
    # kappa = 0.5 + 2.0 * switch
    # So kappa ranges from 0.5 to 2.5.
    
    switch = 1.0 / (1.0 + np.exp((f_best - 1.0) / 0.5))
    kappa = 0.5 + 2.0 * switch
    
    lcb_adaptive = kappa * sigma_safe - mu
    
    # Normalize EI
    mean_ei = np.mean(ei)
    std_ei = np.std(ei)
    if std_ei > 1e-9:
        norm_ei = (ei - mean_ei) / std_ei
    else:
        norm_ei = np.zeros_like(ei)
        
    # Normalize LCB
    mean_lcb = np.mean(lcb_adaptive)
    std_lcb = np.std(lcb_adaptive)
    if std_lcb > 1e-9:
        norm_lcb = (lcb_adaptive - mean_lcb) / std_lcb
    else:
        norm_lcb = np.zeros_like(lcb_adaptive)
        
    # Blend
    # Weight towards EI when exploiting (low f_best), towards LCB when exploring (high f_best).
    # Actually, the switch already controls kappa.
    # Let's just take the max of the two normalized scores?
    # Or a weighted sum?
    # Let's try weighted sum with weight = switch.
    # If exploring (switch=1), weight LCB more?
    # If exploiting (switch=0), weight EI more?
    
    # Let's try: final = switch * norm_lcb + (1 - switch) * norm_ei
    # This seems logical.
    
    final_scores = switch * norm_lcb + (1.0 - switch) * norm_ei
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time
    if time.time() > deadline:
        pass
        
    return final_scores
# EVOLVE-BLOCK-END
