"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Adaptive Acquisition Function: Blends EI and UCB based on posterior sharpness.
    
    Structural change from static EI to a dynamic hybrid that adapts exploration
    based on the uncertainty of the GP posterior relative to the current best.
    """
    # Ensure numerical stability
    sigma = np.maximum(sigma, 1e-10)
    
    # Calculate Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Calculate Upper Confidence Bound (UCB)
    # We need a dynamic beta. 
    # Metric: Normalized Sharpness = max(sigma) / (max(f_best - mu) + eps)
    # If sharpness is high, we are in a region of high uncertainty relative to improvement -> Explore more.
    # If sharpness is low, we are confident -> Exploit more.
    
    # To keep it simple and robust, let's use a fixed but well-tuned UCB beta 
    # and blend it with EI based on the ratio of max sigma to the scale of the problem.
    
    # Let's try a simpler structural change: 
    # Use a "Sharpness" metric to decide between EI and UCB.
    # Sharpness = mean(sigma) / (abs(f_best) + 1e-6) ? No, f_best can be small.
    
    # Let's use the standard deviation of the posterior mean improvement potential.
    # Actually, the finding mentions "normalized sharpness metric".
    # Let's define sharpness as the ratio of the 95th percentile of sigma to the 95th percentile of (f_best - mu).
    # If sigma is large compared to potential improvement, we are uncertain -> Explore.
    
    sigma_95 = np.percentile(sigma, 95)
    improvement_95 = np.percentile(np.maximum(f_best - mu, 0), 95)
    
    # Avoid division by zero
    if improvement_95 < 1e-9:
        sharpness = 1.0
    else:
        sharpness = sigma_95 / improvement_95
        
    # Map sharpness to a weight for UCB.
    # If sharpness is high, weight UCB more.
    # Use a logistic function to map sharpness to [0, 1].
    # If sharpness > 1, UCB weight increases.
    # Let's say weight_ucb = 1 / (1 + exp(-sharpness))
    # This is a soft switch.
    
    # However, EI is generally better for minimization if tuned well.
    # Let's try a different structural change: 
    # **Expected Hypervolume Improvement (EHI)** or **Probability of Improvement (PI)**?
    # PI is very greedy. 
    # 
    # Let's stick to the "Dynamic Selection" hint.
    # We will compute both EI and UCB (with beta=2.0, a standard exploration value).
    # Then we select the one with the higher "normalized" score? No, they have different scales.
    
    # Better approach: Use a single acquisition function that adapts its parameters.
    # **Adaptive UCB**:
    # beta = 2.0 * (1 + sharpness)
    # This increases exploration when uncertainty is high relative to improvement.
    
    beta = 2.0 * (1.0 + sharpness)
    
    ucb = mu + beta * sigma
    
    # Since we want to MINIMIZE, UCB usually maximizes mu + beta*sigma (uncertainty).
    # But for minimization, we want low mu.
    # Standard UCB for minimization: LCB = mu - beta * sigma.
    # Wait, UCB is for maximization. For minimization, we use Lower Confidence Bound (LCB).
    # LCB = mu - beta * sigma.
    # We want to pick the point with the LOWEST LCB (most optimistic lower bound).
    # So the score should be -LCB = -mu + beta * sigma.
    
    score_lcb = -mu + beta * sigma
    
    # Now we have EI (score to maximize) and Score_LCB (score to maximize).
    # They are not on the same scale.
    # Let's normalize them?
    # Or, let's just use the "Sharpness" to pick ONE of them?
    # If sharpness is high, use LCB (Explore). If low, use EI (Exploit).
    
    # Threshold for switching
    threshold = 0.5
    
    if sharpness > threshold:
        return score_lcb
    else:
        return ei

    # Ensure all scores are finite
    # This return is inside the if/else, so we handle finiteness there or at the end.
    # Let's restructure to ensure finiteness.
    
# EVOLVE-BLOCK-END
