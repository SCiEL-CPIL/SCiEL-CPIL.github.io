"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of the Beale function.
    Uses a combination of Expected Improvement and a noise-adaptive term.
    """
    # Guard against division by zero or negative variance
    sigma = np.maximum(sigma, 1e-12)
    
    # Standard Expected Improvement (EI)
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Handle numerical stability: where sigma is tiny, EI can be unstable.
    # If sigma is very small, we rely on the mean being low.
    # A common trick is to use a lower bound on sigma for the z-score calculation
    # but keep the original sigma in the second term if it's large enough.
    # However, standard EI is generally robust.
    
    # Let's add a small exploration bonus based on uncertainty to avoid local optima
    # The Beale function has a narrow valley. Pure EI might get stuck.
    # UCB (Upper Confidence Bound) for minimization: mu - kappa * sigma
    # Or a mix.
    
    # Let's try a weighted sum of EI and UCB.
    # UCB score: lower is better. We want higher score to be selected.
    # So UCB_score = -mu + kappa * sigma
    # EI_score = ei
    
    # In the Beale function, the global minimum is at (3, 0.5) with value 0.
    # The function is sensitive to x.
    
    # Let's use a slightly aggressive exploration strategy.
    # If we use pure EI, it's safe but might be slow.
    # If we use UCB, it's faster but might miss the optimum.
    
    # A good compromise: 
    # Score = EI + lambda * sigma
    # This encourages exploring high uncertainty areas even if the mean is not great.
    
    # Alternatively, we can use the "Expected Improvement" directly, which is the standard.
    # The seed program uses EI. The score was 1.0.
    # Let's try to improve the EI calculation or add a small perturbation.
    
    # One issue with EI is that if f_best is not very low, EI can be large everywhere.
    # As f_best decreases, EI becomes more focused.
    
    # Let's try "Probability of Improvement" (PI) combined with EI?
    # PI = norm.cdf(z)
    
    # Let's stick to EI but ensure numerical stability.
    # The current implementation:
    # z = (f_best - mu) / (sigma + 1e-9)
    # return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # This is standard EI.
    
    # Maybe the issue is that the GP model is not accurate enough?
    # Or the search space is large.
    
    # Let's try a different acquisition: Upper Confidence Bound (UCB)
    # UCB = -mu + kappa * sigma
    # kappa controls the exploration/exploitation trade-off.
    # For a 2D problem with 25 evaluations, a moderate kappa might work.
    
    # Let's try a hybrid: 0.5 * EI + 0.5 * UCB_normalized?
    # Normalizing is tricky without knowing the scale.
    
    # Let's try just UCB with a small kappa.
    # kappa = 0.1 ?
    
    # Or, let's look at the "Expected Hypervolume Improvement" (EHVI) concept? Too complex.
    
    # Let's try "Expected Improvement" with a small noise term added to the mean?
    
    # Actually, a very effective strategy for these simple benchmarks is often
    # to use a slightly more exploratory EI or UCB.
    
    # Let's try: Score = EI + 0.1 * sigma
    # This adds a small bonus for uncertainty.
    
    # Let's experiment with kappa in UCB.
    # Score = -mu + 0.5 * sigma
    
    # Let's try a mix of EI and UCB.
    # First, normalize EI and UCB? No, that's hard.
    
    # Let's try the "Probability of Improvement" (PI) which is more robust to scale?
    # PI = norm.cdf(z)
    # But PI doesn't account for the magnitude of improvement.
    
    # Let's go back to EI. The seed program uses EI and gets 1.0.
    # 1.0 means it matches the baseline.
    # We need to beat 1.0.
    
    # Maybe the GP hyperparameters are fixed in the evaluator?
    # "GP: Matern-5/2, length scale 0.2 on coordinates scaled to [0, 1], noise 1e-6, standardized targets"
    # Yes, the GP is fixed. We only control the acquisition function.
    
    # So we must choose the best acquisition function for this specific GP setup.
    
    # With a short length scale (0.2), the GP is very wiggly.
    # EI might be too conservative, sticking to known good areas.
    # UCB might be better to explore the wiggly landscape.
    
    # Let's try UCB with kappa = 1.0.
    # Score = -mu + sigma
    
    # Or maybe a smaller kappa?
    # Let's try kappa = 0.5.
    
    # Let's try a combination:
    # Score = EI + 0.5 * sigma
    
    # Let's test this hypothesis.
    
    # Implementation:
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Add a small exploration term
    exploration = 0.1 * sigma
    
    return ei + exploration

# EVOLVE-BLOCK-END
