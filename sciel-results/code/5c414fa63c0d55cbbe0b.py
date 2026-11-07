"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We will track call count to potentially adapt, but keep it simple first.
def acquisition(mu, sigma, f_best):
    start_time = time.time()
    
    # Hypervolume based acquisition
    # We consider the "front" defined by the current best point (x_best, f_best) 
    # but since we don't have x_best, we approximate the hypervolume contribution
    # by considering the area under the curve of the expected improvement 
    # relative to a reference point.
    
    # A common HV-based acquisition for single-objective is related to 
    # the area between the posterior mean and the reference point, 
    # but weighted by the probability of being better than f_best.
    
    # Let's try a formulation:
    # Score(x) = E[ (f_best - f(x))_+ * (1 - P(f(x) < f_best)) ] ? No, that's not HV.
    
    # Hypervolume Indicator for a single point x with value f(x) against reference r:
    # HV = (f_ref - f(x)) if f(x) < f_ref else 0.
    # Expected HV = E[ max(0, f_ref - f(x)) ].
    # This is exactly Expected Improvement (EI) with f_ref = f_best.
    # So standard EI IS the expected hypervolume improvement for a single point.
    
    # To make it "Hypervolume-based" in a different way, we might consider 
    # the hypervolume of the *Pareto front* formed by the observed points and the candidate.
    # But we don't have the observed points' locations, only f_best.
    
    # Alternative interpretation: 
    # Use a different reference point or a different geometric measure.
    # For example, the area under the CDF?
    
    # Let's try the "Probability of Improvement" (PI) combined with EI, 
    # or a "Spread" measure.
    
    # Actually, the prompt suggests: "maximizes the expected increase in the hypervolume 
    # of the Pareto front defined by the objective and the current best value".
    
    # If we treat the current best as a point in a 2D space (e.g., index vs value, 
    # or just value vs uncertainty), we can define a hypervolume.
    
    # Let's try a score that balances EI and Uncertainty (sigma) in a way that 
    # mimics hypervolume expansion.
    # HV expansion is proportional to the "area" added.
    # For a new point (f, s), if it beats f_best, it adds area.
    
    # Let's compute:
    # 1. EI (standard)
    # 2. PI (Probability of Improvement)
    # 3. UCB (Upper Confidence Bound)
    
    # A hypervolume-inspired score might be:
    # Score = EI + lambda * sigma * PI
    # Or simply a weighted sum that emphasizes the "tail" of the distribution.
    
    # Let's try a specific HV proxy:
    # The expected hypervolume contribution of a candidate x, assuming the 
    # reference point is (0, f_best) in a (value, uncertainty) space?
    
    # Let's stick to a robust modification of EI that incorporates 
    # a "spread" bonus which helps in multimodal landscapes by 
    # rewarding high uncertainty more strongly when the mean is close to f_best.
    
    z = (f_best - mu) / (sigma + 1e-9)
    
    # Standard EI
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Uncertainty bonus
    # We want to encourage exploration. 
    # A simple HV-like term: area of the rectangle defined by (f_best - mu) and sigma?
    # Let's try: Score = EI + alpha * sigma * pi
    # This increases the weight of high sigma when there's a reasonable chance of improvement.
    
    alpha = 0.5  # Tuning parameter
    
    scores = ei + alpha * sigma * pi
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Safety margin check
    elapsed = time.time() - start_time
    if elapsed > 280:
         # If we are running out of time, just return the EI part which is faster?
         # No, this is just one call. The loop is outside.
         pass
         
    return scores
# EVOLVE-BLOCK-END
