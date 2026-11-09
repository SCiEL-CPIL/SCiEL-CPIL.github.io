import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 325.0  # 5s margin before 330s limit

def acquisition(mu, sigma, f_best):
    # Deadline guard: if we are too close to the limit, return a safe fallback
    # Since this function is very fast, this is unlikely to trigger, but it's required.
    if time.time() > _DEADLINE:
        # Fallback to a simple EI-like score to avoid hanging
        return -mu + 0.1 * sigma

    # Keep the Expected Improvement (EI) design and investigate improvements
    # within its family.
    #
    # The current program used an "exploration_factor" of 1.1, which effectively
    # increased the sigma in the EI calculation. This was measured to be worse
    # than pure EI (score 0.909 vs 1.0). The inspirations show that pure EI
    # works well (score 1.0).
    #
    # However, Inspiration 3 suggests a temperature scaling based on the median
    # sigma. This is a structural refinement within the EI family.
    # The idea is that early on, when sigma is large, EI values are large.
    # Later, sigma shrinks, and EI values shrink.
    # Dividing by a temperature that tracks the scale of sigma can normalize
    # the acquisition landscape, potentially helping the optimizer maintain
    # a consistent exploration/exploitation balance relative to the current
    # posterior uncertainty.
    #
    # Let's implement pure EI first, as that is the strongest baseline (score 1.0).
    # Then, let's see if the temperature scaling from Inspiration 3 adds value.
    # Given the directive to "KEEP this design and investigate improvements within its family",
    # and the observation that removing the exploration bonus (going to pure EI) improved
    # the score from 0.909 to 1.0, the most robust move is to return pure EI.
    # The temperature scaling in Insp 3 is a modification. Let's test if it helps or hurts.
    # Since Insp 3 scored 1.0 as well, it is at least not worse.
    # But pure EI is simpler and a proven standard.
    #
    # Let's look closer at the "Lessons": "Removing the exploration bonus reverts to pure EI... Measured: 0.08261 -> 1."
    # This confirms pure EI is the target.
    #
    # Can we improve on pure EI?
    # Inspiration 3 uses: scores = ei / temp, where temp = clip(median(sigma), 1e-6, 1.0).
    # If median(sigma) is, say, 0.5, then scores = ei / 0.5 = 2 * ei. This is just a constant scaling.
    # Since the evaluator picks the argmax, a positive constant scaling does not change the argmax.
    # Therefore, the temperature scaling in Insp 3 is mathematically equivalent to pure EI in terms of selection,
    # UNLESS the scaling is non-uniform across candidates (which it isn't, it's a scalar).
    # Wait, `temp` is a scalar derived from the median of the sigma array.
    # So `ei / temp` is just `c * ei` where `c` is a scalar.
    # This does not change the ranking of candidates.
    # So Insp 3 is functionally identical to pure EI in terms of which point is selected.
    #
    # Thus, the best strategy is simply to implement pure EI correctly.
    # The current program's exploration factor of 1.1 changed the ranking (it boosted high-sigma regions more).
    # Pure EI is the optimal balance for this problem according to the measurements.

    # Ensure sigma is strictly positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Standardized improvement
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement formula
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where Phi is CDF and phi is PDF of standard normal
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finiteness
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
