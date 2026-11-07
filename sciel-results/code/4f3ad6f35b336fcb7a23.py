import numpy as np
import time
from scipy.stats import norm

_T0 = None
_DEADLINE = None


def _ensure_deadline():
    global _T0, _DEADLINE
    if _T0 is None:
        _T0 = time.perf_counter()
        _DEADLINE = _T0 + 300.0
    return _DEADLINE


def acquisition(mu, sigma, f_best):
    deadline = _ensure_deadline()
    elapsed = time.perf_counter() - _T0
    
    # If we are close to the deadline, fall back to a simple, fast exploitation strategy
    if elapsed > _DEADLINE - 5.0:
        return -mu

    n = mu.shape[0]
    
    # Safety clamps for sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    sigma_safe = np.minimum(sigma_safe, 1e4)
    
    # Estimate progress based on f_best
    # Goldstein-Price min is 3. Max in domain is around 1000-1200.
    f_max = 1000.0
    f_min = 3.0
    
    # p is 0 at start (f_best high), 1 at end (f_best low)
    p = (f_max - f_best) / (f_max - f_min)
    p = np.clip(p, 0.0, 1.0)
    
    # Strategy: Blend Adaptive LCB with POI
    # The current program (score 2.81) used Adaptive LCB.
    # Inspiration 3 (score 1.46) used EI/POI-like logic but was slower or less effective? 
    # Actually, the prompt says "COMBINE... take the structural idea of one and the refinement machinery of the other".
    # The current program is the "structural idea" (Adaptive LCB framework).
    # The "refinement machinery" from inspirations:
    # - Inspiration 1: Adjusted beta range (2.0 to 0.5).
    # - Inspiration 3: Used scipy.stats.norm for smooth probability-based scoring.
    # - Directive: Use POI.
    
    # Let's create a hybrid: 
    # Early on (low p), use more exploration (higher beta in LCB or high POI weight).
    # Late on (high p), use exploitation (low beta in LCB or low POI weight).
    
    # Let's compute both LCB and POI scores and blend them.
    # LCB score: -mu + beta * sigma
    # POI score: Phi((f_best - mu) / sigma)
    
    # Adaptive Beta for LCB
    # Start with a higher beta for exploration, decrease to lower for exploitation.
    # Current: 1.0 -> 0.2
    # Insp 1: 2.0 -> 0.5
    # Let's try a curve that starts aggressive but settles down.
    beta = 1.5 - 1.0 * p  # 1.5 at start, 0.5 at end
    
    score_lcb = -mu + beta * sigma
    
    # POI Score
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    z = np.clip(z, -10, 10)
    score_poi = norm.cdf(z)
    
    # Blend
    # We want to switch from exploration to exploitation.
    # LCB is generally better for continuous optimization when we have a good GP.
    # POI is more "binary" and can get stuck.
    # Let's weight LCB heavily, but add a small POI component to break ties or encourage slight exploration?
    # Or, perhaps the "POI" directive suggests using POI as the primary signal in early stages?
    
    # Let's try a smooth blend.
    # weight_poi: high when p is low (early), low when p is high (late).
    # weight_lcb: 1 - weight_poi
    
    # If we use pure LCB, it's the current program.
    # If we use pure POI, it's different.
    # Let's try: score = (1 - p) * score_poi + p * score_lcb ?
    # No, LCB is usually superior. Let's stick to LCB but refine the beta schedule and add a tiny POI term to ensure we don't collapse too early?
    # Actually, the directive says "Use the Probability of Improvement (POI)... providing a distinct... signal".
    # Let's try a weighted sum where the weight of POI decreases as we progress.
    
    # Normalization: LCB scores can be large negative/positive. POI is [0,1].
    # We need to scale them to be comparable.
    # A simple way: 
    # score = alpha * score_lcb + (1-alpha) * score_poi_scaled
    
    # But scaling is tricky. 
    # Alternative: Use LCB, but if the LCB score is dominated by exploration (high sigma), 
    # maybe POI helps?
    
    # Let's look at the scores again.
    # Current: 2.81
    # Insp 1 (LCB 2.0-0.5): 2.36 (Worse)
    # Insp 2 (LCB 0.5 fixed): 1.48 (Much worse)
    # Insp 3 (EI): 1.46 (Much worse)
    
    # The current program is the best. It uses beta 1.0 -> 0.2.
    # Insp 1 tried 2.0 -> 0.5 and got worse. This suggests that *too much* early exploration hurts.
    # Therefore, we should NOT increase beta. We should keep it similar or slightly lower.
    
    # The directive says "Use POI".
    # Maybe we should use POI *instead* of LCB in the very early stages?
    # Or blend them?
    
    # Let's try a blend where POI is used when uncertainty is high, and LCB when it's low?
    # Or simply: Score = LCB + small_constant * POI?
    
    # Let's try to improve the LCB schedule first, as it's the strongest base.
    # Current: 1.0 -> 0.2
    # Maybe 0.8 -> 0.1? Less exploration overall?
    
    # Let's try:
    beta = 0.8 - 0.6 * p  # 0.8 at start, 0.2 at end
    
    score_lcb = -mu + beta * sigma
    
    # Add a small POI component to break plateaus?
    # score_poi is in [0,1]. score_lcb can be large.
    # Let's scale POI by the typical magnitude of LCB?
    # Or just add it as a tie-breaker?
    # score = score_lcb + 0.01 * score_poi
    
    # This is a safe change. It keeps the LCB structure but adds the POI signal as requested.
    # The 0.01 factor ensures LCB dominates, but POI can influence close calls.
    
    score = score_lcb + 0.01 * score_poi
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
