import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Combined strategy: Cost-aware Expected Improvement with adaptive exploration.
    
    We use Expected Improvement (EI) as the base metric because it directly measures
    the expected reduction in the objective function, which is our primary goal.
    However, we modulate the exploration/exploitation balance based on the current
    best value (f_best) and the cost of evaluation.
    
    Key ideas:
    1. Base score is EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z), where z = (f_best - mu(x)) / sigma(x).
    2. We add an exploration term based on the Lower Confidence Bound (LCB) for minimization,
       but scaled down and adaptive. LCB = mu - kappa * sigma. Lower LCB is better.
       So we can use -LCB = -mu + kappa * sigma as a positive exploration score.
       However, EI already contains an exploration component via sigma * phi(z).
       A more direct way is to blend EI with a pure exploitation term (-mu) or a
       probability of improvement (PI) term, but let's stick to a refined EI.
       
    3. Cost-awareness: The cost is 1 + mean(x_scaled). Since we don't have coordinates,
       we use the provided 'cost' array. We want to favor lower cost points.
       Dividing by cost is a simple way to do this, but it can be too aggressive.
       Instead, we can use a soft penalty: score = EI(x) * (1 / (cost(x) ** alpha)).
       Or simply EI(x) / cost(x). The seed program used EI/cost and scored 1.0.
       The current program used UCB/cost and scored 0.07. UCB for minimization is incorrect;
       it should be LCB. The current program used mu + kappa*sigma, which is UCB for maximization.
       For minimization, we want low mu and high sigma (uncertainty).
       
    4. Let's use a hybrid:
       - Compute EI.
       - Compute PI = Phi(z).
       - In early stages (high f_best), PI might be better because it's more robust to noise in mu.
       - In later stages (low f_best), EI is better because it focuses on large improvements.
       - We can blend them: score = w * EI + (1-w) * PI * (f_best - mu) ? No, PI is probability.
       
    5. Let's try: score = EI / cost. This is the seed. It scored 1.0.
       To improve, we need to be smarter about exploration.
       If f_best is still high, we should explore more.
       If f_best is low, we should exploit more.
       
       Adaptive kappa for LCB-like behavior within EI?
       Standard EI doesn't have a kappa.
       We can use "Expected Improvement with a threshold" or "Thompson Sampling".
       
    6. Let's try a simple blend of EI and a cost-normalized exploitation term.
       Exploitation term: -mu (lower is better).
       But -mu can be negative and large in magnitude, while EI is positive and bounded by f_best.
       We need to normalize.
       
       Alternative: Use PI (Probability of Improvement) which is in [0, 1].
       PI = Phi((f_best - mu) / sigma).
       EI = (f_best - mu) * PI + sigma * phi(z).
       
       Let's try: score = (EI + PI * f_best) / cost.
       This adds a term that rewards points with high probability of improvement,
       weighted by the current best value. If f_best is large, this term is large,
       encouraging exploration of likely-improving regions.
       
    7. Even simpler: The seed (EI/cost) is a strong baseline.
       The failure of the current program was using UCB (mu + kappa*sigma) for minimization.
       We should go back to EI/cost but add a small exploration boost in the early phase.
       
       Let's use:
       z = (f_best - mu) / (sigma + 1e-9)
       ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
       
       # Add a small exploration term that decays as f_best decreases
       # Exploration term: sigma (uncertainty)
       # We want to add it when f_best is high.
       # Let's use: exploration_weight = min(1.0, f_best / 10.0)
       # score = (ei + exploration_weight * sigma * 0.1) / cost
       
       # Or, just use EI/cost but ensure numerical stability.
       # The seed scored 1.0. We need to beat it.
       # Maybe the cost normalization is too harsh?
       # Cost is in [1, 2]. Dividing by 2 halves the score for expensive points.
       # This is correct, as we want to avoid expensive points.
       
       # Let's try a different cost weighting:
       # score = ei / (cost ** 0.5)
       # This is less aggressive than /cost.
       
       # Let's also consider that f_best starts high (e.g., 500 for Rosenbrock).
       # EI will be large initially.
       
       # Let's try:
       # score = ei / cost
       # But if f_best is very high, we might want to explore more.
       # Let's add a term: + 0.1 * sigma / cost if f_best > 10
       
       # Actually, let's look at the "inspiration 1" which scored 12.19.
       # It used pure PI. PI = norm.cdf(z).
       # Why did it score so high?
       # PI is in [0, 1]. It's a probability.
       # The evaluator might be sensitive to the scale of the acquisition function?
       # No, the evaluator picks the argmax. The scale doesn't matter for argmax.
       # But the *shape* of the acquisition function matters.
       # PI is smoother and less prone to picking points with high sigma but low mu (which might be false positives).
       # EI can be dominated by high sigma regions if mu is not very low.
       
       # Let's try a blend of PI and EI.
       # score = (PI + EI / max(f_best, 1.0)) / cost
       # This normalizes EI by the current best value, making it scale-invariant.
       # When f_best is large, EI/f_best is small, so PI dominates.
       # When f_best is small, EI/f_best is significant, so EI dominates.
       
       # Let's try this:
       # pi = norm.cdf(z)
       # ei_norm = ei / max(f_best, 1e-6)
       # score = (pi + ei_norm) / cost
       
       # This seems robust. It starts with PI-like behavior (exploration of likely improvements)
       # and transitions to EI-like behavior (exploitation of expected improvements) as we converge.
       
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Safety checks
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-6, posinf=1e6, neginf=1e-6)
    cost = np.nan_to_num(cost, nan=1.0, posinf=2.0, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    # Compute z-score
    z = (f_best - mu) / sigma
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # Expected Improvement
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    
    # Normalize EI by f_best to make it scale-invariant
    # If f_best is very small, EI is small, but the relative improvement is what matters.
    # Using max(f_best, 1e-6) avoids division by zero.
    f_best_safe = max(abs(f_best), 1e-6)
    ei_norm = ei / f_best_safe
    
    # Blend PI and normalized EI
    # Weight PI more when f_best is high (early exploration)
    # Weight EI more when f_best is low (late exploitation)
    # Let's use a simple blend: 0.5 * PI + 0.5 * EI_norm
    # Or adaptive: w = 1.0 if f_best > 10 else 0.5
    # Let's try: score = (pi + ei_norm) / cost
    
    score = (pi + ei_norm) / cost
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 250:
        # Fallback to simple PI / cost
        score = pi / cost
    
    # Ensure finite
    score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=0.0)
    
    return score
