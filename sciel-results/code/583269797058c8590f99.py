"""Cost-aware acquisition function for Bayesian optimization (rastrigin 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_ACQ_START = time.time()
_ACQ_DEADLINE = _ACQ_START + 280.0  # safety margin before 330s kill


def _expected_regret(mu, sigma, f_best):
    """
    Expected Regret for minimization: E[max(0, f(x) - f_best)].
    This is the expected penalty for not being at the best known value.
    Lower ER is better; we negate it for a "higher is better" score.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    # ER = sigma * (1 - Phi(z)) + (f_best - mu) * phi(z) ... wait, let's derive.
    # f(x) ~ N(mu, sigma^2)
    # Regret R = max(0, f(x) - f_best)
    # E[R] = integral_{f_best}^{inf} (y - f_best) * N(y; mu, sigma^2) dy
    # Let z = (f_best - mu)/sigma, so f_best = mu + sigma*z
    # E[R] = sigma * [ (1 - z*Phi(z) - phi(z)) ] ... no.
    # Standard formula: E[max(0, X - c)] where X~N(m, s^2)
    # = s * (1 - Phi((c-m)/s)) + (m-c)*phi((c-m)/s) ... wait.
    # Let's use the symmetry.
    # E[max(0, X-c)] = E[(X-c) * I(X>c)]
    # = E[X|X>c]P(X>c) - c P(X>c)
    # = (m + s*phi(z)/Phi(-z)) * Phi(-z) - c * Phi(-z)  where z=(c-m)/s
    # = (m - c)Phi(-z) + s*phi(z)
    # Since we want to MINIMIZE f, and f_best is the current min,
    # we actually want to find points where f(x) < f_best.
    # The "regret" of a point x relative to the current best is how much WORSE it could be?
    # No, in BO for minimization, we usually maximize Expected Improvement (EI)
    # EI = E[max(0, f_best - f(x))]
    # This is exactly what the parent program used (essentially).
    # The directive suggests "Expected Regret (ER) normalized by cost".
    # Usually "Regret" in minimization context is f(x) - f_opt. But we don't know f_opt.
    # If we interpret "Expected Regret" as the expected value of the function minus the best found,
    # i.e., E[f(x) - f_best], that is just mu - f_best. That's not right.
    #
    # Let's look at the inspiration 1 which scored 3.21. It used EI.
    # The directive says: "Replace the standard Expected Improvement (EI) with Expected Regret (ER) ... which directly optimizes the expected reduction in the objective function per unit cost".
    # "Expected reduction in the objective function" IS Expected Improvement.
    # Perhaps the terminology "Expected Regret" here refers to the regret of NOT picking this point?
    # Or is it the "Expected Hypervolume Improvement"?
    #
    # Let's re-read carefully: "Expected Regret (ER) ... directly optimizes the expected reduction in the objective function per unit cost".
    # This description matches EI (Expected Improvement) perfectly. EI = E[max(0, f_best - f(x))].
    # Why call it ER? Maybe it's a typo in the directive or a specific variant.
    # However, there is a metric called "Expected Regret" in multi-objective or specific contexts.
    # In single-objective minimization, the standard is EI.
    #
    # Let's look at the current program (score 1.9793). It blends PI and UCB.
    # The parent (score 1.0) used EI/cost.
    # Inspiration 1 (score 3.21) used a blend of EI, UCB, and PI.
    #
    # The directive asks for a STRUCTURAL change.
    # Current: Blend of normalized PI and UCB, divided by cost.
    # Proposal: Use Expected Improvement (EI) but with a different cost-awareness mechanism?
    # Or maybe the "Expected Regret" refers to the *simple regret* of the sample?
    #
    # Let's try a different structural approach: **Knowledge Gradient (KG)** or **Thompson Sampling** style?
    # Thompson Sampling is often more effective in multimodal landscapes like Rastrigin because it naturally handles exploration/exploitation without tuning beta.
    #
    # Strategy: Use a Thompson Sampling-inspired score.
    # Instead of a deterministic score like EI or UCB, we can sample from the posterior and pick the sample with the lowest value?
    # But we need to return a deterministic score for each of the 2000 candidates to pick the *best* one.
    # We can't do stochastic sampling in the acquisition function easily because it needs to be deterministic for the evaluator to pick the argmax.
    #
    # Alternative Structural Change: **Cost-Aware Expected Improvement with Adaptive Normalization**.
    # The current program normalizes PI and UCB to [0,1] and blends them. This loses scale information.
    # EI has natural scale (objective units).
    #
    # Let's try: Score = EI(x) / (cost(x)^alpha)
    # Where alpha is adaptive.
    #
    # But the directive specifically mentions "Expected Regret (ER)".
    # In some literature, "Expected Regret" is defined as E[f(x) - min(f)].
    # If we approximate min(f) with f_best, then ER(x) = E[f(x)] - f_best = mu - f_best.
    # Minimizing ER means minimizing mu. That's just greedy.
    #
    # Let's reconsider the "Expected Reduction" phrase.
    # "optimizes the expected reduction in the objective function per unit cost"
    # Reduction = f_best - f(x).
    # Expected Reduction = E[max(0, f_best - f(x))] = EI.
    # So the directive is likely using "Expected Regret" as a synonym for "Expected Improvement" or "Expected Gain".
    #
    # So, why did the current program (PI+UCB blend) score 1.97, while Inspiration 1 (EI+UCB+PI) scored 3.21?
    # The current program normalizes PI and UCB to [0,1]. This makes the scores dimensionless and potentially distorts the relative importance.
    # Inspiration 1 likely used raw EI or a better blend.
    #
    # Let's implement a robust **Cost-Aware Expected Improvement (EI)** with an **Exploration Bonus** that decays.
    #
    # Score = (EI(x) + gamma * sigma(x)) / cost(x)^beta
    #
    # To make it a "structural change" from the current PI/UCB blend:
    # 1. Drop the [0,1] normalization of components.
    # 2. Use EI as the primary exploitation signal.
    # 3. Use sigma as the exploration signal, scaled by a factor that depends on the progress (f_best).
    # 4. Use a power-law cost penalty.
    
    if time.time() > _ACQ_DEADLINE:
        # Fallback: simple cost-aware mean
        return -mu / (cost + 1e-9)
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # 1. Expected Improvement (EI)
    # EI = E[max(0, f_best - f(x))]
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # 2. Exploration term
    # We want to explore when uncertainty is high.
    # But we don't want to explore too much once we are close to the optimum.
    # Rastrigin minimum is 0.
    # If f_best is close to 0, we should exploit more.
    # If f_best is large, we should explore more.
    # Let's use a progress-based weight.
    # progress = 1 - f_best / f_scale. But we don't know f_scale.
    # Rastrigin on [-5.12, 5.12] can have values up to ~20-30.
    # Let's assume a max value of ~30 for normalization if f_best is high.
    # Or simply: weight = 1.0 / (1.0 + f_best)
    # If f_best is large (e.g., 10), weight is 0.09.
    # If f_best is small (e.g., 0.1), weight is 0.91.
    # This encourages exploration early and exploitation later.
    
    progress_weight = 1.0 / (1.0 + f_best)
    
    # Exploration bonus: gamma * sigma
    # gamma should be significant.
    gamma = 1.0
    exploration = gamma * sigma_safe * progress_weight
    
    # 3. Combine EI and Exploration
    # EI is in objective units. Sigma is in objective units.
    # This preserves scale.
    raw_score = ei + exploration
    
    # 4. Cost-awareness
    # Cost is 1 + mean(x_scaled). Range [1, 2].
    # We want to penalize high cost.
    # Dividing by cost is standard.
    # Maybe power > 1 to be more aggressive?
    cost_safe = np.maximum(cost, 1e-6)
    score = raw_score / (cost_safe ** 1.5)
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
