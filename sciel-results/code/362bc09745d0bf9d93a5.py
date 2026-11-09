"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    # Deadline guard
    if time.time() - _start_time > 320:
        return np.zeros_like(mu)

    sigma_safe = np.maximum(sigma, 1e-9)
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted LCB / Resource Allocation Strategy
    # The directive suggests: "exploration term is scaled by the inverse square root of the cost"
    # In a standard LCB: score = mu - beta * sigma
    # Here we treat the acquisition score as a blend of exploitation (EI) and exploration.
    # However, the specific structural idea is to penalize uncertainty in high-cost regions.
    # Let's interpret "Cost-Adjusted Lower Confidence Bound" in the context of acquisition functions.
    # Usually, LCB favors low mu and high sigma.
    # If we use LCB directly: lcb = mu - beta * sigma
    # But we need to incorporate cost.
    # The inspiration says: "exploration term is scaled by the inverse square root of the cost".
    # This implies: effective_sigma = sigma / sqrt(cost) ?
    # Or does it mean the exploration bonus (beta * sigma) is divided by sqrt(cost)?
    # Let's look at the current program's approach: PI / cost^gamma.
    # The inspiration 1 approach: EI / cost.
    # The directive strategy: "Cost-Adjusted LCB ... exploration term scaled by 1/sqrt(cost)".
    
    # Let's construct a score that combines the strong exploitation signal of EI
    # with an exploration term that is damped by cost.
    
    # A common cost-aware formulation is:
    # Score = EI - lambda * cost
    # Or Score = EI / (cost + epsilon)
    
    # Let's try to implement the specific "inverse square root of cost" scaling on the exploration part.
    # We can define a composite score.
    # Exploitation: EI (high when f_best is far above mu)
    # Exploration: sigma (high in uncertain regions)
    # Standard UCB/LCB mix: alpha * EI + beta * sigma (for maximization of score)
    # But EI already contains sigma.
    
    # Let's stick to the directive's specific hint:
    # "Model the optimization as a constrained resource allocation problem by using a 
    # Cost-Adjusted Lower Confidence Bound (LCB) where the exploration term is scaled 
    # by the inverse square root of the cost"
    
    # LCB for minimization (we want to minimize f, so we want low f).
    # Standard LCB: L(x) = mu(x) - beta * sigma(x)
    # We want to maximize an acquisition function.
    # If we use LCB, we would select the point with the lowest LCB.
    # But the task asks for "higher = evaluate next".
    # So we can define Acquisition = -LCB = -mu + beta * sigma.
    # This is essentially a UCB (Upper Confidence Bound) for the *negative* objective, or just a standard UCB if we were maximizing.
    # Wait, for minimization, we want to find the minimum.
    # LCB gives a lower bound on the true function value. We pick the point with the lowest LCB.
    # To fit the "higher score is better" interface, we can use:
    # Score = - (mu - beta * sigma) = beta * sigma - mu.
    # This is a UCB-like score for minimization? No, UCB is for maximization.
    # For minimization, we often use LCB. To convert to "higher is better", we can just negate it.
    # Score = -mu + beta * sigma.
    
    # Now, apply the cost adjustment.
    # "exploration term is scaled by the inverse square root of the cost"
    # Exploration term is beta * sigma.
    # So, adjusted_exploration = (beta * sigma) / sqrt(cost)
    # Score = -mu + (beta * sigma) / sqrt(cost)
    
    # However, simple LCB/UCB can be too greedy or too exploratory.
    # The current program (PI/cost^1.5) scored 1.39.
    # Inspiration 1 (EI/cost) scored 1.00 (worse).
    # The knowledge says: "Increased UCB weight in early stages... Measured: 1.146 -> 1.336".
    # This suggests UCB/LCB based methods *can* work well if tuned.
    
    # Let's combine the robustness of EI (which is 0 when mu > f_best) with the cost-aware exploration.
    # If we use pure LCB (-mu + beta*sigma/sqrt(cost)), it might explore areas where mu is already worse than f_best.
    # EI is zero there.
    
    # Hybrid Idea:
    # Use EI as the base exploitation signal.
    # Add an exploration bonus that is cost-aware.
    # Exploration bonus = beta * sigma / sqrt(cost)
    # But we should only add this bonus if there is potential for improvement?
    # Or just add it?
    
    # Let's try: Score = EI + beta * sigma / sqrt(cost)
    # This ensures we don't waste cost on points where EI is 0 (mu >> f_best) unless sigma is huge and cost is low.
    # Actually, if mu >> f_best, EI=0. But sigma might be high.
    # If we add beta * sigma / sqrt(cost), we might still explore high-cost, high-uncertainty, high-mu regions.
    # The 1/sqrt(cost) penalty helps, but is it enough?
    
    # Alternative: Scale the entire EI by 1/cost? No, that was inspiration 1 (score 1.0).
    # The current program (score 1.39) uses PI / cost^1.5.
    # PI is bounded [0,1]. EI is unbounded.
    # Dividing unbounded EI by cost might be too aggressive in penalty if EI is large.
    
    # Let's look at the "inverse square root" hint again.
    # It specifically mentions scaling the *exploration term*.
    # In EI, the exploration part is roughly sigma * phi(z).
    # In LCB, it is beta * sigma.
    
    # Let's try a Cost-Aware UCB (for minimization, so -mu + exploration).
    # But to avoid exploring hopeless regions, let's gate it or blend with EI.
    
    # Proposed Structural Change:
    # 1. Compute EI.
    # 2. Compute a cost-adjusted exploration term: exp_term = sigma / sqrt(cost)
    # 3. Combine: Score = EI + beta * exp_term
    # 4. To prevent exploring regions where improvement is impossible (mu >> f_best),
    #    we can multiply the exploration term by a factor that decays as mu increases relative to f_best.
    #    Or simply rely on the fact that EI=0 there, and the exploration term is penalized by cost.
    
    # Let's test this logic.
    # If cost is high (e.g., 2.0), sqrt(cost) ~ 1.41. Exploration is reduced by ~30%.
    # If cost is low (e.g., 1.0), sqrt(cost) = 1. Exploration is full.
    # This is a milder penalty than dividing by cost (which halves exploration at cost=2).
    # The current program divides by cost^1.5. At cost=2, penalty is 2^1.5 ~ 2.8.
    # So 1/sqrt(cost) is much less aggressive than the current program's penalty.
    # Since the current program scores 1.39 and the less aggressive penalty (1/cost) scored 1.0,
    # maybe the current penalty is good, but the *signal* (PI) is weak.
    # EI is a stronger signal.
    
    # Let's try: Score = EI / (cost ** 0.5)
    # This applies the "inverse square root of cost" to the whole score, not just exploration.
    # But the directive says "exploration term".
    
    # Let's try to be more faithful to the directive:
    # Score = EI + (beta * sigma) / sqrt(cost)
    # With beta = 0.5?
    
    # Let's also consider that EI can be negative due to numerical issues, we clipped it to 0.
    # The exploration term is always positive.
    # This might cause us to explore high-cost areas if sigma is large.
    # But the 1/sqrt(cost) helps.
    
    # Let's refine the combination.
    # Maybe: Score = EI / (1 + cost) + beta * sigma / sqrt(cost) ?
    # No, let's keep it simple.
    
    # Let's go with:
    # base_ei = EI
    # exploration = sigma / sqrt(cost + 1e-9)
    # score = base_ei + 0.5 * exploration
    
    # Why 0.5? It's a guess. The knowledge said "UCB weight up to 0.7" improved scores in another context.
    # Here, EI is the main driver. The exploration is a secondary nudge.
    
    # Let's ensure finite values.
    scores = ei + 0.5 * (sigma_safe / np.sqrt(cost + 1e-9))
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
