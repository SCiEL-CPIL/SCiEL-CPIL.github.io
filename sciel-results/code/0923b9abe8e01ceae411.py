import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return -mu

    # Strategy: Greedy Batch Selection with Diversity
    # The evaluator picks the single candidate with the highest score.
    # However, by structuring the scores to reflect a "batch" selection process
    # where we value diversity, we can influence the trajectory of the GP.
    #
    # Wait, the evaluator only picks ONE point per step.
    # How does "batch selection" help if only one point is evaluated?
    # It doesn't directly help unless we are in a batch setting.
    # BUT, the prompt says "greedy batch selection strategy... constructs a diversity-aware set... thereby explicitly preventing the search from collapsing".
    # If only one point is evaluated, we must return a score for each of the 2000 points.
    # The evaluator picks argmax(score).
    #
    # Re-reading the directive: "Replace the myopic point-wise acquisition with a greedy batch selection strategy that constructs a diversity-aware set of candidates using a greedy maximization of a hypervolume or spacing-based objective over the top-k scored points".
    # This implies we should simulate the effect of batch selection or use a surrogate that encourages diversity.
    #
    # A common technique when only one point can be selected but diversity is desired is to use an acquisition function that penalizes points close to previously visited points.
    # However, we don't have access to previously visited points (x, y) directly, only mu, sigma, f_best.
    #
    # Alternative interpretation: The "batch" aspect might refer to the fact that we are selecting from a pool of 2000 candidates.
    # If we cannot access coordinates, we cannot compute distances between candidates.
    # We only have mu and sigma.
    #
    # Let's look at the inspiration: "greedy batch selection strategy... diversity-aware set... spacing-based objective".
    # If we don't have coordinates, we cannot compute spacing.
    #
    # Is it possible the coordinates are encoded in mu/sigma? No, mu/sigma are 1D arrays of length 2000.
    #
    # Perhaps the "diversity" must be derived from the GP posterior itself?
    # Or perhaps the strategy is to use an acquisition function that is NOT purely myopic, but considers the "coverage" implied by the posterior?
    #
    # Actually, if we look at the constraint: "No coordinates are passed."
    # This makes "spacing-based objective" impossible to compute directly between candidates.
    #
    # Let's reconsider the "Greedy Batch Selection" idea in the context of single-point selection.
    # Maybe the idea is to use a **Lookahead** or **Multi-step** strategy?
    # But we don't have the GP kernel to update the posterior for lookahead.
    #
    # Let's look at the "Inspiration 1" which scored 2.2504. It used "Step 1: ... materially different strategy is An | uses numpy, scipy, time; defines acquisition". The text is cut off.
    #
    # Let's try a different structural change: **Probability of Improvement (PI) with a dynamic threshold** or **Knowledge Gradient (KG)** approximation?
    # KG is often superior to EI for multimodal functions because it considers the value of information for the entire remaining budget, not just the immediate improvement.
    #
    # However, the directive specifically asks for: "greedy batch selection strategy that constructs a diversity-aware set... spacing-based objective".
    # If I cannot compute spacing, I must find a proxy for diversity.
    #
    # One proxy for diversity in the absence of coordinates is to use the **uncertainty** (sigma) as a proxy for "unexplored" regions.
    # High sigma = likely unexplored = diverse.
    # Low sigma = likely explored = redundant.
    #
    # The current program already uses EI which balances mu and sigma.
    #
    # What if we use a **Two-Stage Selection**?
    # 1. Compute EI scores.
    # 2. Select top-K candidates based on EI.
    # 3. From these top-K, select the one that is most "different" from the others in terms of their mu/sigma profile?
    # This doesn't make sense without coordinates.
    #
    # Let's re-read the prompt carefully.
    # "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    #
    # If coordinates are not passed, any strategy relying on spatial distance between candidates is impossible.
    # The directive might be a generic template that doesn't perfectly fit the "no coordinates" constraint, OR I must interpret "diversity" differently.
    #
    # Alternative Strategy: **Entropy-based Acquisition** or **Information Gain**.
    # Maximizing the information gain about the location of the minimum.
    #
    # Another idea: **Thompson Sampling with Diversity**.
    # Sample multiple functions from the posterior? No, we only have mu/sigma for 2000 points.
    #
    # Let's try **Upper Confidence Bound (UCB) with a stronger exploration term** but structured differently.
    # The current program uses a linear decay of kappa.
    # What if we use a **Non-Monotonic** schedule or a **Sine-wave** schedule to force re-exploration later?
    #
    # Or, what if we use **Bayesian Optimization with a Fixed Number of Restarts**?
    # We can't restart the GP.
    #
    # Let's look at the "Idea target": "Does operator 'diff' change the chance that a child improves its island...?"
    # This is meta-instructional.
    #
    # Let's try a **Hybrid of EI and Pure Uncertainty** with a **Dynamic Switch**.
    #
    # Actually, there is a known issue with EI in multimodal landscapes: it can get stuck in a local mode.
    # A strategy to counter this is to use **Multiple EI thresholds** or **Clustered EI**.
    #
    # Since we can't cluster in space, we can cluster in **Value Space**.
    #
    # Let's try **Probability of Improvement (PI)** with a **decaying beta**.
    # PI = P(f(x) < f_best - beta) = Phi((f_best - beta - mu) / sigma).
    # If we set beta to decay, it becomes more exploitative.
    #
    # But the directive says "Structural change... different design or modelling family".
    #
    # Let's try **Knowledge Gradient (KG)** approximation.
    # KG approximates the expected reduction in regret after observing the value at x.
    # For a single observation, KG is similar to EI but with a different weighting.
    #
    # Another strong candidate: **Variance-based Acquisition**.
    # Score = sigma^2. This maximizes exploration.
    #
    # Let's try a **Switching Strategy**:
    # Steps 0-10: Use UCB with high kappa (Exploration).
    # Steps 11-24: Use EI (Exploitation/Exploration balance).
    # This is a structural change from a smooth decay to a hard switch.
    #
    # Or, **Randomized Acquisition**:
    # With probability p, pick the best EI point.
    # With probability 1-p, pick a point with high sigma (random exploration).
    # But we must return scores, not a single point. The evaluator picks argmax.
    # We can't randomize the return value easily without breaking the "deterministic" nature if not careful, but we can return a score that reflects this mixture.
    #
    # Let's implement a **Two-Phase Acquisition**:
    # Phase 1 (Steps 0-12): Maximize Uncertainty (sigma) to cover the domain.
    # Phase 2 (Steps 13-24): Maximize EI to refine the minimum.
    #
    # This is a structural change from "smooth decay" to "hard phase transition".
    #
    # Implementation:
    # if step < 13:
    #     score = sigma  # Pure exploration
    # else:
    #     score = EI     # Standard exploitation
    
    # Let's refine the transition.
    # A sudden jump from sigma to EI might be jarring.
    # Let's blend them.
    # score = alpha * sigma + (1-alpha) * EI
    # where alpha goes from 1.0 to 0.0.
    # This is similar to the current kappa schedule, but using sigma directly instead of in the UCB form.
    
    # Let's try a different structural change: **Negative Log-Likelihood of the Posterior**.
    # Not applicable.
    
    # Let's go with the **Phase-Based Strategy**:
    # 1. High exploration (sigma) for the first 10 steps.
    # 2. Balanced (EI) for the next 10 steps.
    # 3. High exploitation (mu) for the last 5 steps.
    
    step = _step_counter
    if step < 0:
        step = 0
        
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate EI
    Z = (f_best - mu) / sigma_safe
    Phi_Z = norm.cdf(Z)
    phi_Z = norm.pdf(Z)
    EI = (f_best - mu) * Phi_Z + sigma_safe * phi_Z
    
    # Calculate PI (Probability of Improvement)
    # PI = Phi(Z)
    
    # Calculate UCB
    # UCB = -(mu + kappa * sigma)
    
    # Phase 1: Steps 0-9 -> Maximize Sigma (Exploration)
    # Phase 2: Steps 10-19 -> Maximize EI (Balance)
    # Phase 3: Steps 20-24 -> Minimize Mu (Exploitation)
    
    if step <= 9:
        # Pure exploration: High sigma means uncertain, likely unexplored.
        # We want to maximize sigma.
        scores = sigma_safe
    elif step <= 19:
        # Balanced: EI
        scores = EI
    else:
        # Exploitation: Minimize mu (Maximize -mu)
        scores = -mu
        
    return scores
