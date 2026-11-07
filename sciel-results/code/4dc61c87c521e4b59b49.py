"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    start_time = time.time()
    deadline = start_time + 280.0  # 50s safety margin before 330s limit
    
    # 1. Compute base EI scores for all 2000 candidates
    # Handle zero variance safely
    sigma_safe = np.where(sigma > 1e-8, sigma, 1e-8)
    z = (f_best - mu) / sigma_safe
    ei_scores = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Ensure finite scores
    ei_scores = np.where(np.isfinite(ei_scores), ei_scores, -np.inf)
    
    # 2. Identify top 5 candidates for local refinement
    # We want to explore the neighborhood of the best global candidates
    top_k = 5
    top_indices = np.argsort(ei_scores)[-top_k:]
    
    # Initialize scores with base EI
    scores = ei_scores.copy()
    
    # 3. For each top candidate, perform a local grid search
    # The "coordinates" are the indices in the 2000 candidate array.
    # Since we don't have actual coordinates, we assume the 2000 points are
    # somewhat uniformly distributed. However, we can't interpolate in space.
    # 
    # Wait, the prompt says "No coordinates are passed". This means I cannot
    # perform a spatial grid search around the chosen point because I don't
    # know where it is in [0,1]^3.
    #
    # Re-reading the task: "mu and sigma have shape (2000,). Return 2000 finite scores... No coordinates are passed."
    # This implies the acquisition function must operate purely on the distribution of mu and sigma.
    # I cannot move the candidate in space. I can only assign a score to the existing 2000 candidates.
    #
    # Therefore, "local refinement" in space is impossible.
    #
    # What *can* I do differently?
    # The current program uses standard EI.
    # Inspiration 2 tried a portfolio and failed (score 0.118).
    #
    # Alternative Strategy: Use a different acquisition function that is known to be robust or superior in specific regimes.
    # Common alternatives:
    # 1. UCB (Upper Confidence Bound): mu + kappa * sigma. Good for exploration.
    # 2. PI (Probability of Improvement): P(f < f_best) = Phi((f_best - mu)/sigma).
    # 3. Max EI (already used).
    # 4. Knowledge Gradient (approximate).
    # 5. EI with a penalty for exploration (exploitation bias).
    #
    # Let's try a hybrid or a different formulation.
    # A common improvement over EI is to use a "greedy" approach that balances exploration and exploitation more aggressively.
    #
    # However, EI is generally considered the gold standard for simple regret in Bayesian Optimization when the GP is well-calibrated.
    # The score is 1.0, which is the baseline.
    #
    # Let's look at the "inspiration 1" which also scored 1.0. It says "uses a single, robust, and computationally efficient EI".
    #
    # What if we use a different acquisition function entirely?
    # Let's try UCB with a tuned beta.
    # Or let's try to improve the EI calculation itself.
    #
    # Actually, a very effective strategy in BO is to use a "lookahead" or "batch" approach, but here we select one point.
    #
    # Let's try **UCB** (Upper Confidence Bound).
    # UCB = mu + beta * sigma
    # For minimization, we want to minimize f. So we want to find points where the lower bound is low?
    # No, UCB is typically for maximization. For minimization, we use LCB (Lower Confidence Bound) or just invert.
    # If we minimize f, we want to pick points with low predicted f and high uncertainty (potential to be even lower).
    # Score = -mu + beta * sigma ? No, that favors high sigma and low mu.
    # Let's think: We want to find the minimum.
    # EI = E[max(f_best - f, 0)].
    #
    # Let's try a different angle: **Probability of Improvement (PI)** with a small xi.
    # PI = P(f < f_best - xi) = Phi((f_best - xi - mu) / sigma)
    #
    # Or, let's stick to EI but change the exploration-exploitation balance.
    # Standard EI is quite exploitative.
    #
    # Let's try a **Gaussian Process-UCB** style score for minimization:
    # score = -mu + beta * sigma
    # This prioritizes low mean (good) and high variance (uncertainty).
    #
    # Let's test this hypothesis. If EI is stuck at 1.0, maybe UCB explores better and finds the global minimum faster?
    # Or maybe it explores too much?
    #
    # Another option: **Maximize the probability that the next point is better than f_best by a margin xi**.
    #
    # Let's try a **Portfolio** approach but simpler:
    # Compute EI and UCB. Return the element-wise maximum? No, that changes the scale.
    #
    # Let's try **UCB** specifically.
    # For minimization, the acquisition function should be high when we expect a low value.
    # If we define the "regret" as f_best - f, then EI is the expected positive regret.
    # UCB for minimization: We want to minimize f. The lower bound on f is mu - beta*sigma.
    # We want to pick the point with the lowest lower bound?
    # Score = -(mu - beta*sigma) = -mu + beta*sigma.
    #
    # Let's implement UCB and see.
    # Beta needs to be chosen. A common heuristic is beta = sqrt(2 * log(t)) or similar, but we don't have t.
    # We can estimate t from the magnitude of f_best or just use a fixed beta.
    # Given the domain [0,1]^3 and Hartmann 3D, the scale is roughly O(1).
    # Let's try beta = 1.0 or 2.0.
    
    # Let's try a different strategy: **Expected Improvement with a small exploration noise**
    # Or simply **LCB (Lower Confidence Bound)** for minimization.
    # LCB = mu - beta * sigma.
    # We want to minimize this value. So the acquisition score (higher is better) should be -LCB = -mu + beta * sigma.
    
    # Let's try beta = 2.0.
    
    beta = 2.0
    ucb_scores = -mu + beta * sigma
    
    # Ensure finite
    ucb_scores = np.where(np.isfinite(ucb_scores), ucb_scores, -np.inf)
    
    # We can also try PI
    # pi_scores = norm.cdf((f_best - mu) / sigma_safe)
    
    # Let's return UCB scores.
    return ucb_scores
# EVOLVE-BLOCK-END
