# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    # Safety deadline guard
    t_start = time.time()
    deadline = t_start + 290.0  # Keep a 40s buffer before the 330s kill

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)

    n = len(mu)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Prevent division by zero and handle edge cases
    sigma_safe = np.where(sigma > 1e-10, sigma, 1e-10)
    
    # Ensure mu and sigma are finite
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma_safe = np.nan_to_num(sigma_safe, nan=1e-10, posinf=1e6, neginf=1e-10)

    # --- Deterministic Thompson Sampling Proxy ---
    # 
    # Logic:
    # 1. We want to estimate the probability that a candidate x_i is the global minimizer.
    # 2. Let M* be the random variable representing the true global minimum value.
    # 3. A common approximation for M* in BO is derived from the posterior predictions.
    #    Since we don't have the full GP kernel matrix inversion here (only mu/sigma at candidates),
    #    we approximate the distribution of M* using the "best" of the current candidates, 
    #    or more robustly, we model M* as a distribution that is informed by f_best 
    #    and the uncertainty of the candidates.
    #
    # A simpler and effective structural alternative is the "Probability of Improvement" 
    # but formulated relative to the *global* minimum estimate.
    #
    # However, a very powerful and distinct strategy from EI/LCB is **Knowledge Gradient (KG)** 
    # or **Upper Confidence Bound (UCB)** for minimization. 
    # The current code uses a blend of EI and LCB. 
    # 
    # Let's try a **Pareto-based** or **Multi-objective** view? No, that's complex.
    #
    # Let's implement **Thompson Sampling** via a deterministic proxy:
    # Score(x) = P(f(x) < M*)
    # We need a distribution for M*. 
    # A reasonable estimator for M* is the minimum of the observed f_best and the 
    # posterior predictive lower bounds. 
    # 
    # Alternative: Use the **Expected Improvement** but with a dynamic baseline.
    # The current program uses a fixed f_best.
    # 
    # Let's try **Max Value Entropy Search (MVES)** approximation?
    # Too complex.
    #
    # Let's go with a **Adaptive UCB** that is structurally different from the 
    # current "Blend of Normalized EI and LCB".
    # The current code normalizes EI and LCB separately and blends them. 
    # This can lead to "double counting" or scaling issues.
    #
    # New Strategy: **Direct UCB with Adaptive Beta**
    # Score(x) = - (mu(x) - beta * sigma(x))
    # But instead of a simple sigmoid on f_best, we use a **Bayesian** update for beta.
    # We treat the choice of beta as a hyperparameter that should be larger when 
    # we have few samples (high uncertainty about the function) and smaller when 
    # we are close to the optimum.
    # 
    # We can estimate the "number of samples" or "progress" from the spread of mu/sigma?
    # No, we only have mu/sigma at 2000 candidates.
    #
    # Let's use the **variance of the candidates** as a proxy for exploration needs.
    # If the candidates are all clustered in low sigma regions, we might be exploiting.
    # If there are high sigma regions, we might need to explore.
    #
    # Actually, the most distinct and robust change is to use **Log-Expected Improvement (LogEI)** 
    # or **Probability of Improvement (PI)** with a penalty for distance from f_best?
    #
    # Let's implement **Thompson Sampling** properly by sampling from the posterior 
    # once and using that sample to score all points?
    # Wait, we must return scores for ALL 2000 points. 
    # Standard TS would require sampling the function at all 2000 points, which is 
    # not possible without the kernel.
    #
    # So we stick to pointwise acquisition functions.
    #
    # Structural Change: **Hybrid of UCB and PI with a Dynamic Switch based on 
    # the Ratio of Uncertainty to Improvement**.
    #
    # Let's calculate the **Information Gain** proxy?
    #
    # Let's try **Slope-over-Confidence (SOC)**? No, that's for hyperparameter tuning.
    #
    # Let's try **Expected Hypervolume Improvement**? No, single objective.
    #
    # I will implement a **Robust UCB** that uses the **median** of the sigma distribution 
    # to normalize the exploration term, making it scale-invariant relative to the 
    # current candidate set, rather than just raw sigma.
    #
    # Score = -mu + kappa * sigma
    # But kappa is adjusted by the "optimism" required.
    #
    # Let's look at Inspiration 1 (score 1.946). It likely used a strong exploitation 
    # or a specific UCB variant.
    #
    # I will implement **UCB with a Time-Decaying Exploration Term** based on 
    # the assumption that f_best decreases monotonically.
    # We can estimate the "step" by how low f_best is.
    #
    # But to be *structurally different*, I will use **Quantile-based Selection**.
    # Instead of a single score, I will compute the score such that the top 1% 
    # of candidates are selected based on a **Mixture of Experts**:
    # 1. Exploitation Expert: -mu
    # 2. Exploration Expert: -sigma (or sigma, depending on sign)
    #
    # The score will be:
    # S(x) = w_e * (-mu(x)) + w_x * sigma(x)
    # where w_e + w_x = 1.
    #
    # This is linear, but the weights are determined by a **Logistic Regression** 
    # on the features (mu, sigma, f_best) trained implicitly by the current state?
    # No, we can't train.
    #
    # Let's use a **Fixed Mixture** but with **Non-linear Weighting**:
    # w_e = sigmoid(k * (f_best - mu(x)))
    # This makes the weight dependent on the specific point's mu relative to f_best.
    # If mu(x) is close to f_best, w_e is high (exploit).
    # If mu(x) is far, w_e is low (explore).
    #
    # Score(x) = w_e(x) * (-mu(x)) + (1 - w_e(x)) * sigma(x)
    #
    # This is **Pointwise Adaptive UCB**.
    #
    # Let's refine this.
    # For minimization:
    # We want high score for low mu and high sigma.
    # 
    # Score = -mu + kappa(x) * sigma
    #
    # Let kappa(x) = K_max * sigmoid( -K_slope * (mu(x) - f_best) )
    # If mu(x) < f_best (improvement), sigmoid is high -> high kappa? 
    # No, if mu < f_best, we should exploit (low kappa).
    # So we want kappa to be LOW when mu < f_best.
    # sigmoid( - (mu - f_best) ) = sigmoid( f_best - mu ).
    # If mu < f_best, f_best - mu > 0, sigmoid > 0.5.
    # If mu >> f_best, f_best - mu < 0, sigmoid ~ 0.
    #
    # So:
    # kappa(x) = K_min + (K_max - K_min) * sigmoid( (f_best - mu(x)) / T )
    #
    # If mu is very good (low), kappa is high? 
    # Wait. If mu is low, we are close to the best. We should exploit (low kappa).
    # If mu is high, we are far. We should explore (high kappa)?
    # Actually, UCB logic: 
    # LCB = mu - kappa * sigma.
    # We maximize -LCB = -mu + kappa * sigma.
    #
    # If mu is low (good), we want to pick it. -mu is large positive. 
    # Adding kappa*sigma might push it away if sigma is high.
    # But if mu is low, it's likely a good point.
    #
    # Standard UCB uses fixed kappa.
    #
    # Let's try **Inverse Variance Weighting**?
    #
    # I will implement **Thompson Sampling** via **Gaussian Sampling** of the 
    # acquisition function parameters? No.
    #
    # Let's go with **Normalized UCB** where the normalization is done on the 
    # **entire candidate set** to ensure the exploration term is scaled 
    # relative to the current uncertainty landscape.
    #
    # 1. Calculate raw UCB scores for a range of kappas? No, too slow.
    # 2. Calculate a single UCB score with an adaptive kappa.
    #
    # Adaptive Kappa Strategy:
    # kappa = K0 / (1 + f_best)
    # This decays exploration as f_best approaches 0.
    #
    # To make it structural, I will **Normalize the UCB score** by the 
    # standard deviation of the UCB scores across all candidates.
    # This ensures that the "top" candidate is always selected, but the 
    # relative ordering is preserved.
    #
    # Score = (UCB - mean(UCB)) / std(UCB)
    #
    # This is robust to the scale of the function.
    
    # Define Adaptive Kappa
    # f_best is the current best value.
    # As f_best -> 0, we want less exploration.
    # Kappa = K_base / (1.0 + f_best)
    # If f_best is large (early), Kappa ~ K_base.
    # If f_best is small (late), Kappa ~ K_base / f_best (large)? 
    # No, we want Kappa to DECREASE.
    # Kappa = K_base * (1.0 / (1.0 + f_best))
    # If f_best = 100, Kappa ~ 0.01 * K_base.
    # If f_best = 0.1, Kappa ~ 0.5 * K_base.
    # This seems backwards. We want MORE exploration early (high f_best) and LESS late (low f_best).
    # So Kappa should be HIGH when f_best is HIGH.
    # Kappa = K_max * (f_best / (f_best + C))
    # If f_best is large, Kappa ~ K_max.
    # If f_best is small, Kappa ~ K_max * (f_best / C) -> 0.
    
    K_max = 3.0
    C = 1.0
    kappa = K_max * (f_best / (f_best + C))
    
    # Add a small floor to kappa to ensure some exploration always
    kappa = max(kappa, 0.1)
    
    # Calculate UCB Score (for minimization, we maximize -mu + kappa*sigma)
    ucb_scores = -mu + kappa * sigma_safe
    
    # Normalize the scores to be zero-mean and unit-variance
    # This makes the selection robust to the absolute scale of mu and sigma
    mean_score = np.mean(ucb_scores)
    std_score = np.std(ucb_scores)
    
    if std_score > 1e-9:
        final_scores = (ucb_scores - mean_score) / std_score
    else:
        final_scores = np.zeros_like(ucb_scores)
        
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Check time
    if time.time() > deadline:
        pass
        
    return final_scores
# EVOLVE-BLOCK-END
