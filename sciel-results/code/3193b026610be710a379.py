import numpy as np
from scipy.stats import norm
import time

# Strategy: Time-Adaptive GP-UCB with Diversification Penalty
#
# The current program uses a static kappa=3.0 UCB. This often leads to
# "sticking" in high-uncertainty regions that are not actually promising
# or repeatedly exploring areas that were already probed in early steps
# because the uncertainty there is still high relative to the noise.
#
# This new strategy uses:
# 1. A time-adaptive beta_t: In Bayesian Optimization, the standard UCB
#    uses beta_t = log(t * pi^2 * M / (6 * log(2 * delta))).
#    Since we don't have access to 't' (the step count) directly, we can
#    estimate it from the number of evaluations. However, we only have
#    mu, sigma, and f_best.
#    
#    Alternative inference: 
#    The problem states 30 evaluations total, starting with 5 random.
#    So t goes from 5 to 30.
#    We can estimate t by looking at the "quality" of f_best or simply
#    assume a typical trajectory. But a more robust way without 't' is
#    to use the variance of the observed data? No, we don't have X or Y.
#    
#    Wait, we can infer the step count roughly from the distribution of
#    mu and sigma? No, that's unreliable.
#    
#    Let's look at the constraints again.
#    "For each seed 0, 1 and 2 ... until 30 evaluations"
#    The function `acquisition` is called at each step.
#    We can maintain a global counter!
#    Is that allowed? "Keep the function names and signatures".
#    Yes, global state is usually allowed in these coding challenges unless
#    explicitly forbidden. Let's use a global step counter.
#
# 2. Diversification Penalty:
#    Early on, the GP has high uncertainty everywhere except near the
#    initial points. UCB will naturally explore near the initial points
#    because sigma is high there. To force exploration of *new* areas,
#    we can penalize points that are "close" to where the GP is already
#    very confident? No, that's hard to detect.
#    
#    Actually, a simpler structural change: 
#    Instead of just -mu + beta*sigma, we can use a "Lower Confidence Bound"
#    for minimization but with a **non-monotonic** beta that starts high,
#    drops, and then rises again? No, standard UCB works well if beta is
#    tuned.
#
#    Let's try a different acquisition family: **Expected Hypervolume Improvement (EHVI)**
#    or **Probability of Improvement (POI)**?
#    POI is often too greedy. EHVI is expensive to compute for 2000 points
#    but might be feasible? 2000 points in 2D.
#    EHVI requires integrating the posterior. That's complex.
#
#    Let's stick to UCB but change the **structure** of the exploration term.
#    Instead of linear sigma, use a **quantile-based** exploration.
#    Score = -mu + beta * sigma * (1 + 0.5 * np.exp(-sigma)) ? No.
#
#    Let's look at Inspiration 3 (score 1.4761) which used time-adaptive GP-UCB.
#    It scored worse than the parent (1.7075). So pure time-adaptive beta might
#    be too aggressive or too conservative.
#
#    Inspiration 1 (score 2.8295) is the best. What did it do?
#    "uses numpy, scipy; defines acquisition". The description is vague.
#    But the score is much higher.
#
#    Let's try a **Hybrid EI + UCB** but with a **dynamic weight** based on
#    the ratio of sigma to the current best improvement potential.
#
#    Actually, the most robust structural change for 2D bounded domains
#    with a single minimum is to use **Random Forests** or **Kriging**
#    with a **multi-start** local optimization? No, we only return scores.
#
#    Let's try **Thompson Sampling** approximation.
#    Instead of picking the max of -mu + beta*sigma, we can pick the max of
#    -mu + z*sigma where z is drawn from a standard normal.
#    But we must return deterministic scores for the 2000 candidates.
#    The evaluator picks the argmax.
#    If we use a fixed random seed for the z-draws, it becomes deterministic
#    but pseudo-random.
#    
#    However, the task says "Return 2000 finite scores".
#    If we use Thompson Sampling, we would typically sample a function,
#    then pick the argmin of the sample.
#    Here, we return scores. The evaluator picks the max score.
#    So if we set score = -mu + z*sigma, and the evaluator picks max score,
#    it's equivalent to minimizing mu - z*sigma.
#    This is exactly UCB with a random beta.
#    
#    The problem with static UCB is that it can get stuck.
#    The problem with random UCB is variance.
#
#    Let's try a **Deterministic Thompson Sampling** using a fixed set of
#    "virtual" beta values.
#    No, that's just multi-UCB.
#
#    Let's look at the **Beale Function** structure.
#    f(x,y) = (1.5 - 2x + x^2)(3.0 - 3y + y^2)^2
#    Minimum at (3, 2) with value 0.
#    Domain [-4.5, 4.5]^2.
#    
#    The function is very flat in some directions and steep in others.
#    A standard GP with Matern-5/2 length scale 0.2 (scaled) might struggle
#    to capture the global structure if the initial points are far apart.
#    
#    Key Insight: The "stagnation" signal suggests the population is not
#    improving. This might be because the acquisition function is too
#    conservative or too aggressive.
#    
#    Let's try **Entropy Search** approximation?
#    Score = H(f(x)) - E[H(f(x) | f_best)]
#    This is complex.
#
#    Let's try a **Simpler Structural Change**:
#    Use **Log-EI** (Log Expected Improvement) or **q-EI**?
#    No, we are picking one point.
#
#    Let's try **PI (Probability of Improvement)** combined with **UCB**.
#    Score = alpha * POI + (1-alpha) * UCB
#    
#    But the current program is already a blend (according to the comment
#    "blends EI and UCB", though the code shows pure UCB kappa=3.0).
#    Wait, the code provided in the prompt is:
#    `score = -mu + kappa * sigma_safe`
#    So it IS pure UCB.
#    
#    The comment says "blends EI and UCB" but the code does not.
#    Maybe the "current program" description in the prompt is slightly
#    misleading or refers to a previous version?
#    The prompt says: "CURRENT PROGRAM (score 1.7075) and its strategy: ...
#    The current program blends EI and UCB. Let's try pure UCB..."
#    But the code block shows pure UCB.
#    I will assume the code is the truth.
#    
#    So, I will implement a **True EI (Expected Improvement)** for minimization.
#    EI(x) = E[max(f_best - f(x), 0)]
#    For a GP, if f(x) ~ N(mu, sigma^2):
#    EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
#    where Phi is the CDF and phi is the PDF of the standard normal.
#    
#    Why EI? It is theoretically grounded and balances exploration/exploitation
#    naturally without a hyperparameter kappa. It focuses on points that are
#    likely to improve the best value.
#    
#    However, EI can be too greedy if sigma is very small.
#    To fix this, we can use **q-EI** or add a small exploration term.
#    Or, we can use **Log-EI** which is more robust.
#    
#    Let's implement **Standard EI** for minimization.
#    
#    Formula:
#    z = (f_best - mu) / sigma
#    EI = sigma * (z * Phi(z) + phi(z))
#    
#    Note: If sigma is 0, EI is max(f_best - mu, 0).
#    
#    This is a materially different design family (Probabilistic Improvement
#    vs. Confidence Bound).

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate Z-score
    z = (f_best - mu) / sigma_safe
    
    # Calculate EI
    # Phi(z) = CDF of standard normal
    # phi(z) = PDF of standard normal
    # Using scipy.stats.norm for accuracy
    # We can use norm.cdf and norm.pdf
    # Or use the error function for speed?
    # 2000 points is small, scipy is fine.
    
    cdf_vals = norm.cdf(z)
    pdf_vals = norm.pdf(z)
    
    ei = sigma_safe * (z * cdf_vals + pdf_vals)
    
    # EI is always non-negative.
    # Higher EI means more expected improvement.
    # We want to maximize EI, so we return EI as the score.
    
    # Handle numerical issues
    ei = np.where(np.isfinite(ei), ei, 0.0)
    
    return ei
