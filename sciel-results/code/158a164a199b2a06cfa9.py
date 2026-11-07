"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()

# We do not have access to the actual observed points x, only mu, sigma, cost.
# However, we can infer "unexplored" regions from high sigma.
# The key structural change: Instead of EI (which is 0 outside the current best's neighborhood 
# in a probabilistic sense, or very small), we use a hybrid that strongly rewards high sigma 
# (uncertainty) relative to the cost, but also keeps a strong exploitation term.
#
# A major failure mode of standard CEI on Hartmann is getting stuck in local minima 
# because the cost penalty (1 + mean(x)) biases towards x near 0, but the true minimum 
# is shifted. If the GP has already sampled near the true minimum, EI will stay high there.
# If it hasn't, EI might be low because f_best is still far away.
#
# Strategy: Use a "Confidence-Weighted" score.
# Score = (f_best - mu) * w_exploit + sigma * w_explore
# But we make it cost-aware by dividing by cost.
# Crucially, we add a "diversity" proxy: we don't have x, but we can assume that 
# regions with very low sigma are "explored". We want to boost regions where sigma is high 
# compared to the global mean sigma, to encourage exploration of unexplored areas.
#
# Let's try a different acquisition: **Cost-Aware Upper Confidence Bound (UCB)** 
# but with an adaptive beta that increases as the number of evaluations grows 
# (we can estimate progress from f_best or just use a fixed, slightly higher beta than typical).
# Actually, the prompt says "Do not micro-tune constants". 
#
# Let's look at the "inspiration" that failed: "Multi-Resolution Expected Improvement".
# Let's look at the "inspiration" that succeeded (island 5): "Cost-penalized EI ... small penalty (0.1)".
# The current program uses EI / cost.
#
# What is a *materially different* design?
# **Thompson Sampling** is a Bayesian alternative to EI/UCB. 
# We don't have samples from the GP posterior, only mu and sigma. 
# We can approximate a single sample from the posterior: y = mu + sigma * N(0,1).
# But we need a deterministic score for 2000 points. 
# We can use the **expected value of the best of a batch** or simply **Posterior Sampling** 
# where we estimate the probability that a point is the best.
#
# A simpler, robust alternative: **Probability of Improvement (PI)** is too greedy. 
# **Expected Improvement (EI)** is standard. 
# **Knowledge Gradient (KG)** is better but expensive to compute for 2000 points without gradients.
#
# Let's try **Cost-Aware UCB** with a **dynamic exploration bonus** based on the ratio of sigma to the 
# current best potential improvement.
#
# Actually, a very effective structural change for multimodal functions is to **normalize the acquisition 
# function by the standard deviation of the posterior** or to use a **rank-based** selection.
#
# Let's try: **Normalized UCB**.
# score = (f_best - mu + beta * sigma) / cost
# The current program uses EI. EI is equivalent to UCB in the limit of high confidence, but it is 
# smoother. 
#
# What if we use **Confidence Bound** but with a **beta** that is derived from the 
# **optimism principle**? 
#
# Let's try a **Hybrid** that blends EI and UCB, but weighted by the **relative uncertainty**.
#
# Actually, let's look at the cost structure: cost = 1 + mean(x).
# This means points with lower x are cheaper.
# If the minimizer is in the inner 80% box, it's not necessarily at x=0.
#
# **New Strategy: Cost-Aware Expected Improvement with Uncertainty Scaling**
# The standard EI can be "saturated" if f_best is very good. 
# We will use **Normalized EI**: EI / sigma. This is equivalent to the Z-score (f_best - mu)/sigma.
# Maximizing Z-score is equivalent to maximizing the Probability of Improvement (PI) in a certain limit? 
# No, PI = Phi(z). Maximizing PI is equivalent to maximizing z.
# EI = (f_best - mu) * Phi(z) + sigma * phi(z).
#
# Let's try **Probability of Improvement (PI)** divided by cost.
# PI = Phi((f_best - mu) / sigma).
# This is more aggressive than EI. It doesn't care *how much* improvement, just the *chance*.
# This might help escape local minima where EI is small because the expected gain is small, 
# but the chance of finding a *better* minimum is still there? No, if you are near a local min, 
# PI is low because mu is close to f_best.
#
# Let's try **UCB** with a **fixed beta = 2.0** (standard for 3D problems often uses 1.0-4.0).
# The current program is EI. UCB is structurally different (additive vs multiplicative-like).
#
# Let's implement **Cost-Aware UCB**.
# Score = (f_best - mu + beta * sigma) / cost
#
# To make it robust, we will use a **beta** that adapts slightly to the spread of sigma.
# beta = 1.0 + 1.0 * (sigma_mean / sigma_std + 1) ? No, too complex.
# Let's use **beta = 2.0**.
#
# Wait, the "inspiration" that improved island 5 used "small penalty (0.1)". 
# The current program uses `ei / cost`. 
# If cost is ~1.5, dividing by cost is a significant penalty.
#
# Let's try **UCB** instead of EI.
#
# Additionally, we can add a **diversity bonus** using the sigma itself.
# If sigma is high, we want to explore. UCB does this.
#
# Let's go with **Cost-Aware UCB**.
#
# One more idea: **Batch-aware UCB**. 
# We don't have x, so we can't do true batch. 
#
# Let's stick to **Cost-Aware UCB** with a slightly higher beta than typical to force exploration 
# of the "inner 80%" box which might be under-explored if the initial points were unlucky.
#
# Actually, a better structural change: **Use the GP posterior to estimate the number of local minima** 
# is not possible.
#
# Let's try **Normalized UCB**:
# z = (f_best - mu) / sigma
# score = (z + beta) / cost
# This is equivalent to (f_best - mu + beta * sigma) / (sigma * cost).
# Dividing by sigma in the denominator penalizes high-uncertainty regions? 
# No, UCB is (f_best - mu + beta * sigma).
#
# Let's use standard **UCB**:
# Score = (f_best - mu + beta * sigma) / cost
#
# Why UCB? EI is "greedy" in a Bayesian sense. UCB is "optimistic". 
# For Hartmann, which has many local minima, optimism can help jump out of a basin 
# if the posterior variance is high in another basin.
#
# We will set beta = 2.0.
#
# To be safe against numerical issues, we clip sigma.

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Aware Upper Confidence Bound (UCB) acquisition function.
    
    Strategy:
    Switch from Expected Improvement (EI) to Upper Confidence Bound (UCB).
    UCB is an optimistic strategy that explicitly rewards high posterior variance (sigma),
    which can help escape local minima in multimodal landscapes like Hartmann 3D.
    By dividing by the cost, we make it cost-aware.
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 280:
        # Fallback: simple greedy exploitation if time is critical
        scores = -mu / cost
        return np.where(np.isfinite(scores), scores, -np.inf)

    # Ensure sigma is non-negative
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Exploration constant beta. 
    # 2.0 is a reasonable default for 3D problems.
    beta = 2.0
    
    # UCB Score: (f_best - mu) + beta * sigma
    ucb = (f_best - mu) + beta * sigma_safe
    
    # Cost-aware score: UCB per unit cost
    scores = ucb / cost
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
