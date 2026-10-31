"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import scipy
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import weibull_min

# Strategy: Model the problem as a Markov Decision Process where the state is
# the current set of bin remainders. Score each bin based on the expected future
# regret of the remainder it creates, approximated by the probability that this
# remainder cannot be perfectly filled by items from the known Weibull distribution.
#
# Specifically:
# 1. When placing an item in bin j, the new remainder is R_j = bins[j] - item.
# 2. We want to minimize the "regret" of R_j. Regret is high if R_j is a "bad"
#    size that is unlikely to be filled by future items.
# 3. Future items are drawn from a Weibull distribution. We estimate the Weibull
#    parameters (shape k, scale lambda) from the current set of bin remainders,
#    assuming they are representative of the item sizes (or rather, the gaps
#    left behind are correlated with item sizes).
# 4. The probability that a gap of size R can be filled by future items is
#    approximated by the CDF of the Weibull distribution at R. If R is small,
#    the probability that a future item is <= R is low (if R is very small) or
#    high (if R is large). Actually, for filling a gap, we need a sequence of
#    items that sum to R. This is complex.
#
# A simpler proxy for "regret" in online bin packing with known distribution:
# Prefer remainders that are "common" sizes for items, or sizes that are likely
# to be filled.
#
# Let's refine the MDP idea:
# The "value" of a remainder R is how easily it can be utilized.
# If we assume items are Weibull(k, lambda), the density of items of size x is
# f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k).
#
# A remainder R is "good" if it is likely to be filled.
# A simple heuristic: The probability that a random item fits in R is CDF(R).
# However, we need to fill the *entire* gap.
#
# Let's look at the inspirations again.
# Inspiration 1 & 2 used Best-Fit (minimize remaining).
# Inspiration 3 (omitted) likely used a distribution-based approach.
# Current program uses Variance Minimization.
#
# The directive says: "Model the problem as a Markov Decision Process... score
# each bin based on the expected future regret of the remainder it creates,
# approximated by the probability that this remainder cannot be perfectly filled
# by items from the known Weibull distribution."
#
# How to approximate "probability that remainder R cannot be perfectly filled"?
# This is hard. But we can use the Weibull CDF as a proxy for "fillability".
# If R is very small, it's hard to find an item that fits exactly, but items
# are continuous.
#
# Let's try a different angle:
# The "regret" of creating a remainder R is proportional to the "waste" it
# represents in the long run.
# If we assume the items follow a Weibull distribution, we can estimate the
# expected number of items needed to fill a gap of size R.
#
# Actually, a common approach in these evolutionary tasks is to use the
# distribution to weight the best-fit.
#
# Let's combine:
# 1. Estimate Weibull parameters from the current `bins` (remainders).
# 2. For each valid bin, compute the new remainder R.
# 3. Score = - (some function of R based on Weibull).
#
# What function?
# If we want to minimize the number of bins, we want to keep bins full.
# Best-fit (minimize R) is strong.
# Worst-fit (maximize R) is weak.
#
# Maybe the "regret" is high for remainders that are "awkward" sizes.
# Awkward sizes are those with low probability density in the item distribution?
# No, if an item of size R is common, it's likely to be filled.
# So, prefer remainders R where the Weibull PDF f(R) is HIGH?
# Or where the CDF F(R) is in a "good" range?
#
# Let's try: Score = - (bins[j] - item)  [Best Fit]
# Plus a bonus if the resulting remainder R is a "likely" item size.
# Bonus = f(R) / max(f(R) for all valid R)?
#
# Or, simpler:
# Score = - (bins[j] - item) + alpha * f(R)
# where f(R) is the Weibull PDF evaluated at R.
#
# We need to estimate k and lambda.
# Method of moments for Weibull:
# Mean = lambda * Gamma(1 + 1/k)
# Var = lambda^2 * [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2]
#
# We can use the current remainders `bins` to estimate the distribution of
# "gaps". But gaps are not items.
# However, in a steady state, the distribution of gaps might be related to
# the item distribution.
#
# Let's assume the items themselves are Weibull. We don't have the items,
# only the remainders.
# But we can estimate the distribution of the *items* that would have been
# placed if the bins were empty? No.
#
# Let's stick to the directive's specific suggestion:
# "approximated by the probability that this remainder cannot be perfectly filled
# by items from the known Weibull distribution"
#
# Let P_fill(R) be the probability that a gap of size R can be filled.
# This is effectively 1 - P(regret).
# Score = P_fill(R).
#
# How to estimate P_fill(R)?
# If items are Weibull, the probability that a single item fits in R is F(R).
# But we need multiple items.
#
# Let's use a simpler proxy:
# The "regret" is the expected waste.
# If we place item in bin j, new remainder R_j.
# The expected number of additional items needed to fill R_j is roughly
# R_j / E[Item].
# The expected waste after filling is the remainder after the last item.
#
# Let's try a different structural change:
# Use the Weibull CDF to define a "compatibility" score.
# A remainder R is good if it is close to a typical item size.
# Typical item size is the median or mean.
#
# Let's estimate the Weibull parameters from the `bins` array, assuming
# `bins` represents a sample of the item distribution (which is an assumption,
# but often works in these tasks if the packing is tight).
#
# If `bins` are the remainders, they are small. Items are larger.
# So `bins` is NOT a sample of items.
#
# Wait, the task says "five generated Weibull sequences". The items are Weibull.
# We don't have the items. We only have `bins` (remainders).
#
# Can we infer the item distribution from the remainders?
# Not directly.
#
# However, we can use a fixed Weibull shape if we knew it. We don't.
#
# Let's look at the "L1 lower bound" part.
# L1 bound = sum(items) / capacity.
#
# Maybe the "Markov Decision Process" hint implies using the current state
# to predict the future state.
#
# Let's try a hybrid:
# 1. Best-Fit component: - (bins[j] - item)
# 2. Distribution component:
#    Estimate the "typical" item size from the history? We don't have history.
#    Estimate from `bins`?
#    If the packing is efficient, the remainders are small.
#
# Let's try this:
# Assume the items are Weibull with shape k=1.5 (common in such tasks) and
# scale lambda estimated from the mean of `bins` + item? No.
#
# Let's go with the most robust interpretation of the directive:
# Use the Weibull CDF of the *items* to score the remainder.
# Since we don't have items, we must estimate the item distribution.
# A common trick: The mean of the items is roughly (Capacity * Number of Bins Used - Sum(Remainders)) / Number of Items.
# We don't know Number of Items or Bins Used.
#
# Alternative:
# Use the variance of `bins` to estimate the scale of the distribution.
#
# Let's try a simpler structural change that fits the "MDP/Regret" theme:
# Score = - (bins[j] - item)  # Best Fit
#        - beta * (bins[j] - item)^2  # Penalize large remainders quadratically?
#        + gamma * f_weibull(bins[j] - item)  # Bonus for "likely" remainder sizes
#
# We need a Weibull PDF.
# Let's estimate k and lambda from the `bins` array, assuming `bins` is a
# sample of the "gap distribution".
# If gaps are Weibull, then:
# mean_gap = lambda * Gamma(1 + 1/k)
# var_gap = lambda^2 * (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
#
# We can solve for k and lambda from the mean and variance of `bins`.
#
# Then, for each valid bin, compute R = bins[j] - item.
# Compute f(R) using the estimated Weibull PDF.
# Score = -R + alpha * f(R)
#
# This combines Best-Fit (minimize R) with a preference for remainders that
# are "typical" gap sizes (high PDF).
#
# Why would this help?
# If a remainder is a "typical" gap size, it might be more likely to be filled
# by future items that create similar gaps? Or it might just be a heuristic
# that correlates with good packing.
#
# Let's implement this.
#
# Steps:
# 1. Calculate mean and variance of `bins`.
# 2. Estimate Weibull k and lambda from mean and variance.
#    Use a simple iteration or lookup for k.
#    For Weibull, the coefficient of variation (CV) = std/mean is a function of k.
#    CV(k) = sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2) / Gamma(1+1/k)
#    We can invert this to find k.
# 3. Compute f(R) for each valid R.
# 4. Score = -R + alpha * f(R).
#
# Need to handle edge cases (variance 0, etc.).
#
# Let's write the code.

def _estimate_weibull_params(data):
    """
    Estimate Weibull shape k and scale lambda from data mean and variance.
    Returns (k, lambda).
    """
    n = len(data)
    if n < 2:
        return 1.0, 1.0
    
    mean = np.mean(data)
    var = np.var(data)
    
    if mean <= 0 or var <= 0:
        return 1.0, mean if mean > 0 else 1.0
    
    cv = np.sqrt(var) / mean
    
    # Find k such that CV_weibull(k) = cv
    # CV_weibull(k) = sqrt(Gamma(1+2/k) - Gamma(1+1/k)^2) / Gamma(1+1/k)
    # We use a simple bisection search for k in [0.5, 10.0]
    
    def cv_weibull(k):
        try:
            g1 = scipy.stats.gamma(1 + 1/k).mean()  # Gamma(1+1/k)
            g2 = scipy.stats.gamma(1 + 2/k).mean()  # Gamma(1+2/k)
            var_term = g2 - g1**2
            if var_term < 0:
                return 0
            return np.sqrt(var_term) / g1
        except:
            return 0
    
    # Bisection
    lo, hi = 0.5, 10.0
    for _ in range(50):
        mid = (lo + hi) / 2
        cv_mid = cv_weibull(mid)
        if cv_mid > cv:
            lo = mid
        else:
            hi = mid
    
    k = (lo + hi) / 2
    # lambda = mean / Gamma(1 + 1/k)
    g1 = scipy.stats.gamma(1 + 1/k).mean()
    lambda_ = mean / g1
    
    return k, lambda_

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    n = len(bins)
    scores = np.full(n, -1e9)
    
    valid = bins >= item
    
    if not np.any(valid):
        return scores
    
    valid_indices = np.where(valid)[0]
    valid_bins = bins[valid_indices]
    remainders = valid_bins - item
    
    # Estimate Weibull parameters from the current remainders
    # Note: This assumes remainders are representative of the distribution
    # of "gaps", which is a heuristic.
    k, lambda_ = _estimate_weibull_params(bins)
    
    # Compute Weibull PDF for each remainder
    # f(x) = (k/lambda_) * (x/lambda_)^(k-1) * exp(-(x/lambda_)^k)
    # Handle x=0
    pdf_values = np.zeros_like(remainders)
    mask_pos = remainders > 1e-9
    if np.any(mask_pos):
        x = remainders[mask_pos]
        pdf_values[mask_pos] = (k / lambda_) * (x / lambda_) ** (k - 1) * np.exp(-(x / lambda_) ** k)
    
    # Normalize PDF values to [0, 1] for stable weighting
    max_pdf = np.max(pdf_values)
    if max_pdf > 0:
        pdf_norm = pdf_values / max_pdf
    else:
        pdf_norm = pdf_values
    
    # Score = -remainder + alpha * pdf_norm
    # alpha controls the weight of the distribution-based term
    # Start with a small alpha to keep Best-Fit dominant
    alpha = 0.1
    
    scores_valid = -remainders + alpha * pdf_norm
    
    scores[valid_indices] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
