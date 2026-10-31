"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 140.0  # Safety margin before 150s kill

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score = better choice.
    
    Strategy: Weibull Survival Minimization (Auction Framework)
    - Treats each bin's remaining capacity as a 'bid' for the item.
    - Scores bins by the negative of the Weibull survival function evaluated at the resulting remainder.
    - The survival function S(x) = exp(-(x/k)^c) represents the probability that a future item 
      (drawn from the Weibull distribution) will be large enough to NOT fit in the remainder x, 
      OR more accurately, we want to minimize the 'waste' potential.
    
    Let's refine the logic:
    If we place an item of size `item` into a bin with current capacity `bin_cap`, 
    the new remainder is `r = bin_cap - item`.
    
    We want to choose `r` such that it is "good".
    A remainder `r` is "bad" if it is large and hard to fill, or small and inefficient?
    Actually, in bin packing, we generally want to minimize the total number of bins.
    This is equivalent to maximizing the density.
    
    The prompt suggests: "scoring bins by the negative of the Weibull survival function 
    evaluated at the resulting remainder to directly minimize the probability that the 
    leftover space becomes permanently unusable waste."
    
    Let the Weibull distribution of item sizes have shape k and scale lambda.
    The survival function is S(x) = P(X > x) = exp(-(x/lambda)^k).
    
    If we interpret "probability that leftover space becomes permanently unusable waste" 
    as the probability that NO future item fits into that space? 
    No, if S(x) is high, it means items are typically larger than x. So if S(r) is high, 
    it's unlikely any future item fits into remainder r. Thus r becomes "dead" waste.
    We want to MINIMIZE this probability of death. 
    So we want to MINIMIZE S(r).
    Since higher score is better, we want to MAXIMIZE -S(r).
    
    Wait, if S(r) is low, it means it's likely that a future item will fit (X < r).
    So we prefer remainders r where S(r) is small.
    Therefore, score = -S(r) = -exp(-(r/lambda)^k).
    
    We need to estimate k and lambda from the problem context.
    The problem states items are from "five generated Weibull sequences of 1000 items with capacity 100".
    Typically, Weibull distributions for such tasks might have specific parameters.
    Without explicit parameters, we can estimate them or use robust defaults.
    Common Weibull parameters for random bin packing benchmarks:
    If items are uniform, k=1 (Exponential).
    If items are more regular, k > 1.
    
    Let's assume a standard Weibull shape k=1.5 and scale lambda=20 (mean ~ 20*Gamma(1+1/1.5) ~ 17-20).
    Actually, let's look at the "Inspiration 3" which used `_weibull_cdf`.
    
    Let's try to adapt the parameters. If we don't know them, we can use a generic approach.
    However, the prompt specifically asks to use the Weibull survival function.
    
    Let's assume k=1.0 (Exponential) first, as it's the most "memoryless" and common default.
    Or better, we can estimate the parameters from the `bins` array if we had history, 
    but we only have current `bins` and `item`.
    
    Let's use a fixed set of parameters that are reasonable for capacity 100.
    If items are Weibull(10, 1.5), mean is ~ 8.
    If items are Weibull(20, 1.0), mean is ~ 20.
    
    Let's pick k=1.5, lambda=15.
    S(r) = exp(-(r/15)**1.5)
    
    Score = -S(r)
    
    We also need a deadline guard.
    """
    if time.time() - _START_TIME > _DEADLINE:
        # Fallback: Simple Best Fit
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full_like(bins, -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    remaining = bins - item
    valid = remaining >= 0
    
    scores = np.full_like(bins, -np.inf, dtype=np.float64)
    
    if not np.any(valid):
        return scores
    
    valid_remaining = remaining[valid]
    
    # Weibull Survival Function: S(x) = exp(-(x/lambda)^k)
    # We want to minimize the probability that the remainder is "dead" (no item fits).
    # P(no item fits) is related to the probability that items are larger than the remainder.
    # If items are i.i.d. Weibull, the probability a single item fits is CDF(r) = 1 - S(r).
    # The probability it remains empty depends on the number of future items, which is unknown.
    # However, minimizing S(r) is a proxy for maximizing the chance that *at least one* item fits soon.
    
    # Parameters for Weibull distribution. 
    # Let's use k=1.5, lambda=15.0 as a heuristic for capacity 100.
    k = 1.5
    lambda_scale = 15.0
    
    # Calculate survival function for valid remainders
    # Handle r=0 carefully: S(0) = 1. Score = -1.
    # Handle large r: S(r) -> 0. Score -> 0.
    # So 0 is the best score (best remainder), -1 is the worst (remainder 0? No, remainder 0 is perfect packing).
    
    # Wait. If r=0, the bin is full. This is GOOD.
    # S(0) = 1. Score = -1.
    # If r is large (e.g., 100), S(100) is very small. Score is close to 0.
    # So this logic favors LARGE remainders? 
    # "Minimize probability that leftover space becomes permanently unusable waste".
    # If r=0, there is NO leftover space. So it cannot be waste. 
    # My previous logic: "S(r) is probability item > r". 
    # If r=0, P(item > 0) = 1. So S(0)=1.
    # Does r=0 become "unusable waste"? No, it's fully used.
    # The "waste" is the space r itself. 
    # If r is large, it is a large chunk of potential waste.
    # If r is small, it is a small chunk of potential waste.
    # So we should prefer SMALL r? 
    # That is Best Fit.
    
    # Let's re-read the prompt's inspiration:
    # "scoring bins by the negative of the Weibull survival function evaluated at the resulting remainder 
    # to directly minimize the probability that the leftover space becomes permanently unusable waste."
    
    # If we interpret "unusable waste" as "space that will never be filled", 
    # then for a remainder r, the probability it stays empty is high if S(r) is high (items are usually too big).
    # So we want to MINIMIZE S(r).
    # Score = -S(r).
    # Maximize Score -> Minimize S(r).
    # S(r) is decreasing in r. 
    # So we prefer LARGE r?
    # This contradicts Best Fit.
    
    # Alternative interpretation:
    # Maybe the "waste" is the bin itself if it's not filled efficiently?
    # Or maybe the Weibull parameterization is different?
    
    # Let's look at Inspiration 3 again. It used `_weibull_cdf`.
    # If we use CDF(r) = P(X <= r) = 1 - S(r).
    # If we want to maximize the chance that a future item fits, we want to MAXIMIZE CDF(r).
    # CDF(r) is increasing in r.
    # So we prefer LARGE r?
    # That is Worst Fit.
    
    # However, Best Fit usually works better for bin packing.
    # Why would Worst Fit be better?
    # If items are small, Worst Fit keeps bins open for larger items.
    # If items are large, Best Fit consolidates.
    
    # The prompt says: "prioritizes remainders that fall within the high-density region of the item size distribution".
    # High density region of Weibull PDF.
    # PDF f(r) = (k/lambda) * (r/lambda)^(k-1) * exp(-(r/lambda)^k).
    # We want to maximize f(r)?
    # This would push r towards the mode of the distribution.
    # Mode = lambda * ( (k-1)/k )^(1/k).
    # For k=1.5, lambda=15: Mode = 15 * (0.5/1.5)^(1/1.5) = 15 * (1/3)^(0.66) = 15 * 0.48 = 7.2.
    # So we prefer remainders around 7.2.
    # This is a "Target Fit" strategy.
    
    # Let's try scoring by the PDF of the Weibull distribution.
    # Score = f(r).
    # This is a structural change from Polarization/Best-Fit to a Distribution-Matching approach.
    
    if valid_remaining.size == 0:
        return scores
        
    # Calculate PDF
    # f(r) = (k/lambda) * (r/lambda)^(k-1) * exp(-(r/lambda)^k)
    # For r=0, if k>1, f(0)=0. If k=1, f(0)=1/lambda. If k<1, f(0)=inf.
    # Weibull with k>1 is unimodal.
    
    # Let's use k=1.5, lambda=15.
    r = valid_remaining
    # Avoid division by zero or log of zero issues if r=0 and k<1, but k=1.5 is safe.
    # f(0) = 0 for k=1.5.
    
    # To handle r=0 (perfect fit), we might want to boost it.
    # A perfect fit (r=0) is ideal. 
    # If we use PDF, r=0 gets 0. 
    # We can add a bonus for r=0.
    
    pdf = np.zeros_like(r)
    mask_nonzero = r > 0
    if np.any(mask_nonzero):
        r_nz = r[mask_nonzero]
        # f(r)
        pdf[mask_nonzero] = (k / lambda_scale) * (r_nz / lambda_scale) ** (k - 1) * np.exp(-(r_nz / lambda_scale) ** k)
        
    # Add a strong bonus for exact fits (r == 0)
    # The PDF at 0 is 0, but exact fit is best.
    # Let's set score for r=0 to be higher than max PDF.
    max_pdf = np.max(pdf) if len(pdf) > 0 else 0
    bonus = max_pdf * 1.5 + 1.0
    
    scores_valid = pdf.copy()
    scores_valid[r == 0] = bonus
    
    scores[valid] = scores_valid
    
    return scores
# EVOLVE-BLOCK-END
