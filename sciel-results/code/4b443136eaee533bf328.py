"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE = 140.0  # safety margin before 150s kill
_START_TIME = time.time()

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard: if we're too close to the limit, return a safe default
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        # Fallback to simple Best Fit
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores

    # Determine which bins have enough remaining capacity
    valid = bins >= item
    
    if not np.any(valid):
        return np.full_like(bins, -np.inf)

    # Calculate the waste (remaining space) for each valid bin
    waste = bins - item
    
    # Structural Change: Statistical Anticipatory Packing
    # The current approach uses a fixed target waste (e.g., 10.0) or simple best-fit.
    # This new approach models the future: what is the probability that a future item 
    # will fit into the remaining space?
    #
    # We assume the items follow a Weibull distribution. The task mentions "Weibull sequences".
    # Typical Weibull parameters for bin packing tasks often involve shape k=1.0 (Exponential) 
    # or k=2.0 (Rayleigh), and scale lambda such that mean is reasonable (e.g., 10-20).
    # Given capacity 100, a mean item size of ~15-20 is common.
    # Let's estimate the Weibull parameters. If we don't know them exactly, we can use 
    # a robust heuristic based on the current item size as a proxy for the distribution's scale.
    #
    # However, a more robust structural change is to use the **CDF of the assumed distribution** 
    # to score the remaining space.
    # Score = P(future_item <= waste)
    # This favors bins with enough space to likely fit another item, but doesn't necessarily 
    # maximize the space (Worst Fit) because P(x) is bounded by 1 and increases with x.
    # Actually, P(x) is monotonic increasing. So this is similar to Worst Fit if the distribution 
    # is heavy-tailed? No, it saturates.
    #
    # Let's refine: We want to balance "filling the bin" (reducing current waste) with 
    # "keeping the bin useful" (ensuring waste is likely to be filled).
    #
    # A powerful heuristic is the **Expected Future Value** of the bin.
    # If we leave `w` space, the expected value is the probability that a future item fits 
    # times the value of that item? Or simply the probability?
    #
    # Let's try a specific Weibull CDF.
    # Assume a Weibull distribution with shape k=1.5 and scale lambda=15 (mean ~ 15 * Gamma(1+1/1.5) ~ 15*1.18 ~ 17.7).
    # CDF(x) = 1 - exp(-(x/lambda)^k)
    #
    # But we can adapt to the current item size. If the current item is large, the distribution 
    # might be shifted or we are seeing the tail.
    #
    # Alternative Structural Change: **Entropy of the Bin Pool**
    # Instead of looking at individual bin waste, look at the distribution of all valid wastes.
    # If we have many bins with waste ~10, and we put an item in one, we reduce that cluster.
    # If we have diverse wastes, we should aim to reduce the most "valuable" (likely to be filled) 
    # or the most "wasteful" (largest)?
    #
    # Let's try **Probability of Fit** as the score.
    # This is effectively a "Probabilistic Best Fit" if we invert it? No.
    # High probability of fit means large waste (for typical distributions).
    # So this leans towards Worst Fit.
    #
    # Let's try a **Hybrid**: 
    # Score = -waste + alpha * P(future_item fits in waste)
    # This balances minimizing current waste (Best Fit) with maximizing future utility.
    #
    # Let's estimate the Weibull parameters from the item itself? 
    # Or use fixed reasonable defaults for the "100 capacity" task.
    # Common default: k=1.0 (Exponential), lambda=10. Mean=10.
    # CDF(x) = 1 - exp(-x/10)
    #
    # Let's try: Score = -(waste * (1 - CDF(waste)))
    # This is a penalty for waste that is "likely to be filled" (low penalty) vs "likely to be empty" (high penalty).
    # If waste is small, CDF is small, (1-CDF) is large, penalty is small * large = moderate?
    # If waste is large, CDF is ~1, (1-CDF) is ~0, penalty is large * 0 = 0.
    # This would favor LARGE waste (Worst Fit) because the penalty goes to 0.
    # That's bad. We want to favor SMALL waste (Best Fit) usually, unless we expect to fill it.
    #
    # Let's try: Score = CDF(waste) - waste
    # If waste is small, CDF is small, -waste is small negative. Score ~ 0.
    # If waste is large, CDF is ~1, -waste is large negative. Score ~ -large.
    # This favors small waste. It's Best Fit with a soft cap.
    #
    # Let's try a different angle: **Gap to Mean Item Size**
    # If we leave space equal to the mean item size, we have the highest probability of filling it.
    # Score = -|waste - mean_item_size|
    # This is the "Target Fit" strategy with Target = Mean Item Size.
    # In the current program, Target was fixed at 10.
    # Let's estimate Mean Item Size. If we don't know it, we can assume it's proportional to the current item?
    # Or just use a standard value like 15-20 for capacity 100.
    #
    # Let's go with **Target Fit with Adaptive Target**.
    # Target = 15.0 (assumed mean item size for Weibull with cap 100)
    # But to make it structurally different from the current "fixed 10", let's use the **Median** of the valid wastes?
    # No, that's circular.
    #
    # Let's try **Worst Fit Decreasing** style logic? No, we are online.
    #
    # Let's try **Harmonic Mean of Waste**? No.
    #
    # Let's try **Probability of Success** (Fit) multiplied by **Efficiency** (Fill).
    # Score = (waste / max_waste) * CDF(waste)
    # This is high when waste is large AND likely to be filled.
    # But we want to MINIMIZE waste usually.
    #
    # Let's stick to the most robust structural change: **Using the Weibull CDF directly as a non-linear penalty**.
    # The current program uses absolute distance to 10.
    # The new program will use the CDF to weigh the waste.
    #
    # Hypothesis: For Weibull items, the optimal remaining space is not 0 or 10, but related to the distribution's mode or median.
    # Let's use the **Complementary CDF (CCDF)** to penalize waste that is unlikely to be filled.
    # Penalty = waste * (1 - CDF(waste))
    # Score = -Penalty
    #
    # Let's assume Weibull k=1.5, lambda=15.
    # CDF(w) = 1 - exp(-(w/15)**1.5)
    #
    # If w=0, CDF=0, Penalty=0. Score=0.
    # If w=15, CDF~0.63, Penalty=15*0.37=5.55. Score=-5.55.
    # If w=30, CDF~0.95, Penalty=30*0.05=1.5. Score=-1.5.
    # If w=50, CDF~1.0, Penalty=50*0=0. Score=0.
    #
    # This function has a minimum (most negative score, best choice) around w=15-20?
    # Let's check derivative of P(w) = w * exp(-(w/lambda)^k)
    # This is the PDF of the Weibull distribution!
    # So minimizing Penalty is equivalent to maximizing the PDF at the waste point.
    # This means we prefer bins where the remaining space is near the **Mode** of the Weibull distribution.
    #
    # Mode of Weibull(k, lambda) = lambda * ((k-1)/k)^(1/k) for k>1.
    # For k=1.5, lambda=15: Mode = 15 * (0.5/1.5)^(1/1.5) = 15 * (1/3)^(0.666) = 15 * 0.48 = 7.2.
    # For k=2.0, lambda=10: Mode = 10 * (0.5)^(0.5) = 7.07.
    #
    # So this strategy effectively targets a waste of ~7-10.
    # The current program targets exactly 10.
    # Is this different enough? It's a non-linear target.
    #
    # Let's try a different distribution assumption.
    # What if we assume the items are **Uniform**?
    # No, task says Weibull.
    #
    # Let's try **k=1.0 (Exponential)**.
    # Mode is 0. PDF is max at 0.
    # Penalty = w * exp(-w/lambda). Max at w=lambda.
    # So it targets waste = lambda.
    # If lambda=10, it targets waste 10.
    #
    # Let's try **k=2.0 (Rayleigh)**.
    # Mode = lambda / sqrt(2).
    # If lambda=10, Mode = 7.07.
    #
    # The current program uses a fixed target of 10.
    # My new program will use a **dynamic target** based on the assumption of a specific Weibull shape.
    # To make it structurally different, I will **estimate the scale parameter** from the current item size?
    # No, that's risky.
    #
    # Let's try **Worst Fit** but with a cap.
    # Score = -min(waste, 20)
    # This is Best Fit up to 20, then constant.
    #
    # Let's go with the **PDF Maximization** (Mode Targeting) strategy.
    # It is structurally different because it uses a non-linear, distribution-derived penalty 
    # that naturally creates a "valley" at the mode, rather than a V-shape at a fixed constant.
    #
    # Assumption: Weibull k=1.5, lambda=15.
    # Mode ~ 7.2.
    #
    # Let's implement:
    # waste = bins - item
    # # Weibull PDF proxy: w * exp(-(w/lambda)^k)
    # # We want to MAXIMIZE this, so Score = - (w * exp(-(w/lambda)^k))
    # # Wait, if we maximize PDF, we pick the bin with waste closest to Mode.
    # # This is good.
    
    # Parameters
    k = 1.5
    lambda_scale = 15.0
    
    # Calculate PDF-like score
    # Handle w=0 carefully (exp(0)=1, w*1=0, score=0)
    # For large w, exp term goes to 0, score goes to 0.
    # The minimum score (best choice) is at the mode.
    
    # To avoid numerical issues with very large waste, we can just compute it.
    # w is max 100. (100/15)^1.5 ~ 6.6^1.5 ~ 17. exp(-17) is tiny.
    
    pdf_scores = waste * np.exp(-((waste / lambda_scale) ** k))
    
    # We want the bin with the HIGHEST pdf (closest to mode).
    # So Score = -pdf_scores
    scores = -pdf_scores
    
    # Mask invalid bins
    scores[~valid] = -np.inf
    
    return scores
# EVOLVE-BLOCK-END
