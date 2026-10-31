import numpy as np
import time
from scipy.stats import weibull_min

# Global state
_start_time = None
# Weibull parameters estimated for typical online bin packing tasks
# with capacity 100.
# We tune k and lambda to better match the expected item size distribution.
# Based on the observed improvement in similar tasks, a slightly higher shape
# parameter and a slightly larger scale factor might capture the tail behavior
# of the generated Weibull sequences more accurately, leading to a better
# estimate of the "dead space" risk.
_WEIBULL_K = 2.5
_WEIBULL_LAMBDA = 18.0
_WEIBULL_C = _WEIBULL_LAMBDA ** _WEIBULL_K

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _start_time
    
    # Initialize timing if not started
    if _start_time is None:
        _start_time = time.time()
        
    # Deadline guard: 150s limit, keep 5s margin
    elapsed = time.time() - _start_time
    if elapsed > 145:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_indices = np.where(valid)[0]
    new_rems = remaining[valid]
    
    # Adversarial Robustness Strategy:
    # We want to minimize the "Regret" of leaving a remainder `r`.
    # Regret is defined as the probability that a future item `x` 
    # drawn from the Weibull distribution will NOT fit in `r`.
    # P(x > r) = S(r) = 1 - CDF(r)
    # For Weibull(k, lambda): S(r) = exp(-(r/lambda)^k)
    
    # However, simply minimizing S(r) would always prefer the largest remainder (Worst Fit),
    # which is bad for consolidation.
    
    # Instead, we define "Dead Space Risk" as the integral of the survival function
    # from 0 to r? No, that's just the expected value of min(X, r).
    
    # Let's look at the "Gap Efficiency".
    # If we leave remainder r, we are "wasting" the opportunity to use that space
    # if the next item is small.
    
    # A better metric for adversarial robustness in packing:
    # Maximize the "Usable Probability" of the remainder.
    # But we also want to consolidate.
    
    # Let's use a composite score:
    # 1. Penalty for Dead Zone: The area under the PDF between 0 and r? 
    #    If r is small, P(X < r) is small. This doesn't hurt much.
    #    If r is medium (e.g., 5), and most items are < 5, then P(X < r) is high.
    #    This means we could have filled it.
    
    # Let's try: Score = - (P(X < r) + alpha * r)
    # P(X < r) = CDF(r) = 1 - exp(-(r/lambda)^k)
    # This penalizes remainders that are large enough to likely fit a future item.
    # The term alpha * r penalizes large remainders to encourage consolidation.
    
    # If r is very small, CDF(r) is small. Score is high (good).
    # If r is medium, CDF(r) is significant. Score is lower.
    # If r is very large, CDF(r) -> 1. Score is very low.
    
    # This effectively implements a "Best Fit" variant that is aware of the distribution.
    # It prefers bins where the remainder is so small that it's unlikely to fit anything,
    # OR where the remainder is part of a consolidation strategy (though the CDF term
    # dominates for large r).
    
    # Wait, if we prefer small remainders, we are doing Best Fit.
    # Best Fit is usually strong.
    # The current program (Thermodynamic) was trying to avoid the "dead zone" 
    # (0 < r < ENI).
    
    # Let's refine the Adversarial view:
    # The adversary chooses the next item.
    # If we leave r, the adversary can pick an item x <= r.
    # The "damage" is that we didn't use this space for a larger item that might come later?
    # No, online packing doesn't let us reserve.
    
    # Let's stick to the "Dead Space" concept but with distribution awareness.
    # Dead space is r such that P(X <= r) is high (likely to be filled) but we didn't fill it now?
    # No, we ARE filling it now with `item`. The remainder is `r`.
    
    # If `r` is such that many future items fit, we should be happy?
    # Yes, because we can pack more.
    # If `r` is such that few future items fit, it's "dead".
    
    # So we want to MAXIMIZE P(X <= r) ?
    # That would be Worst Fit (maximize r).
    # But Worst Fit is bad for consolidation.
    
    # Let's look at Inspiration 1: "effective capacity".
    # It subtracted expected next item.
    
    # Let's try a different structural change: 
    # **Minimize the Variance of the Remaining Capacity Distribution**
    # No, that's stateful and complex.
    
    # Let's try: **Score based on the "Utilization Efficiency" of the bin after placement.**
    # Utilization = (bin_capacity - r) / bin_capacity = 1 - r/100.
    # We want to maximize Utilization? (Best Fit)
    
    # Let's try a **Hybrid of Best Fit and Distributional Awareness**:
    # Score = - (CDF(r) * r)
    # This penalizes remainders that are both large AND likely to fit items.
    # If r is small, CDF(r) is small, product is small (good score).
    # If r is large, CDF(r) is ~1, product is ~r (bad score).
    # This is essentially Best Fit, but with a non-linear penalty.
    
    # Let's try: **Score = - (1 - CDF(r)) * r**
    # This penalizes remainders that are large AND unlikely to fit items (Dead Space).
    # If r is small, 1-CDF(r) is ~1, product is ~r (small penalty).
    # If r is medium (dead zone), 1-CDF(r) is significant, product is significant.
    # If r is large, 1-CDF(r) is small, product is small?
    # Wait, if r is 50, and lambda=15, (50/15)^2 = 11. 1-CDF = exp(-11) ~ 0.
    # So large r is not penalized by this term.
    # This would favor large r (Worst Fit) for very large r, and small r for small r.
    # It penalizes the middle.
    # This is similar to the Thermodynamic model but using the actual distribution.
    
    # Let's implement: Score = - (exp(-(r/lambda)^k) * r)
    # This is the "Expected Dead Space" if we assume the next item is the only one.
    # Actually, E[min(X, r)] = integral_0^r P(X > t) dt.
    # Let's use the integral of the survival function, which is the expected value of min(X, r).
    # If we leave remainder r, the "waste" is the part of r that won't be used by the next item.
    # Waste = E[max(0, r - X)] = r - E[min(X, r)].
    # We want to minimize Waste.
    # Minimize r - E[min(X, r)].
    # E[min(X, r)] = integral_0^r S(t) dt.
    
    # For Weibull(k, lambda):
    # S(t) = exp(-(t/lambda)^k)
    # integral_0^r exp(-(t/lambda)^k) dt
    # Let u = (t/lambda)^k, t = lambda * u^(1/k), dt = lambda/k * u^(1/k - 1) du
    # Integral = (lambda/k) * Gamma(1/k) * gamma(1/k, (r/lambda)^k) ... incomplete gamma.
    
    # Approximation for small k or using numerical integration?
    # Too slow.
    
    # Let's use a simpler proxy for Waste:
    # If r < lambda, Waste is high.
    # If r > lambda, Waste is low (relative to r).
    
    # Let's try: **Score = - (r * exp(-(r/lambda)^k))**
    # This is the "Dead Space" metric derived above.
    # It creates a valley (high score) at r=0 and r->infinity.
    # It has a peak (low score) at some r*.
    # This encourages either perfect fit or very large remainders.
    # This is structurally different from the previous "min distance to 0 or ENI".
    # It uses the actual Weibull tail.
    
    # Calculate scores
    # r = new_rems
    # score = - (r * exp(-(r/lambda)^k))
    
    # Add a small consolidation bias to break ties between large remainders?
    # If we prefer consolidation, we should prefer smaller r among the "large" ones.
    # But the function already decreases for r > r*.
    # Derivative of f(r) = r * exp(-(r/lambda)^k)
    # f'(r) = exp(...) + r * exp(...) * (-k/lambda^k) * r^(k-1)
    # f'(r) = exp(...) * (1 - k * (r/lambda)^k)
    # Zero at (r/lambda)^k = 1/k => r = lambda * k^(-1/k)
    # For k=2, lambda=15: r* = 15 * 2^(-0.5) = 15 * 0.707 = 10.6.
    # So the penalty is maximized at r ~ 10.6.
    # Scores are high at r=0 and r=100.
    # This might be too aggressive towards Worst Fit for large bins.
    
    # Let's add a linear penalty for large r to encourage consolidation.
    # Score = - (r * exp(-(r/lambda)^k) + beta * r)
    # If beta is small, it still favors large r.
    # If beta is large, it favors small r.
    
    # Let's try beta = 0.1.
    
    exp_term = np.exp(-np.power(new_rems / _WEIBULL_LAMBDA, _WEIBULL_K))
    dead_space = new_rems * exp_term
    
    # Refine the consolidation penalty.
    # The previous linear penalty (0.1 * r) was a simple heuristic.
    # We can model the cost of fragmentation more accurately.
    # A larger remainder r is "wasted" if it cannot be filled by future items.
    # The probability that a future item x fits in r is CDF(r).
    # So the "expected waste" is r * (1 - CDF(r)) = r * exp(-(r/lambda)^k).
    # This is exactly the `dead_space` term!
    # So, `dead_space` already captures the expected waste due to the distribution.
    # Adding a linear penalty might be double-counting or distorting the signal.
    
    # However, we also want to encourage consolidation to avoid opening new bins.
    # A pure `dead_space` minimization might still lead to a spread of bins
    # if the distribution is wide.
    # Let's try a non-linear penalty that is stronger for larger remainders,
    # reflecting the higher opportunity cost of leaving large gaps.
    # A quadratic penalty: 0.01 * r^2.
    # Or a penalty based on the "probability of being dead" itself?
    
    # Let's try removing the linear penalty and see if the distributional term
    # is sufficient, or adjust its weight.
    # The finding "Replaces linear consolidation bias with a probabilistic fragmentation penalty"
    # suggests that the probabilistic term is the key.
    # Let's trust the `dead_space` term more and reduce the arbitrary linear penalty.
    
    # Let's try a very small linear penalty to break ties or slight preference for smaller bins.
    consolidation_penalty = 0.01 * new_rems
    
    scores_valid = -(dead_space + consolidation_penalty)
    
    # Tie-breaker: prefer lower index
    scores_valid -= np.arange(len(valid_indices)) * 1e-9
    
    scores[valid] = scores_valid
    
    return scores
