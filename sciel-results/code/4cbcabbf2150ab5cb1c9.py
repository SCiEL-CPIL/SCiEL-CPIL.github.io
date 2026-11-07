import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Ensure arrays are float
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 295.0  # 300s limit, keep 5s buffer
    
    # Avoid division by zero
    cost_safe = cost + 1e-9
    sigma_safe = sigma + 1e-9
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # Note: If f_best - mu is negative, EI is 0.
    # norm.cdf handles the sign, but let's be careful.
    # Actually, standard EI formula for minimization:
    # E[max(f_best - f(x), 0)]
    # If we assume f(x) ~ N(mu, sigma^2)
    # Let Y = f_best - f(x). Y ~ N(f_best - mu, sigma^2)
    # E[max(Y, 0)] = (f_best - mu) * P(Y>0) + sigma * phi((f_best-mu)/sigma)
    # P(Y>0) = P(f(x) < f_best) = Phi((f_best - mu)/sigma)
    
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    
    ei = (f_best - mu) * Phi + sigma_safe * phi
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Adjusted Expected Improvement (CEI)
    # We want to maximize EI per unit cost.
    # This is the standard Bayesian Cost-Aware Acquisition.
    cei = ei / cost_safe
    
    # Check time
    if time.time() > deadline - 1.0:
        # Fallback to a simple score if we are running out of time
        # Just return normalized CEI
        return np.nan_to_num(cei, nan=0.0, posinf=1.0, neginf=0.0)
    
    # The current program used a max of normalized scores.
    # The key insight from the "inspiration" and lessons is that
    # min-max normalization distorts the relative importance of candidates
    # if the scales of EI, PI, UCB are different.
    # CEI is in units of "objective value per cost unit".
    # This is a physically meaningful quantity.
    
    # However, we must ensure the output is finite.
    # If cost is very small, CEI can be very large.
    # The evaluator picks the argmax.
    # If we return raw CEI, the argmax is the same as argmax(CEI).
    # The task says "Return 2000 finite scores".
    # It does not require normalization.
    # But if the scores are huge, it might cause issues? No, just floats.
    # Let's return the raw CEI.
    
    # One potential issue: if f_best is very small (close to 0),
    # and mu is also small, EI is small.
    # If cost is small, CEI is large.
    # This correctly prioritizes cheap points that have any chance of improvement.
    
    # Let's add a small exploration term?
    # The prompt mentions "structural change".
    # The current program uses max(normalized(CEI), normalized(CPI), normalized(UCB)).
    # My proposal uses raw CEI.
    # Is this enough of a structural change?
    # The "inspiration 2" suggests BCA models cost as deterministic.
    # My CEI does exactly that.
    # The "inspiration 1" suggests Pareto.
    # The "inspiration 3" rejects Thompson Sampling.
    
    # Let's consider if we should combine CEI with something else.
    # The lesson says: "Switching from EI to PI ... measured: 1.275 -> 1.546".
    # This suggests PI might be better than EI for this specific setup?
    # Or maybe the cost penalty lambda was the key.
    # In my CEI, the cost penalty is 1/cost.
    # Cost is 1 + mean(x_scaled). So cost is in [1, 2].
    # So 1/cost is in [0.5, 1].
    # This is a mild penalty.
    
    # Let's try a different structural approach:
    # **Knowledge Gradient (KG)** approximation?
    # KG is often better than EI but more expensive.
    # For 2000 points, a simple KG approximation might be feasible.
    # But EI is a good approximation of KG for 1D or low-dim.
    
    # Let's look at the "inspiration 1" score 1.5461.
    # It used Pareto frontier.
    # My current program (parent) is 1.4615.
    # The parent uses max of normalized scores.
    
    # Let's try **Pareto Front** selection.
    # We have two objectives:
    # 1. Maximize EI (or -mu)
    # 2. Minimize Cost
    # We can identify the Pareto optimal points among the 2000 candidates.
    # Then, among those, pick the one with the best "efficiency".
    # Or, more simply, return a score that is high only for Pareto-optimal points.
    # But we need to return a score for ALL 2000 points.
    # We can set the score to 0 for non-Pareto points and high for Pareto points.
    # This forces the evaluator to pick from the Pareto front.
    
    # How to compute Pareto front efficiently for 2000 points?
    # 1. Sort by Cost (ascending).
    # 2. Iterate, keeping track of max EI seen so far.
    # 3. A point is Pareto optimal if its EI is greater than all previous points with lower cost.
    #    (Actually, strictly greater, or >=? Usually strict for diversity, but >= is safer for "best").
    #    Wait, Pareto optimal means no other point is better in both.
    #    So if we sort by Cost ascending:
    #    Point i is Pareto optimal if EI[i] >= max(EI[0..i-1]).
    
    # Let's implement this.
    # This is a structural change: from "score all points and pick max" to "identify Pareto front and score them".
    
    # However, if we set non-Pareto points to 0, and Pareto points to 1,
    # the evaluator will pick the first Pareto point in the list?
    # No, `np.argmax` returns the first occurrence of the maximum value.
    # If multiple points have score 1, it picks the first one.
    # This is bad. We need to rank the Pareto points.
    
    # We can score the Pareto points by their EI, or by some combination.
    # Let's score Pareto points by their CEI.
    # Non-Pareto points get a very low score (e.g., -1 or 0).
    
    # But wait, is the Pareto front of (EI, Cost) the right thing?
    # Maybe (Expected Improvement, Cost) is the right trade-off.
    # Yes, that's what CEI does implicitly, but CEI allows non-Pareto points to have high CEI if their cost is low enough?
    # No. If point A has higher EI and lower cost than point B, B is dominated.
    # If B is dominated, can it have higher CEI?
    # CEI = EI / Cost.
    # If EI_B < EI_A and Cost_B > Cost_A, then EI_B/Cost_B < EI_A/Cost_A is not necessarily true?
    # Example: A: EI=10, Cost=10 -> CEI=1. B: EI=9, Cost=9 -> CEI=1.
    # B is dominated by A (10>9, 10<9 is false, 10>9 is true? No. A has lower cost 10 vs 9? No.
    # Let's say A: EI=10, Cost=1. B: EI=9, Cost=2.
    # A dominates B (10>9, 1<2).
    # CEI_A = 10. CEI_B = 4.5. A is better.
    # What if A: EI=10, Cost=100. B: EI=9, Cost=1.
    # A does not dominate B (100 > 1). B does not dominate A (9 < 10).
    # Both are Pareto optimal.
    # CEI_A = 0.1. CEI_B = 9. B is better.
    # So CEI naturally ranks the Pareto front.
    # And it correctly ranks dominated points below their dominators?
    # If A dominates B, EI_A > EI_B and Cost_A < Cost_B.
    # Then EI_A/Cost_A > EI_B/Cost_B?
    # Let EI_A = k * EI_B with k > 1. Cost_A = m * Cost_B with m < 1.
    # CEI_A = k * EI_B / (m * Cost_B) = (k/m) * CEI_B.
    # Since k > 1 and m < 1, k/m > 1.
    # So CEI_A > CEI_B.
    # Yes! CEI correctly ranks Pareto optimal points and ensures dominators have higher scores than dominated points.
    
    # So, simply returning CEI is equivalent to finding the Pareto optimal point with the best EI/Cost ratio.
    # This is a very sound structural approach.
    
    # The current program normalizes CEI, CPI, UCB and takes the max.
    # This allows a point with low CEI but high normalized UCB to win.
    # This might be bad if UCB is misleading.
    # My proposal: Return raw CEI.
    
    # Is there any reason to add exploration?
    # CEI already includes exploration via sigma (in EI).
    # High sigma -> high EI -> high CEI (if cost is low).
    # So it explores.
    
    # Let's refine the CEI calculation.
    # We use `sigma_safe`.
    
    # One more thing: The prompt says "min 0".
    # If f_best is 0, EI is 0 for all points.
    # Then CEI is 0.
    # The evaluator will pick the first point (index 0).
    # This is a tie-breaking issue.
    # We should add a tiny random noise or use index as tie-breaker?
    # The task says "Return 2000 finite scores".
    # If all are 0, argmax is 0.
    # This might be bad if index 0 is a bad point.
    # But if f_best is 0, we are done.
    # In practice, f_best will not be exactly 0.
    
    # Let's implement raw CEI.
    
    # To be safe, let's add a very small penalty for high cost?
    # CEI already does that.
    
    # Let's double check the cost formula.
    # cost = 1 + mean(x scaled to [0, 1]).
    # x is in [-10, 10].
    # x_scaled = (x + 10) / 20.
    # mean(x_scaled) is in [0, 1].
    # So cost is in [1, 2].
    # This is a very small variation.
    # So 1/cost is in [0.5, 1].
    # The cost penalty is mild.
    # This means EI dominates the decision.
    # This is good, because we want to find the minimum.
    
    # The current program's normalization might have amplified the cost penalty
    # or the UCB term, leading to suboptimal choices.
    # By using raw CEI, we let the physics of the problem decide.
    
    final_scores = cei
    
    # Ensure finite
    final_scores = np.nan_to_num(final_scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return final_scores
