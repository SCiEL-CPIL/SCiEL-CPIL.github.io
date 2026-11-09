"""
Strategy: Greedy Batch Knowledge Gradient (KG) with Cost Awareness.

This strategy replaces the standard Expected Improvement (EI) or Lower Confidence Bound (LCB)
with an approximation of the Knowledge Gradient (KG). 

The KG measures the expected reduction in the global minimum value if we were to evaluate a 
specific candidate point. For a set of candidates with posterior distributions N(mu_i, sigma_i^2),
the expected minimum of the *unobserved* points is lower than the expected minimum of the 
*observed* points. Evaluating a point x_i updates the posterior for x_i (collapsing the uncertainty)
and potentially updates the posteriors of other points via the GP (though in this simplified 
acquisition step, we often approximate the effect locally or assume the "best of the rest" 
is dominated by the current global best).

A computationally efficient approximation for the "value of information" of evaluating point i
is the difference between the expected minimum of the current best (f_best) and the expected 
minimum of the distribution at x_i, weighted by the probability that x_i becomes the new best.

However, a more robust and common approximation for batch-aware acquisition in BO is to use
the "Probability of Improvement" (PI) adjusted for the number of candidates, or simply 
use the fact that EI is a lower bound on KG. 

Here, we implement a specific structural change: **Cost-Adjusted Probability of Being the Global Minimizer**.
Instead of just improving on f_best, we estimate P(x_i is the global min of the 2000 candidates).
This is approximated by integrating over the other candidates. Since we don't have coordinates 
to compute GP correlations between the 2000 candidates easily in this function signature 
(we only have mu, sigma for the candidates), we use a surrogate:

Score_i = (f_best - mu_i + beta * sigma_i) / cost_i  is LCB.
Score_i = EI_i / cost_i is EI.

The KG approximation for the expected value of the minimum of N independent normals is complex.
But there is a simpler structural alternative: **Thompson Sampling**.
Thompson Sampling samples a function from the GP posterior and picks the argmin.
We cannot sample the full GP in 330s for 2000 points repeatedly? Actually, we can.
But we only return scores. We can simulate the "expected" Thompson sample score.
The expected value of the minimum of N samples from N(mu, sigma^2) is not just mu - k*sigma.

Let's try a different structural family: **Expected Improvement with a "Diversity" penalty**.
The current program fails to account for the fact that evaluating a point very close to an
already evaluated high-performing point yields little new information. 
However, we don't have coordinates. We only have mu, sigma, cost.
Wait, the prompt says "No coordinates are passed." 
This is a critical constraint. We cannot compute distances between candidates.
We only have the GP predictions (mu, sigma) for the 2000 candidates.
This implies the "spatial" information is embedded in the mu/sigma vectors themselves 
relative to the training points (which are not passed).
Actually, if we don't have coordinates, we cannot compute a diversity penalty based on distance.
We can only use the statistical properties: mu, sigma, cost.

So, the structural change must be in how we combine mu and sigma.
Current: Linear blend of EI and LCB.
New: **Bayesian Optimization via Gaussian Process UCB with a Non-Linear Exploration Schedule based on Uncertainty Entropy**.

Or better: **The "Expected Regret" minimization via a more accurate tail approximation**.
EI uses the standard normal CDF/PDF. 
Let's try **Probability of Improvement (PI)** but scaled by the *uncertainty* itself, 
not just the potential improvement. 
PI = Phi((f_best - mu) / sigma).
This is bounded [0,1]. It favors points that are *likely* to be better, but not necessarily 
the *most* better. 
If we use PI / cost, we might explore too much.

Let's look at the "Inspiration 1" which scored 1.46. It was a "cost-aware, blended acquisition".
The current program is 1.29.
The difference might be in the cost handling or the blend.

Let's try a **Pure Exploitation with Uncertainty-Driven Step Size**.
In Powell's method, we are doing a line search / coordinate descent.
The GP is a surrogate.
If we assume the landscape is smooth, the points with the lowest mu are the most promising.
Points with high sigma are risky.
A structural change: **Use the "Optimistic" bound (UCB) but with a beta that is a function of the *relative* uncertainty**.

Actually, the most distinct structural change available without coordinates is to change the 
**Objective Function** from "Expected Improvement" to **Expected Simple Regret Reduction**.
For a single step, EI is the expected reduction in the best value found so far.
But we have a budget. We want to minimize the final best value.
This is equivalent to maximizing the expected improvement per cost, which is what EI/cost does.

Let's try **Thompson Sampling Scores**.
We can approximate the expected value of the minimum of the 2000 points *after* evaluating point i.
This is hard.

Let's try a simpler structural change: **Non-linear Cost Penalty**.
The current program uses `sqrt(cost)`. 
What if we use `cost ** 0.25`? Or `log(cost)`?
This is tuning, not structural.

**Structural Change: Switch from Point-Wise Acquisition to a "Batch" Approximation using the Top-K Candidates.**
Since we must return 2000 scores, and the evaluator picks the *best* one, we are effectively doing 
greedy batch selection of size 1.
However, we can exploit the fact that the GP predictions are correlated.
If we don't have coordinates, we can't compute correlation.

Wait, the prompt says "No coordinates are passed."
This is very restrictive.
If I can't see coordinates, I can't compute distances.
I must rely solely on mu, sigma, cost.

Let's look at the "Inspiration 1" code (score 1.46).
It likely used a better blend or cost handling.
The current program (1.29) uses `sqrt(cost)`.
Maybe `cost` should be in the numerator? No, higher cost is worse.

Let's try **Lower Confidence Bound (LCB) with a Dynamic Beta based on the Global Sigma Distribution**.
Instead of a fixed decay or a linear decay, we set beta to be a percentile of the sigma distribution.
For example, beta = k * sigma_max.
This ensures we always explore the most uncertain points relative to the current uncertainty scale.

**Strategy: Adaptive LCB with Global Uncertainty Scaling.**
1. Calculate `sigma_max` and `sigma_mean` from the 2000 candidates.
2. Set `beta = 2.0 * (sigma_max / (sigma_mean + 1e-9))`. This makes beta adaptive to the "spread" of uncertainty.
3. Score = `(-mu + beta * sigma) / cost`.

This is a structural change in how the exploration parameter is determined (global stats vs. time-based decay).

Let's refine this.
If the GP is well-trained, sigma is small everywhere. beta becomes small. Exploitation.
If the GP is poorly trained, sigma is large and variable. beta becomes large. Exploration.
This is more principled than a time-based decay.

Also, we should consider the cost.
If a point has high cost and high sigma, the LCB might still be high.
But dividing by cost handles the efficiency.

Let's implement this.
"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track time and potentially adapt if needed, 
# but the new strategy uses global stats, so state might not be needed 
# except for time guarding.
_start_time = None

def acquisition(mu, sigma, f_best, cost):
    global _start_time
    
    if _start_time is None:
        _start_time = time.time()
        
    elapsed = time.time() - _start_time
    
    # Safety check: if we are close to the limit, return a very fast, simple score
    # The limit is 330s. We assume each call is fast. 
    # But if we are called many times, we might accumulate time.
    # However, the prompt says "Each evaluation must finish within 300s".
    # And "The evaluator kills the program after 330s".
    # So we have a 30s buffer.
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # 1. Compute Global Uncertainty Statistics
    # This is the structural change: beta is derived from the distribution of sigmas
    # rather than a fixed schedule.
    sigma_mean = np.mean(sigma_safe)
    sigma_max = np.max(sigma_safe)
    
    # If sigma is very small everywhere, we are in exploitation mode.
    # If sigma is large, we are in exploration mode.
    # We want beta to scale with the "active" uncertainty.
    # Let's use a robust measure: the 95th percentile of sigma.
    sigma_95 = np.percentile(sigma_safe, 95)
    
    # Beta should be proportional to the typical uncertainty.
    # If sigma_95 is 0, beta should be 0.
    # If sigma_95 is 1, beta should be ~2.
    # Let's normalize sigma_95 by the range of the objective? 
    # We don't know the range. But f_best gives a scale.
    # If f_best is 0, and sigma is 1, the function is very uncertain relative to the best.
    
    # Let's use a simple adaptive beta:
    # beta = 2.0 * (sigma_95 / (abs(f_best) + sigma_95 + 1e-9))
    # This ensures beta is between 0 and 2.
    # If f_best is 0 and sigma is large, beta -> 2.
    # If f_best is -100 (very good) and sigma is small, beta -> 0.
    
    denom = abs(f_best) + sigma_95 + 1e-9
    beta = 2.0 * (sigma_95 / denom)
    
    # Ensure beta is not too small to allow some exploration
    beta = np.maximum(beta, 0.1)
    
    # 2. Compute LCB Score
    # For minimization, we want low mu and high sigma (optimistic).
    # Score = -mu + beta * sigma
    lcb_score = -mu + beta * sigma_safe
    
    # 3. Apply Cost Penalty
    # The previous program used sqrt(cost). 
    # Let's stick with sqrt(cost) as it was part of the 1.29 score, 
    # but the structural change is the adaptive beta.
    # Actually, let's try a softer penalty: cost ** 0.5 is standard.
    # If we want to be more aggressive on cost, use cost ** 0.75.
    # Let's keep sqrt(cost) to isolate the effect of the beta change.
    
    cost_penalty = np.sqrt(cost_safe)
    
    scores = lcb_score / cost_penalty
    
    # 4. Add a small EI component? 
    # The inspiration 1 (1.46) was a "blended acquisition".
    # Let's blend LCB and EI with a fixed weight, but use the adaptive beta for LCB.
    # This combines the structural change (adaptive beta) with the known good blend.
    
    z = (f_best - mu) / sigma_safe
    pi = norm.cdf(z)
    ei = (f_best - mu) * pi + sigma_safe * norm.pdf(z)
    
    # Blend: 0.5 LCB + 0.5 EI
    # The weights can be fixed, as the adaptivity is in beta.
    w_lcb = 0.5
    w_ei = 0.5
    
    blended_scores = w_lcb * lcb_score + w_ei * ei
    
    final_scores = blended_scores / cost_penalty
    
    # Ensure finite values
    final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
    
    # Time guard
    if elapsed > 5.0:
         # Fallback to simple EI/cost
         return ei / cost_safe

    return final_scores
# EVOLVE-BLOCK-END
