import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
                # Ensure arrays are 1D
                mu = np.asarray(mu).flatten()
                sigma = np.asarray(sigma).flatten()
                cost = np.asarray(cost).flatten()
                
                # Avoid division by zero
                sigma_safe = np.maximum(sigma, 1e-9)
                cost_safe = np.maximum(cost, 1e-9)
                
                # Expected Improvement
                z = (f_best - mu) / sigma_safe
                ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
                
                # Cost-aware EI
                score_ei = ei / cost_safe
                
                # Exploration term
                # Normalize sigma to [0, 1]
                sigma_max = np.max(sigma_safe)
                if sigma_max > 1e-9:
                    sigma_norm = sigma_safe / sigma_max
                else:
                    sigma_norm = np.zeros_like(sigma_safe)
                
                # Adaptive weight based on cost progress
                # Max cost is 30.
                # Initial cost is ~7.5.
                # We want exploration to be high early, low late.
                # cost is the *current* total cost passed in?
                # "cost have shape (2000,)". This is the cost of *each candidate*.
                # Wait. "mu, sigma and cost have shape (2000,)."
                # "Evaluating x costs 1 + mean(x scaled to [0, 1])."
                # So `cost` array contains the cost of each of the 2000 candidates.
                # It does NOT contain the *cumulative* budget spent so far.
                # How do we know the progress?
                # We don't have the cumulative cost in the input signature.
                # We only have `f_best`.
                # We can infer progress from `f_best`?
                # If `f_best` is close to 0, we are late.
                # If `f_best` is high, we are early.
                # Levy 2D min is 0. Max is ~100+.
                # Let's use `f_best` to estimate progress.
                # Assume typical early f_best is ~10-20. Late f_best is < 1.
                # Let's use a smooth decay based on f_best.
                # w = exp(-f_best / 5.0) ?
                # If f_best = 20, w ~ 0.018.
                # If f_best = 1, w ~ 0.81.
                # If f_best = 0.1, w ~ 0.98.
                # This seems backwards. We want MORE exploration when f_best is BAD (high).
                # So if f_best is high, w should be high.
                # w = 1 / (1 + f_best / 5.0) ?
                # f_best=20 -> 0.2.
                # f_best=1 -> 0.83.
                # f_best=0.1 -> 0.98.
                # This encourages exploration when f_best is high.
                
                # Let's try this adaptive weight.
                w = 1.0 / (1.0 + f_best / 5.0)
                
                # Scale exploration term
                # We want the exploration term to be comparable to score_ei.
                # score_ei is in objective units / cost.
                # sigma_norm is dimensionless.
                # We need a scale factor.
                # Let's use the current f_best as a scale?
                # Or just a fixed scale?
                # If we use 0.1 * w * sigma_norm, and f_best is 10, score_ei is ~1.
                # Exploration is ~0.1.
                # If f_best is 0.1, score_ei is ~0.01.
                # Exploration is ~0.1.
                # This shifts balance to exploration as f_best improves.
                
                score = score_ei + 0.1 * w * sigma_norm
                
                # Ensure finite
                score = np.nan_to_num(score, nan=0.0, posinf=1e6, neginf=-1e6)
                
                return score
