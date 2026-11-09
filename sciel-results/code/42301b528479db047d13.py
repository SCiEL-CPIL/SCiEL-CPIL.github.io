import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    # Cost-normalized Lower Confidence Bound (LCB)
    # LCB(x) = mu(x) - beta * sigma(x)
    # We want to minimize LCB, so we maximize -LCB(x) / cost(x)
    # Or equivalently, score = (mu - beta * sigma) / cost is wrong because we want lower mu.
    # Wait, the task is minimization.
    # f_best is the lowest observed value (most negative).
    # We want to find points where f(x) might be lower than f_best.
    # LCB for minimization is mu - beta * sigma.
    # We want to select x that minimizes LCB.
    # The acquisition function returns scores where higher = evaluate next.
    # So we should return -LCB / cost, or (f_best - LCB) / cost?
    # Let's stick to the directive: "directly optimizes the posterior lower bound (mu - beta * sqrt(var)) divided by the evaluation cost".
    # If we optimize the lower bound, we are looking for the most promising low values.
    # Since the loop picks the candidate with the HIGHEST score, and we want to MINIMIZE the objective,
    # the score should be inversely related to the predicted minimum value.
    # Common formulation: Score = - (mu - beta * sigma) / cost
    # However, the directive says "optimizes ... (mu - beta * sigma) ... divided by cost".
    # If I return (mu - beta * sigma) / cost, higher scores correspond to HIGHER predicted minimums, which is bad for minimization.
    # Unless the "optimization" implies finding the best (lowest) value, so the score should be negative of that?
    # Let's look at the current POI program. POI = P(f < f_best). Higher POI is better.
    # For LCB, lower LCB is better. So Score = -LCB / cost makes sense.
    # Or perhaps the directive implies the magnitude of the potential improvement?
    # Let's try Score = (f_best - (mu - beta * sigma)) / cost? No, that's similar to EI/POI.
    # Let's try the direct interpretation of "optimizing the lower bound":
    # We want the candidate with the lowest expected value.
    # So Score = - (mu - beta * sigma) / cost.
    
    # Adaptive beta:
    # A common choice is beta = 2 * sqrt(log(1 + n) / n) or similar.
    # But we don't have n directly, we have cost.
    # Total cost is roughly sum of costs.
    # Let's use a fixed beta that encourages exploration, or an adaptive one.
    # Since we don't know the number of iterations exactly, just the cost,
    # we can estimate progress. But a simple fixed beta is a structural change from POI.
    # Let's try beta = 2.0. This is a standard exploration parameter.
    
    beta = 2.0
    
    # Handle sigma = 0
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # LCB = mu - beta * sigma
    lcb = mu - beta * sigma_safe
    
    # Score: We want to pick the point with the lowest LCB (most promising for minimization).
    # Since the evaluator picks the max score, we negate the LCB.
    # Also normalize by cost.
    scores = -lcb / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
