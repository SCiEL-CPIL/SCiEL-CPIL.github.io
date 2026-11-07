"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Sample a single function value from the posterior for each candidate.
       y_sample = mu + sigma * z, where z ~ N(0, 1).
    2. Compute the "cost-aware expected improvement" or simply the normalized
       sample value. Since we want to minimize, lower sampled values are better.
       To make it cost-aware, we can interpret the cost as a penalty or divisor.
       
       The task specifies: "Evaluating x costs 1 + mean(x scaled to [0, 1])".
       And "Return 2000 finite scores (higher = evaluate next)".
       
       A common cost-aware acquisition is Expected Improvement per unit cost.
       However, Thompson Sampling is naturally an optimization over the sample.
       
       Let's use a modified approach:
       We want to maximize the score.
       Score = (Potential Improvement) / Cost.
       
       In Thompson Sampling for minimization, we sample y = mu + sigma * z.
       We prefer candidates with low y.
       To incorporate cost, we can treat the "gain" as (f_best - y) if y < f_best, 
       or generally just -y (since lower is better).
       
       A robust cost-aware metric is:
       score = max(0, f_best - y_sample) / cost
       
       However, if y_sample > f_best, the score is 0. This might be too conservative 
       early on if the GP is uncertain. 
       
       Alternative: 
       Use the posterior mean for the improvement estimate but the sample for the 
       uncertainty, or just use the sample directly in a cost-aware UCB-like formula.
       
       Let's stick to a direct Cost-Aware Thompson Sampling interpretation:
       We simulate the evaluation. The "value" of evaluating a point is the 
       reduction in f_best it might provide, divided by the cost.
       
       Since we must return a score for each point to pick the best one:
       1. Generate a single standard normal vector z of size 2000.
       2. Sample y = mu + sigma * z.
       3. Compute improvement = f_best - y.
       4. To avoid negative scores dominating or zeroing out exploration, we can 
          use a soft version or just rely on the fact that sigma is high in 
          unexplored regions. 
          
       Actually, a simpler and very effective cost-aware strategy is:
       score = (f_best - mu + kappa * sigma) / cost  (This is the current one, basically)
       
       How is Thompson Sampling materially different?
       Instead of a fixed kappa, we sample the function. 
       If we sample y, the "best" point is the one with the lowest y.
       But we have costs. 
       
       Let's define the score as:
       score = (f_best - y_sample) / cost
       
       If y_sample > f_best, score is negative. 
       We should ensure the best score is selected. 
       
       To make it more robust, we can add a small exploration term based on the 
       variance of the sample itself? No, the sample IS the exploration.
       
       Let's refine: 
       score = (f_best - (mu + sigma * z)) / cost
       
       This effectively picks the point that, given a random draw from the posterior, 
       has the highest expected improvement per cost.
       
       To handle the case where all scores are negative (which happens if the GP 
       predicts all points are worse than f_best), we might want to add a 
       regularization or ensure we pick the "least bad" one. 
       
       However, if we just return these scores, the evaluator picks the max.
       If all are negative, it picks the one closest to 0 (least bad).
       
       Let's try this. It's a single vectorized operation, very fast.
       """
    # Deadline guard: not strictly necessary for vectorized ops on 2000 points, 
    # but good practice per directive.
    # start_time = time.time()
    
    # Generate a single set of random numbers for all candidates
    # This ensures we are comparing apples to apples in the same "world"
    z = np.random.standard_normal(len(mu))
    
    # Sample from the posterior
    y_sample = mu + sigma * z
    
    # Calculate cost-aware improvement
    # We want to maximize this score
    # Improvement is f_best - y_sample
    # Divide by cost to account for the expense
    # Add a small epsilon to cost to prevent division by zero (though cost >= 1)
    scores = (f_best - y_sample) / (cost + 1e-9)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
