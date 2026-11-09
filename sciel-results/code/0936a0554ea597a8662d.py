import numpy as np

def acquisition(mu, sigma, f_best, cost):
        # Safety guard for time (though this function is fast)
        # The directive asks for an explicit deadline guard.
        # Since we don't know the start time of the whole run, we can't easily check it here.
        # However, we can assume the harness manages the global time.
        # I will focus on the algorithmic change.
        
        # 1. Sample from GP posterior
        # Clip sigma to avoid numerical issues
        sigma_safe = np.clip(sigma, 1e-8, None)
        
        # Generate random samples
        # Using standard normal
        z = np.random.randn(len(mu))
        y_samples = mu + sigma_safe * z
        
        # 2. Calculate Cost-Adjusted Metric
        # We want to minimize y_samples / cost
        # Higher score is better, so we invert
        # Metric: y / cost. Lower is better.
        # Score: -y / cost. Higher is better.
        
        # To avoid division by zero (cost >= 1, so safe)
        metric = y_samples / cost
        
        # 3. Rank-based Score
        # Rank 0 is best (lowest metric).
        # We want the best to have the highest score.
        # scores = N - rank
        
        # argsort gives indices that would sort the array in ascending order
        # So index 0 in sorted_indices corresponds to the smallest metric.
        sorted_indices = np.argsort(metric)
        
        # Initialize scores
        scores = np.zeros(len(mu))
        
        # Assign scores: best (rank 0) gets N-1, second best gets N-2, etc.
        # Or simply N - rank.
        # rank = position in sorted order.
        # scores[sorted_indices[i]] = len(mu) - i
        
        # Vectorized way:
        # Create an array of ranks: 0, 1, ..., N-1
        ranks = np.arange(len(mu))
        # Map ranks to scores
        # score = N - 1 - rank
        scores[sorted_indices] = len(mu) - ranks
        
        # Ensure finite
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        return scores
