import numpy as np
import time

def priority(adjacency):
    """
    Computes vertex scores for the greedy MIS decoder.
    Uses an iterative soft-competition penalty based on log-sum-exp of neighbor scores.
    """
    adjacency = np.asarray(adjacency, dtype=float)
    n = adjacency.shape[0]
    
    # Degree vector
    degrees = adjacency.sum(axis=1)
    
    # Initial scores: negative degree (low degree = high score)
    scores = -degrees.copy()
    
    # Parameters
    alpha = 0.5  # Damping factor
    beta = 1.0   # Competition strength
    iterations = 50
    start_time = time.time()
    deadline = 150.0 - 5.0  # Safety margin
    
    # Helper for stable LogSumExp
    def log_sum_exp(x):
        # x is a 1D array of scores for neighbors
        if len(x) == 0:
            return 0.0 # No neighbors, no penalty
        m = np.max(x)
        if m == -np.inf:
            return 0.0
        return m + np.log(np.sum(np.exp(x - m)))

    for _ in range(iterations):
        if time.time() > deadline:
            break
        
        new_scores = np.zeros(n)
        
        # Iterate over vertices
        for v in range(n):
            # Get neighbors
            neighbors = np.where(adjacency[v] > 0)[0]
            
            if len(neighbors) == 0:
                # Isolated vertex
                target = 0.0 # High priority
            else:
                neighbor_scores = scores[neighbors]
                lse_val = log_sum_exp(neighbor_scores)
                # Penalty: if neighbors have high scores (less negative), lse is higher.
                # We want to LOWER the score of v if neighbors are good.
                # Formula: target = base - beta * lse
                # If lse is -5 (good neighbors), target = -deg - beta(-5) = -deg + 5beta.
                # Wait, earlier analysis said:
                # Good neighbors (score -5) -> lse -5.
                # Bad neighbors (score -20) -> lse -20.
                # We want v to have LOWER score if neighbors are GOOD.
                # So if lse is -5 (good), penalty should be large positive?
                # If we do target = base - beta * lse:
                # Good neighbors: -10 - 1*(-5) = -5.
                # Bad neighbors: -10 - 1*(-20) = 10.
                # This makes v have HIGHER score if neighbors are BAD.
                # This is what we want!
                # Pick v if its neighbors are bad (so we don't waste v on blocking good ones, and v doesn't block good ones).
                target = -degrees[v] - beta * lse_val
            
            new_scores[v] = alpha * scores[v] + (1 - alpha) * target
        
        scores = new_scores
        
        # Optional: Renormalize or shift to prevent drift?
        # The scores might drift. The greedy decoder only cares about relative order.
        # However, LSE is sensitive to absolute values.
        # Shifting all scores by a constant doesn't change LSE relative differences?
        # LSE(x + c) = LSE(x) + c.
        # So target = base - beta(LSE(x) + c) = target_old - beta*c.
        # This shifts all targets.
        # It's safe to leave as is, or center the scores.
        # Let's center to keep numbers small.
        scores -= np.mean(scores)
        
    # Final polish: ensure isolated vertices have top priority
    # They should have been boosted, but let's be sure.
    # Actually, the iteration should handle it.
    
    return scores
