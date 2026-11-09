import numpy as np
import random
import time

def priority(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([])
    
    # Convert to numpy for fast operations
    adj = np.array(adjacency, dtype=bool)
    degrees = adj.sum(axis=1)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135.0  # 15s margin
    
    # Helper: run greedy decoder given a score vector, return the set picked
    def greedy_decode(scores):
        order = np.argsort(-scores, kind='stable')
        picked = set()
        available = set(range(n))
        for v in order:
            if v in available:
                picked.add(v)
                # Remove v and neighbors
                to_remove = {v}
                nbrs = np.where(adj[v])[0]
                for u in nbrs:
                    if u in available:
                        to_remove.add(int(u))
                available -= to_remove
        return picked
    
    # Initial scores: use negative degree (low degree first) + small random noise
    scores = -degrees.astype(float) + np.random.randn(n) * 0.1
    current_set = greedy_decode(scores)
    current_size = len(current_set)
    
    best_scores = scores.copy()
    best_size = current_size
    
    # Simulated annealing on the score vector
    T = 1.0
    T_min = 0.01
    cooling = 0.995
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        # Perturbation: swap two random vertices' scores
        i, j = random.sample(range(n), 2)
        scores[i], scores[j] = scores[j], scores[i]
        
        new_set = greedy_decode(scores)
        new_size = len(new_set)
        
        delta = new_size - current_size
        
        if delta >= 0 or random.random() < np.exp(delta / T):
            current_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_scores = scores.copy()
        else:
            # Revert
            scores[i], scores[j] = scores[j], scores[i]
        
        T = max(T_min, T * cooling)
        
        # Safety: limit iterations
        if iteration > 50000:
            break
    
    return best_scores
