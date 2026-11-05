import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Start with a strong static heuristic: score based on degree and neighbor degree.
    2. Use local search to improve the vertex ordering:
       - Simulate the greedy decoder to get an independent set.
       - Try to improve by swapping adjacent vertices in the ordering.
       - Accept swaps that improve the independent set size.
    3. Return the scores corresponding to the best ordering found.
    """
    start_time = time.time()
    DEADLINE = 110.0  # Safety margin before 150s limit
    
    try:
        adj = np.asarray(adjacency, dtype=bool)
        n = adj.shape[0]
        
        if n < 2:
            return np.zeros(n)
        
        # Precompute neighbor lists for fast access
        neighbors = [np.where(adj[i])[0] for i in range(n)]
        
        # Initial scores: prefer low degree, low neighbor degree
        deg = adj.sum(axis=1).astype(float)
        neighbor_deg = adj @ deg
        
        # Score: negative of a cost function
        # Low degree = fewer neighbors blocked
        # Low neighbor degree = the neighbors we block are less important
        alpha = 0.5
        scores = -(deg + alpha * neighbor_deg)
        
        # Add small perturbation to break ties deterministically
        scores += np.linspace(0, 1e-6, n)
        
        def greedy_decode(score_vec):
            """Simulate the greedy decoder given a score vector."""
            # Sort vertices by descending score
            order = np.argsort(-score_vec)
            
            # Greedy selection
            selected = set()
            blocked = set()
            
            for v in order:
                if v in blocked:
                    continue
                # Select v
                selected.add(v)
                # Block all neighbors of v
                for nb in neighbors[v]:
                    blocked.add(nb)
            
            return len(selected), selected
        
        # Initial evaluation
        best_size, best_selected = greedy_decode(scores)
        best_scores = scores.copy()
        
        # Local search: try to improve by swapping adjacent vertices in the ordering
        # We'll work with the ordering directly
        order = np.argsort(-best_scores)
        
        max_iterations = 200000
        improved = True
        
        while improved and time.time() - start_time < DEADLINE:
            improved = False
            
            # Try each adjacent pair in the ordering
            for i in range(n - 1):
                if time.time() - start_time > DEADLINE:
                    return best_scores
                
                # Try swapping order[i] and order[i+1]
                v1 = order[i]
                v2 = order[i + 1]
                
                # Only try if they have different scores (avoid redundant swaps)
                if best_scores[v1] == best_scores[v2]:
                    continue
                
                # Perform the swap
                order[i], order[i + 1] = order[i + 1], order[i]
                
                # Evaluate
                selected = set()
                blocked = set()
                count = 0
                
                for v in order:
                    if v in blocked:
                        continue
                    count += 1
                    selected.add(v)
                    for nb in neighbors[v]:
                        blocked.add(nb)
                
                # If improved, keep the swap
                if count > best_size:
                    best_size = count
                    # Update scores to reflect the new ordering
                    # Assign scores based on position in ordering
                    best_scores = np.zeros(n)
                    for pos, v in enumerate(order):
                        best_scores[v] = -(pos + 1)  # Earlier = higher score
                    improved = True
                else:
                    # Revert the swap
                    order[i], order[i + 1] = order[i + 1], order[i]
                
                # Periodic check
                if i % 10000 == 0 and time.time() - start_time > DEADLINE * 0.9:
                    return best_scores
        
        return best_scores
    
    except Exception:
        # Fallback to simple degree-based score
        try:
            deg = np.sum(adjacency, axis=1)
            return -deg.astype(float)
        except:
            return np.zeros(adjacency.shape[0])
