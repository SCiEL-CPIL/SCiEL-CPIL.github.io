import numpy as np
import time
import random

def priority(adjacency):
    N = adjacency.shape[0]
    if N == 0:
        return np.array([])
    
    adj = np.asarray(adjacency, dtype=bool)
    deadline = time.time() + 110.0  # Safety margin before 120s kill

    # Precompute neighbors as lists for fast set operations in Python
    neighbors = [list(np.where(adj[i])[0]) for i in range(N)]
    degrees = np.array([len(neighbors[i]) for i in range(N)], dtype=float)

    # --- Initial Score: Hybrid Heuristic ---
    # 1. Degree: Low degree is better (blocks fewer vertices)
    # 2. Avg Neighbor Degree: High is better (blocks hubs, which is efficient)
    # 3. Spectral: Principal eigenvector (bipartition heuristic)
    
    # Compute Avg Neighbor Degree
    nbr_deg_sums = adj @ degrees
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr_deg = np.where(degrees > 0, nbr_deg_sums / degrees, 0.0)

    # Compute Principal Eigenvector (Perron Vector)
    # For N=200, dense eigh is fast.
    v1 = np.zeros(N)
    try:
        if time.time() < deadline - 5:
            # Use power iteration for speed and robustness, or eigh.
            # eigh is deterministic and fast for 200x200.
            eigvals, eigvecs = np.linalg.eigh(adj.astype(float))
            v1 = eigvecs[:, -1]
            # Normalize
            max_val = np.max(np.abs(v1))
            if max_val > 0:
                v1 = v1 / max_val
    except Exception:
        v1 = np.zeros(N)

    # Combine scores
    # We want high score = pick early.
    # Low degree -> High score. So -degrees.
    # High avg_nbr_deg -> High score. So +avg_nbr_deg.
    # Spectral: The sign of the principal eigenvector often corresponds to a bipartition.
    # Picking one side of the bipartition is a good heuristic for independent sets.
    # If we use the signed value, vertices with high positive v1 will be picked first.
    # This effectively biases the greedy decoder to pick one "color" class.
    # Weights:
    w_deg = 1.0
    w_nbr = 1.0
    w_spec = 2.0  # Boost spectral component to enforce bipartition bias
    
    scores = -w_deg * degrees + w_nbr * avg_nbr_deg + w_spec * v1

    # Helper to evaluate a score vector
    def evaluate_scores(s):
        # Sort indices by descending score
        order = np.argsort(-s, kind='stable')
        blocked = np.zeros(N, dtype=bool)
        count = 0
        for v in order:
            if not blocked[v]:
                count += 1
                blocked[neighbors[v]] = True
        return count

    best_count = evaluate_scores(scores)
    best_scores = scores.copy()

    # --- Local Search: Simulated Annealing / Random Restart ---
    # We will perturb the scores to improve the greedy result.
    # Instead of swapping positions in an order list (which is hard to map back to scores easily
    # without re-sorting), we will directly perturb the score values.
    # A common effective move: pick a random vertex and slightly increase or decrease its score.
    # Or, pick a pair of vertices and swap their scores.
    
    # Given the time budget, we can do many fast evaluations.
    # Each evaluation is O(N^2) worst case (N vertices, N neighbors check), but N=200 is small.
    # 200 * 200 = 40,000 ops. Python loop might be slow.
    # Let's optimize the evaluation loop using numpy if possible, or stick to the fast Python loop.
    # The Python loop with precomputed neighbor lists is quite fast for N=200.
    
    current_scores = best_scores.copy()
    current_count = best_count
    
    # Adaptive temperature / step size
    # Start with large perturbations, shrink over time?
    # Or just random swaps? Swaps preserve the distribution of scores, which might be good.
    # Let's try random swaps of scores between two vertices.
    
    no_improve = 0
    max_no_improve = 5000
    
    while time.time() < deadline and no_improve < max_no_improve:
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        
        # Swap scores
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        new_count = evaluate_scores(current_scores)
        
        if new_count > current_count:
            current_count = new_count
            no_improve = 0
            if current_count > best_count:
                best_count = current_count
                best_scores = current_scores.copy()
        else:
            # Revert
            current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            no_improve += 1
            
            # If stuck for a while, try a larger perturbation or restart from best
            if no_improve > 1000:
                # Restart from best
                current_scores = best_scores.copy()
                current_count = best_count
                no_improve = 0
                # Maybe add some noise to escape local optima?
                # Let's just restart.
                
    # Final check: ensure we return the best found
    return best_scores
