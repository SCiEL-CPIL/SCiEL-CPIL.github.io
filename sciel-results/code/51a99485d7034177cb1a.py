import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    Instead of a static heuristic, we use a local search optimization.
    1. Initialize scores with a strong heuristic (e.g., -degree + small noise).
    2. Simulate the greedy decoder to get the current independent set size.
    3. Iteratively perturb the scores (swap priorities or adjust values) and 
       re-simulate. If the new set size is larger, accept the change.
    4. Use a deadline guard to ensure we finish within the time limit.
    
    This approach directly optimizes for the metric (MIS size) rather than 
    relying on a proxy (centrality/degree) that may not perfectly correlate 
    with the optimal greedy ordering.
    """
    start_time = time.time()
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # Precompute neighbor lists for fast simulation
    # neighbors[i] is a list of neighbors of i
    # Using lists for sparse graphs is faster than matrix operations in tight loops
    neighbors = [np.where(adj[i] > 0)[0].tolist() for i in range(n)]
    
    # Initial scoring: Low degree is generally good for MIS.
    # We start with -degree + random noise to break ties and allow search space exploration.
    deg = adj.sum(axis=1)
    scores = -deg + np.random.uniform(0, 1, n)
    
    # Helper function to simulate the greedy decoder
    # Returns the size of the independent set
    def simulate(sc):
        # Get vertices in descending score order
        order = np.argsort(-sc)
        selected = np.zeros(n, dtype=bool)
        blocked = np.zeros(n, dtype=bool)
        count = 0
        
        for u in order:
            if blocked[u]:
                continue
            # Select u
            selected[u] = True
            count += 1
            # Block neighbors
            for v in neighbors[u]:
                if not blocked[v]:
                    blocked[v] = True
        return count

    best_score_val = simulate(scores)
    best_scores = scores.copy()
    
    # Local Search Loop
    # We perform random swaps of scores between two vertices.
    # This changes the order in which vertices are considered.
    # We use a simple hill-climbing with random restarts if stuck, 
    # but given the time limit, we just do continuous random perturbations.
    
    # To speed up, we can limit the number of iterations based on time.
    # Each simulation is O(n^2) in worst case (argsort + neighbor checks).
    # n=200, so n^2=40000. 1000 iterations = 40M ops. Should be fast in Python/NumPy.
    
    # Optimization: Instead of full argsort every time, we could maintain a heap,
    # but for n=200, argsort is very fast (C implementation).
    
    # We will try to improve the score.
    # If we find a better score, we keep it.
    # We use a "temperature" or just simple hill climbing.
    # Since we want the BEST possible, pure hill climbing might get stuck in local optima.
    # However, the landscape is discrete and rugged.
    # Let's try a mix: 90% hill climbing (accept if better), 10% random restarts/perturbations.
    
    # Actually, a simpler and often effective strategy for this specific greedy decoder:
    # The greedy decoder is sensitive to the ORDER of close scores.
    # Randomly swapping two vertices' scores is a valid move.
    
    # Let's run a fixed number of iterations or until time limit.
    # Time limit is 150s. We should be well under 1s for setup.
    # Let's aim for as many iterations as possible.
    
    # To make it faster, we can pre-sort and only update if the swap affects the relative order significantly?
    # No, just do it.
    
    max_time = 145.0  # Safety margin
    
    # We'll use a simple loop.
    # To avoid getting stuck, we can occasionally do a larger perturbation (e.g., shuffle a subset).
    
    current_scores = best_scores.copy()
    current_val = best_score_val
    
    # We track the best found
    # We also keep a history of "no improvement" steps to trigger a restart or larger step
    
    no_improve_count = 0
    
    while time.time() - start_time < max_time:
        # Choose two random vertices to swap scores
        i, j = np.random.randint(0, n, 2)
        if i == j:
            continue
            
        # Swap
        current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
        
        new_val = simulate(current_scores)
        
        if new_val > current_val:
            current_val = new_val
            if new_val > best_score_val:
                best_score_val = new_val
                best_scores = current_scores.copy()
            no_improve_count = 0
        else:
            # Revert if worse (Hill Climbing)
            # Or accept with some probability (Simulated Annealing)?
            # For deterministic-ish results and maximizing the final score, 
            # strict hill climbing is safer to avoid degrading a good solution.
            # However, strict hill climbing gets stuck.
            # Let's use a slight acceptance probability for worse moves if we are stuck.
            if no_improve_count > 50:
                # Accept worse move to escape local optimum
                prob = 0.1
                if np.random.random() < prob:
                    current_val = new_val
                    # Do NOT update best_scores unless it's better than global best
                    no_improve_count = 0
                else:
                    # Revert
                    current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                    no_improve_count += 1
            else:
                # Revert
                current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
                no_improve_count += 1
        
        # Occasionally, if stuck for a long time, do a larger perturbation
        # e.g., swap 5 random pairs
        if no_improve_count > 200:
            for _ in range(5):
                i, j = np.random.randint(0, n, 2)
                if i != j:
                    current_scores[i], current_scores[j] = current_scores[j], current_scores[i]
            new_val = simulate(current_scores)
            if new_val > current_val:
                current_val = new_val
                if new_val > best_score_val:
                    best_score_val = new_val
                    best_scores = current_scores.copy()
                no_improve_count = 0
            else:
                # Revert to best known if this big perturbation failed?
                # Or just keep searching from here?
                # Let's revert to the last good state to stay close to the basin.
                # Actually, if we accepted worse moves, current_scores is already degraded.
                # Let's reset to best_scores if we are stuck too long.
                if no_improve_count > 500:
                    current_scores = best_scores.copy()
                    current_val = best_score_val
                    no_improve_count = 0

    return best_scores

