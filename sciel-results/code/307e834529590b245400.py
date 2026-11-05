import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = 145.0  # Safety margin

    A = np.asarray(adjacency, dtype=bool)
    n = A.shape[0]

    if n == 0:
        return np.array([], dtype=np.float64)

    # Precompute neighbors as lists of indices for fast access
    # A is bool matrix
    neighbors = [np.where(A[i])[0] for i in range(n)]
    degrees = np.array([len(nei) for nei in neighbors])

    # Initial Score: Luby's heuristic (Min Degree first)
    # Higher score is picked first. So we want Low Degree to have High Score.
    # score = -degree works.
    # Add small noise to break ties and initialize SA
    scores = -degrees.astype(np.float64) + np.random.normal(0, 0.01, n)

    def evaluate(s):
        """
        Simulates the fixed greedy decoder.
        Picks vertices in descending order of s.
        Skips neighbors of picked vertices.
        Returns the size of the independent set.
        """
        # Get indices sorted by score descending
        # argsort is stable? We want consistent behavior.
        order = np.argsort(-s)
        
        blocked = np.zeros(n, dtype=bool)
        count = 0
        
        for v in order:
            if not blocked[v]:
                count += 1
                # Block neighbors
                if len(neighbors[v]) > 0:
                    blocked[neighbors[v]] = True
        return count

    best_scores = scores.copy()
    best_val = evaluate(best_scores)

    # Simulated Annealing Parameters
    T = 1.0
    cooling = 0.9995
    min_T = 1e-5

    # We will iterate until time runs out
    # To speed up, we can batch evaluations? No, each depends on previous state.
    # But we can make the inner loop tight.
    
    while time.time() - start_time < deadline:
        # Generate candidate
        # Strategy: 50% Swap, 50% Local Noise
        if np.random.random() < 0.5:
            # Swap two random vertices' scores
            i, j = np.random.randint(0, n, 2)
            if i == j:
                continue
            new_scores = best_scores.copy()
            new_scores[i], new_scores[j] = best_scores[j], best_scores[i]
        else:
            # Add noise to a random vertex
            new_scores = best_scores.copy()
            i = np.random.randint(0, n)
            new_scores[i] += np.random.normal(0, 0.5)

        new_val = evaluate(new_scores)
        delta = new_val - best_val

        # Accept if better, or with probability exp(delta/T)
        if delta > 0 or np.random.random() < np.exp(delta / T):
            best_scores = new_scores
            best_val = new_val

        T *= cooling
        if T < min_T:
            T = min_T

    return best_scores.astype(np.float64)
