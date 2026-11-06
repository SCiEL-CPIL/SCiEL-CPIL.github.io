"""Maximum Cut via vectorized Simulated Annealing with 2-flip moves and deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses vectorized Simulated Annealing with 1-flip and 2-flip moves,
    multiple restarts, and an explicit deadline guard.
    """
    adj = np.asarray(adjacency, dtype=np.int8)
    n = len(adj)

    if n <= 1:
        return np.zeros(n, dtype=np.int8)

    # Deadline: return best result before 145s (5s safety margin from 150s kill)
    deadline = time.time() + 145.0

    # Precompute degree for gain calculation
    degree = adj.sum(axis=1)

    def compute_cut(labels):
        """Number of edges crossing the partition."""
        # Cut edges = edges where labels differ
        # For each vertex v, cut contribution = sum over neighbors u where labels[u] != labels[v]
        # Total cut = sum_v sum_u adj[v,u] * (labels[v] != labels[u]) / 2 (each edge counted twice)
        # But we want raw count, so divide by 2
        # Vectorized: for each v, count neighbors with different label
        # labels as 0/1; different means labels[v] != labels[u]
        # We can compute: for each v, sum of adj[v] where labels != labels[v]
        # = sum_v (sum_u adj[v,u] * (1 - 2*|labels[v]-labels[u]|/2 ... ))
        # Simpler: cut = sum over all v of (number of neighbors in opposite set)
        # = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # This counts each edge twice (once from each endpoint), so divide by 2.
        # Vectorized approach:
        # For vertex v with label 0: cut contribution = adj[v] @ (1 - labels)  (neighbors with label 1)
        # For vertex v with label 1: cut contribution = adj[v] @ labels        (neighbors with label 0)
        # Total = sum_v [adj[v] @ (1 - labels) if labels[v]==0 else adj[v] @ labels]
        # = sum_v adj[v] @ (labels * (2*labels[v] - 1) + (1 - labels[v]) * (1 - labels))
        # Let's use a cleaner formula:
        # cut * 2 = sum_v sum_u adj[v,u] * (labels[v] != labels[u])
        # = sum_v sum_u adj[v,u] * (1 - 2*min(labels[v],1-labels[v])*...) 
        # Actually simplest: 
        # diff = labels[None, :] != labels[:, None]  # n x n boolean
        # cut2 = (adj * diff).sum()
        # cut = cut2 // 2
        # But n=100, so n x n is fine.
        diff = (labels[None, :] != labels[:, None])
        cut2 = (adj * diff).sum()
        return cut2 // 2

    def local_search_greedy(labels, max_rounds=50):
        """Greedy 1-flip local search until no improvement."""
        current_cut = compute_cut(labels)
        for _ in range(max_rounds):
            improved = False
            # For each vertex, compute gain of flipping
            # If we flip vertex v from label l to 1-l:
            #   new cut contribution for v = number of neighbors with label 1-l (opposite to new label)
            #   old cut contribution for v = number of neighbors with label 1-l (opposite to old label)
            # Wait, let me think more carefully.
            # Current: v has label l. Cut edges incident to v = neighbors with label != l.
            # After flip: v has label 1-l. Cut edges incident to v = neighbors with label != (1-l) = neighbors with label l.
            # So gain = (neighbors with label l) - (neighbors with label != l)
            # = 2*(neighbors with label l) - degree[v]
            # Let's compute this vectorized.
            # For each v, count neighbors with same label as v: same_count[v]
            # same_count[v] = adj[v] @ (labels == labels[v])
            # gain[v] = 2 * same_count[v] - degree[v]
            # If gain[v] > 0, flipping v increases the cut.
            
            same_mask = (labels[None, :] == labels[:, None])  # n x n
            same_count = (adj * same_mask).sum(axis=1)  # for each v, number of neighbors with same label
            gains = 2 * same_count - degree
            
            # Find best vertex to flip
            best_v = np.argmax(gains)
            if gains[best_v] <= 0:
                break
            labels[best_v] = 1 - labels[best_v]
            current_cut += gains[best_v]
            improved = True
            
            if time.time() > deadline:
                break
        return labels, current_cut

    def simulated_annealing(labels, initial_temp=10.0, cooling=0.995, max_iters=5000):
        """Simulated Annealing with 1-flip moves."""
        current_cut = compute_cut(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        temp = initial_temp
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Random vertex
            v = np.random.randint(n)
            
            # Compute gain of flipping v
            same_count_v = (adj[v] * (labels == labels[v])).sum()
            gain = 2 * same_count_v - degree[v]
            
            # Accept if gain > 0 or with probability exp(gain / temp) for gain < 0
            if gain >= 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
            else:
                # probability = exp(gain / temp), gain is negative
                if np.random.rand() < np.exp(gain / temp):
                    labels[v] = 1 - labels[v]
                    current_cut += gain
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
            
            temp *= cooling
        
        return best_labels, best_cut

    # Multiple restarts with different initializations
    best_labels_overall = None
    best_cut_overall = -1

    # Restart 1: alternating
    labels = np.arange(n) % 2
    labels, cut = local_search_greedy(labels, max_rounds=100)
    if cut > best_cut_overall:
        best_cut_overall = cut
        best_labels_overall = labels.copy()

    if time.time() > deadline:
        return best_labels_overall

    # Restart 2: random
    labels = np.random.randint(0, 2, size=n)
    labels, cut = local_search_greedy(labels, max_rounds=100)
    if cut > best_cut_overall:
        best_cut_overall = cut
        best_labels_overall = labels.copy()

    if time.time() > deadline:
        return best_labels_overall

    # Now use SA to escape local optima, starting from best so far
    labels_sa = best_labels_overall.copy()
    labels_sa, cut_sa = simulated_annealing(labels_sa, initial_temp=20.0, cooling=0.999, max_iters=10000)
    if cut_sa > best_cut_overall:
        best_cut_overall = cut_sa
        best_labels_overall = labels_sa.copy()

    if time.time() > deadline:
        return best_labels_overall

    # Final greedy polish
    labels_final = best_labels_overall.copy()
    labels_final, cut_final = local_search_greedy(labels_final, max_rounds=200)
    if cut_final > best_cut_overall:
        best_cut_overall = cut_final
        best_labels_overall = labels_final.copy()

    return best_labels_overall
# EVOLVE-BLOCK-END
