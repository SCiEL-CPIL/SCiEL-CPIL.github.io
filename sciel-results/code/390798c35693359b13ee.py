# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)

    adj = np.array(adjacency, dtype=np.int8)
    degrees = adj.sum(axis=1)  # degree of each vertex
    deadline = time.time() + 110  # Safety margin: 120s limit, kill at 110s to be safe

    def cut_size(labels):
        # Efficient cut size calculation
        # Cut = sum_{i<j} adj[i,j] * (labels[i] != labels[j])
        # = 0.5 * sum_{i,j} adj[i,j] * (1 - 2*labels[i]*(1-labels[j]) - 2*labels[j]*(1-labels[i]) + 2*labels[i]*labels[j] ... ) 
        # Simpler: 0.5 * sum_i (adj[i] @ (1 - labels) + adj[i] @ labels ... ) 
        # Actually: cut = sum_{i} sum_{j > i} adj[i,j] (labels[i] != labels[j])
        # Using vectorization:
        # Let L = labels. 
        # Cut = 0.5 * ( (L @ adj) @ (1-L) + ((1-L) @ adj) @ L ) ? No.
        # Cut = 0.5 * sum_{i,j} adj[i,j] |L[i] - L[j]|
        # Since L is 0/1, |L[i]-L[j]| = L[i] + L[j] - 2 L[i]L[j] if different? No.
        # If L[i]=0, L[j]=1: diff=1. Formula: 0+1-0=1.
        # If L[i]=1, L[j]=0: diff=1. Formula: 1+0-0=1.
        # If L[i]=0, L[j]=0: diff=0. Formula: 0+0-0=0.
        # If L[i]=1, L[j]=1: diff=0. Formula: 1+1-2=0.
        # So |L[i]-L[j]| = L[i] + L[j] - 2 L[i] L[j] is WRONG for 0/1?
        # 0,0 -> 0. 0,1 -> 1. 1,0 -> 1. 1,1 -> 0.
        # Sum_{i,j} adj[i,j] (L[i] + L[j] - 2 L[i] L[j])
        # = sum_i L[i] (sum_j adj[i,j]) + sum_j L[j] (sum_i adj[i,j]) - 2 sum_{i,j} adj[i,j] L[i] L[j]
        # = sum_i L[i] deg[i] + sum_j L[j] deg[j] - 2 L^T adj L
        # = 2 sum_i L[i] deg[i] - 2 L^T adj L
        # Cut = 0.5 * (2 sum_i L[i] deg[i] - 2 L^T adj L) = sum_i L[i] deg[i] - L^T adj L
        return int(np.sum(labels * degrees) - labels @ (adj @ labels))

    def greedy_flip(labels):
        """Greedy local search with incremental updates."""
        labels = labels.copy()
        # Initialize same_side_count
        # same_side_count[i] = number of neighbors j such that labels[j] == labels[i]
        # We can compute this as:
        # For each i, sum of adj[i] * (labels == labels[i])
        # This is O(N^2) to start.
        same_side_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            same_side_count[i] = np.sum(adj[i] * (labels == labels[i]))
        
        improved = True
        while improved:
            improved = False
            # delta for flipping v: 
            # new_same_side_count[v] = deg[v] - same_side_count[v]
            # delta = 2 * (deg[v] - same_side_count[v]) - deg[v] = deg[v] - 2 * same_side_count[v]
            # Wait, Cut change = -delta?
            # Let's verify.
            # Cut = Total Edges - Non-Cut Edges.
            # Non-Cut Edges = 0.5 * sum_i same_side_count[i] (since each edge counted twice).
            # So maximizing Cut is minimizing Non-Cut Edges.
            # Change in Non-Cut if flip v:
            # Old Non-Cut contribution of edges incident to v: same_side_count[v]
            # New Non-Cut contribution: deg[v] - same_side_count[v]
            # Delta Non-Cut = (deg[v] - same_side_count[v]) - same_side_count[v] = deg[v] - 2*same_side_count[v]
            # We want to *decrease* Non-Cut, so we want Delta Non-Cut < 0.
            # i.e., deg[v] - 2*same_side_count[v] < 0 => 2*same_side_count[v] > deg[v].
            # Let's define gain = -Delta Non-Cut = 2*same_side_count[v] - deg[v].
            # If gain > 0, flipping v reduces Non-Cut, thus increases Cut.
            
            gains = 2 * same_side_count - degrees
            flip_mask = gains > 0
            if np.any(flip_mask):
                # Flip all improving vertices
                flip_indices = np.where(flip_mask)[0]
                for v in flip_indices:
                    labels[v] = 1 - labels[v]
                    # Update same_side_count for neighbors of v
                    # For each neighbor u of v:
                    # If labels[u] == labels[v] (new), they are now same side. Before, they were different (since v flipped).
                    # So same_side_count[u] increases by 1.
                    # If labels[u] != labels[v] (new), they are now different. Before, they were same.
                    # So same_side_count[u] decreases by 1.
                    neighbors = np.where(adj[v] == 1)[0]
                    for u in neighbors:
                        if labels[u] == labels[v]: # Now same side
                            same_side_count[u] += 1
                        else: # Now different side
                            same_side_count[u] -= 1
                    # Update v itself
                    same_side_count[v] = degrees[v] - same_side_count[v]
                improved = True
        return labels

    def simulated_annealing(labels, T_start=10.0, T_end=0.01, cooling=0.95, max_iters=5000):
        """Simulated annealing with incremental updates."""
        labels = labels.copy()
        T = T_start
        best_labels = labels.copy()
        best_cut = cut_size(labels)
        
        # Initialize same_side_count
        same_side_count = np.zeros(n, dtype=np.int32)
        for i in range(n):
            same_side_count[i] = np.sum(adj[i] * (labels == labels[i]))
        
        current_cut = best_cut

        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random vertex
            v = random.randint(0, n - 1)
            
            # Calculate gain for flipping v
            # gain = 2 * same_side_count[v] - degrees[v]
            # If gain > 0, cut increases.
            # Delta Cut = gain.
            delta = 2 * same_side_count[v] - degrees[v]
            
            # Accept if improving, or with probability exp(delta/T) if worsening
            if delta >= 0 or (T > 0 and random.random() < np.exp(delta / T)):
                # Flip v
                labels[v] = 1 - labels[v]
                current_cut += delta
                
                # Update same_side_count for neighbors
                neighbors = np.where(adj[v] == 1)[0]
                for u in neighbors:
                    if labels[u] == labels[v]: # Now same side
                        same_side_count[u] += 1
                    else: # Now different side
                        same_side_count[u] -= 1
                same_side_count[v] = degrees[v] - same_side_count[v]
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()

            T *= cooling
            if T < T_end:
                break

        return best_labels, best_cut

    # Strategy: multiple restarts with different initializations
    best_labels = None
    best_cut = -1

    # First, try greedy from alternating start
    labels_alt = np.arange(n) % 2
    labels_greedy = greedy_flip(labels_alt)
    c = cut_size(labels_greedy)
    if c > best_cut:
        best_cut = c
        best_labels = labels_greedy.copy()

    if time.time() > deadline:
        return best_labels

    # Random restarts with simulated annealing
    n_restarts = 50 # Can afford more now
    for r in range(n_restarts):
        if time.time() > deadline - 2:
            break

        # Random initialization
        labels_init = np.random.randint(0, 2, n).astype(np.int8)

        # Quick greedy improvement first
        labels_g = greedy_flip(labels_init)
        cg = cut_size(labels_g)
        if cg > best_cut:
            best_cut = cg
            best_labels = labels_g.copy()

        # Then simulated annealing from the greedily-improved state
        if time.time() > deadline - 1:
            break
        sa_labels, sa_cut = simulated_annealing(labels_g)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()

    return best_labels
# EVOLVE-BLOCK-END
