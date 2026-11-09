"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random


def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Ensure symmetric
    adj = np.maximum(adj, adj.T)
    # Remove self-loops if any
    np.fill_diagonal(adj, 0)
    
    # Precompute degrees
    degrees = adj.sum(axis=1)
    
    # Initialize labels: random or alternating
    # Use alternating as a good start, then randomize slightly?
    # Let's start with alternating
    labels = np.arange(n) % 2
    
    # Compute initial cut size
    # Cut size = sum over edges (i,j) where labels[i] != labels[j]
    # Efficient update: 
    # If we flip vertex v, the change in cut size is:
    # Let same = number of neighbors of v with same label
    # Let other = number of neighbors of v with different label
    # Before flip: v contributes 'other' to cut (edges from v to other side)
    # After flip: v contributes 'same' to cut (edges from v to new other side, which was same side)
    # Wait, if v is in group 0. Neighbors in group 0 are 'same'. Neighbors in group 1 are 'other'.
    # Current contribution of edges incident to v to the cut: 'other' (since edges to group 1 are crossing).
    # If we flip v to group 1:
    #   Edges to old group 0 (now same group as v) become non-crossing.
    #   Edges to old group 1 (now different group from v) become crossing.
    # So new contribution is 'same'.
    # Delta = same - other.
    
    # We can maintain the cut size incrementally.
    # However, for n=100, recomputing is cheap enough, but incremental is better for many iterations.
    
    # Let's implement Simulated Annealing.
    
    def get_cut_size(labels):
        # Count edges where labels differ
        # (labels[:, None] != labels[None, :]) * adj
        # Sum of upper triangle to avoid double counting? 
        # The score is "mean number of edges crossing". Usually cut size is defined as number of edges crossing.
        # If adjacency is symmetric, sum of all (i,j) with i!=j and diff labels / 2? 
        # Or just sum of (i,j) in upper triangle?
        # The prompt says "mean number of edges crossing the partition".
        # In Max Cut, the objective is usually $\sum_{(i,j) \in E} [x_i \neq x_j]$.
        # If the input is a boolean matrix, it likely represents undirected edges.
        # If A[i,j]=1 and A[j,i]=1, the edge is counted once.
        # So cut size = 0.5 * sum_{i,j} A[i,j] * [x_i != x_j].
        # Let's just compute the sum of A[i,j] where i<j and x_i != x_j.
        mask = (labels[:, None] != labels[None, :])
        return np.triu(adj * mask, k=1).sum()
    
    current_cut = get_cut_size(labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Simulated Annealing parameters
    # n=100. Number of iterations should be substantial.
    # Time limit 120s.
    # Let's aim for ~100,000 to 500,000 iterations.
    # Each iteration: pick random v, compute delta, decide.
    # Delta computation:
    # same = adj[v][labels == labels[v]].sum()
    # other = adj[v][labels != labels[v]].sum()
    # delta = same - other
    
    # To speed up, we can precompute neighbor lists or use matrix operations.
    # For n=100, `adj[v][labels == labels[v]].sum()` is very fast.
    
    T_start = 10.0  # High enough to allow moves
    T_end = 0.01
    iterations = 200000
    
    # Schedule: geometric cooling
    # T = T_start * (T_end / T_start) ** (i / iterations)
    cool_factor = (T_end / T_start) ** (1.0 / iterations)
    
    rng = np.random.default_rng(42)
    
    for i in range(iterations):
        # Pick a random vertex
        v = rng.integers(0, n)
        
        # Calculate delta
        current_label = labels[v]
        # Number of neighbors with same label
        same = adj[v][labels == current_label].sum()
        # Number of neighbors with different label
        other = adj[v][labels != current_label].sum()
        
        delta = same - other
        
        if delta > 0:
            # Improvement
            labels[v] = 1 - current_label
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        else:
            # Accept with probability exp(delta / T)
            T = T_start * (cool_factor ** i)
            if rng.random() < np.exp(delta / T):
                labels[v] = 1 - current_label
                current_cut += delta
        
    # Final local search to polish
    # Greedy flip if it improves
    for _ in range(10):
        improved = False
        for v in range(n):
            current_label = labels[v]
            same = adj[v][labels == current_label].sum()
            other = adj[v][labels != current_label].sum()
            if same > other:
                labels[v] = 1 - current_label
                current_cut += (same - other)
                improved = True
        if not improved:
            break
    
    if current_cut > best_cut:
        best_labels = labels.copy()
        best_cut = current_cut
    
    return best_labels
# EVOLVE-BLOCK-END
