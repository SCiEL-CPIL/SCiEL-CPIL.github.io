"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via multiple random restarts of 1-flip local search.
    Uses a time budget to ensure we return within the limit.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degrees for fast gain calculation
    # cut_size = sum over edges (u,v) where label[u] != label[v]
    # Flipping vertex v changes cut by:
    #   new_cut = old_cut - (edges from v to same side) + (edges from v to other side)
    #   gain = (edges to other) - (edges to same)
    #        = 2 * (edges to other) - degree[v]
    
    def cut_size(labels):
        # Efficient computation using matrix ops
        # Cut edges are those where labels differ
        # For each edge (i,j), it contributes if labels[i] != labels[j]
        # Using: cut = sum(i<j, A[i,j] * (labels[i] != labels[j]))
        # Vectorized: create diff matrix
        # diff[i,j] = 1 if labels[i] != labels[j] else 0
        # labels is 0/1, so labels[i] != labels[j] iff labels[i] XOR labels[j]
        # For 0/1: labels[i] != labels[j] iff (labels[i] + labels[j]) % 2 == 1
        # Or: labels[i] * (1-labels[j]) + (1-labels[i]) * labels[j]
        # Simpler: use the fact that for 0/1 labels, 
        # cut = sum_i (labels[i] * (1-labels[i]) ... no
        # Let's use: cut = 0.5 * sum_i sum_j A[i,j] * (1 - 2*labels[i]*labels[j] + ... )
        # Actually for 0/1: indicator(labels[i]!=labels[j]) = labels[i]*(1-labels[j]) + (1-labels[i])*labels[j]
        # = labels[i] + labels[j] - 2*labels[i]*labels[j]
        # So cut = sum_{i<j} A[i,j] * (labels[i] + labels[j] - 2*labels[i]*labels[j])
        # By symmetry: cut = 0.5 * sum_i sum_j A[i,j] * (labels[i] + labels[j] - 2*labels[i]*labels[j])
        # = 0.5 * (sum_i labels[i] * sum_j A[i,j] + sum_j labels[j] * sum_i A[i,j] - 2 * sum_i sum_j labels[i] A[i,j] labels[j])
        # = sum_i labels[i] * degree[i] - 2 * labels^T A labels ... wait
        # Let me just compute directly for clarity and correctness:
        l = labels.astype(np.int8)
        # diff[i,j] = 1 if l[i] != l[j]
        # l[i] != l[j] iff l[i] + l[j] == 1 (for 0/1)
        # diff = l[:,None] + l[None,:] - 2*l[:,None]*l[None,:]
        # But we only need upper triangle or use symmetry
        # cut = sum_{i<j} A[i,j] * (l[i] != l[j])
        # = 0.5 * sum_{i!=j} A[i,j] * (l[i] != l[j])  (since A is symmetric)
        # = 0.5 * sum_{i,j} A[i,j] * (l[i] + l[j] - 2*l[i]*l[j])
        # = 0.5 * (sum_i l[i] * deg[i] + sum_j l[j] * deg[j] - 2 * l^T A l)
        # = sum_i l[i] * deg[i] - l^T A l
        deg = adj.sum(axis=1)
        return int(np.dot(l, deg) - l @ adj @ l)
    
    def flip_gain(v, labels):
        """Gain in cut size if we flip vertex v."""
        # gain = 2 * (edges from v to other side) - degree[v]
        # edges from v to other side = sum_j A[v,j] * (1 - labels[j]) for j where labels[j] != labels[v]
        # If labels[v] == 0: other side is labels==1, so sum A[v,j]*labels[j]
        # If labels[v] == 1: other side is labels==0, so sum A[v,j]*(1-labels[j]) = deg[v] - sum A[v,j]*labels[j]
        lv = labels[v]
        if lv == 0:
            edges_to_other = int(adj[v] @ labels)
        else:
            edges_to_other = int(adj[v].sum() - adj[v] @ labels)
        deg_v = int(adj[v].sum())
        return 2 * edges_to_other - deg_v
    
    def local_search(labels):
        """1-flip local search until no improving flip."""
        labels = labels.copy()
        improved = True
        while improved:
            improved = False
            for v in range(n):
                g = flip_gain(v, labels)
                if g > 0:
                    labels[v] = 1 - labels[v]
                    improved = True
        return labels
    
    # Time budget: leave 10s margin for safety
    deadline = time.time() + 110.0
    
    best_labels = None
    best_cut = -1
    
    # Number of restarts: try many random initializations
    # Each local search on 100 nodes is fast (~ms)
    n_restarts = 200
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Run local search
        labels = local_search(labels)
        
        # Evaluate
        c = cut_size(labels)
        if c > best_cut:
            best_cut = c
            best_labels = labels.copy()
    
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
