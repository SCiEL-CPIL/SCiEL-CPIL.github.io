"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Randomized multi-start local search for Max Cut.
    Uses vectorized gain computation for speed.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    adj = np.array(adjacency, dtype=np.int8)
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    
    # Precompute degrees for faster gain calculation
    degrees = adj.sum(axis=1)
    
    best_labels = np.zeros(n, dtype=np.int8)
    best_cut = -1
    
    # Helper to compute cut size
    def compute_cut(labels):
        # Cut size = 0.5 * sum(labels * (degrees - 2*sum(labels*adj))) ? No, simpler:
        # For each edge (i,j), if labels[i] != labels[j], it contributes 1.
        # Vectorized: 
        # Let x be labels in {-1, 1}. Cut = (n^2 - sum(x_i x_j A_ij)) / 4 ? 
        # Easier: 
        # Cut = 0.5 * (sum_{i,j} A_ij (1 - labels[i]labels[j])/2) ?
        # Let's use the standard formula:
        # Cut = 0.5 * sum_{i} sum_{j} A_ij * (labels[i] != labels[j])
        # Using {-1, 1} encoding:
        # diff = 1 if different, 0 if same.
        # (1 - x_i x_j) / 2 is 1 if different, 0 if same.
        # So Cut = 0.5 * sum_{i,j} A_ij * (1 - x_i x_j) / 2 ? No.
        # Sum over all i,j. Since A is symmetric, each edge counted twice.
        # Cut = 0.5 * ( sum_{i,j} A_ij - sum_{i,j} A_ij x_i x_j ) / 2 ?
        # Let x be in {0,1}.
        # Cut = sum_{i<j} A_ij (x_i - x_j)^2
        # This is O(n^2) to compute from scratch, but we can update incrementally in local search.
        # For initial scoring, we can use:
        # cut = 0.5 * ( (x @ A @ (1-x)) + ((1-x) @ A @ x) ) ?
        # Actually, just use:
        # cut = np.sum(adj * (labels[:, None] != labels[None, :])) // 2
        # This creates an n x n boolean matrix. For n=100, it's 10k ops, very fast.
        return np.sum(adj * (labels[:, None] != labels[None, :])) // 2

    # Local search step
    def local_search(labels, adj, degrees, deadline):
        # labels are 0/1
        # Gain for flipping vertex v:
        # If v is 0, it is connected to some 0s and some 1s.
        # Currently contributing: edges to 1s.
        # If flipped to 1, it will contribute: edges to 0s.
        # Gain = (edges to 0s) - (edges to 1s)
        # Let x be labels.
        # neighbors_same = sum(adj[v, x == x[v]])
        # neighbors_diff = sum(adj[v, x != x[v]])
        # Gain = neighbors_diff - neighbors_same
        # Note: neighbors_same + neighbors_diff = degree[v]
        # So Gain = neighbors_diff - (degree[v] - neighbors_diff) = 2 * neighbors_diff - degree[v]
        
        # We can vectorize the calculation of neighbors_diff for all v at once?
        # For a given label vector x (0/1):
        # Let's define y = 2*x - 1 (so -1 and 1)
        # This might be cleaner for matrix operations, but 0/1 is fine.
        
        # Vectorized gain calculation:
        # For each v, we want sum_{u: x[u] != x[v]} A[v,u]
        # This is (A @ (1-x)) for x=0? No.
        # If x[v]=0, we want sum_{u: x[u]=1} A[v,u] = (A @ x)[v]
        # If x[v]=1, we want sum_{u: x[u]=0} A[v,u] = (A @ (1-x))[v]
        
        # So, gain[v] = (A @ (1-x))[v] if x[v]==1 else (A @ x)[v]
        # Let's compute both vectors:
        # sum_to_1 = A @ x  (sum of neighbors that are 1)
        # sum_to_0 = A @ (1-x) (sum of neighbors that are 0)
        # gain = np.where(x == 1, sum_to_0, sum_to_1) - (degrees - np.where(x == 1, sum_to_0, sum_to_1)) ?
        # Wait, Gain = neighbors_diff - neighbors_same
        # neighbors_diff is sum_to_0 if x[v]==1, sum_to_1 if x[v]==0.
        # neighbors_same is degree[v] - neighbors_diff.
        # Gain = 2 * neighbors_diff - degree[v]
        
        labels = labels.copy()
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Calculate gains for all vertices simultaneously
            # x is 0/1
            # sum_to_1[v] = sum of adj[v,u] where x[u]=1
            sum_to_1 = adj @ labels
            # sum_to_0[v] = sum of adj[v,u] where x[u]=0 = degree[v] - sum_to_1[v]
            # But careful: adj[v,v] is 0 usually, but if not, it doesn't matter as x[v] is either 0 or 1.
            # Actually, degree includes self-loop if any. Standard graphs don't have them.
            # neighbors_diff for v:
            # if labels[v] == 1: neighbors_diff = sum_to_0[v] = degrees[v] - sum_to_1[v]
            # if labels[v] == 0: neighbors_diff = sum_to_1[v]
            
            neighbors_diff = np.where(labels == 1, degrees - sum_to_1, sum_to_1)
            gains = 2 * neighbors_diff - degrees
            
            # Find vertex with max positive gain
            max_gain = np.max(gains)
            if max_gain > 0:
                # Pick one (or all) with max gain? Picking one is safer for stability, picking all is faster.
                # Let's pick all with max gain to speed up convergence.
                flip_mask = (gains == max_gain)
                labels[flip_mask] = 1 - labels[flip_mask]
                improved = True
            # else: local optimum reached
        
        return labels

    # Multi-start loop
    rng = np.random.default_rng()
    
    while time.time() < deadline:
        # Random initialization
        labels = rng.integers(0, 2, size=n).astype(np.int8)
        
        # Run local search
        labels = local_search(labels, adj, degrees, deadline)
        
        # Evaluate
        cut = compute_cut(labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
        # Optional: Try a few other quick heuristics if time permits?
        # For now, just random restarts.
        
    return best_labels
# EVOLVE-BLOCK-END
