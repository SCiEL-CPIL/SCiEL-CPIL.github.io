"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import scipy.sparse.linalg as spla

DEADLINE_MARGIN = 5.0

def _cut_size(labels, adj):
    """Compute the cut size (number of crossing edges) for a labeling."""
    # Vectorized check: edge (i,j) is cut if labels[i] != labels[j]
    # We can sum adj[i,j] where labels differ.
    # For efficiency in the main loop, we track this incrementally.
    # This function is only used for initial verification or final check.
    diff = labels[None, :] != labels[:, None]
    return int(np.sum(adj & diff))

def _initial_gains(labels, adj):
    """
    Compute the initial gain for flipping each vertex.
    gain[v] = (number of neighbors in opposite set) - (number of neighbors in same set)
    If we flip v, the cut changes by gain[v].
    Let s_v = 1 if label is 1, -1 if label is 0.
    neighbor_sum = sum(s_u * adj[v,u])
    gain[v] = -s_v * neighbor_sum
    """
    n = len(labels)
    s = 2.0 * labels - 1.0  # +1 or -1
    # adj is boolean, treat as 0/1 float for multiplication
    adj_float = adj.astype(np.float64)
    neighbor_sum = adj_float @ s
    gains = -s * neighbor_sum
    return gains

def _local_search(labels, adj, deadline):
    """
    Optimized local search using incremental gain updates.
    Returns improved labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    s = 2.0 * labels - 1.0  # Spin representation: +1 or -1
    
    # Precompute adjacency as float for fast matmul/update
    adj_float = adj.astype(np.float64)
    
    # Calculate initial cut
    # Cut = 0.5 * sum(s_i * s_j * adj_ij) * (-1) ... wait.
    # Edge (i,j) contributes to cut if s_i != s_j.
    # If s_i == s_j, s_i*s_j = 1. If s_i != s_j, s_i*s_j = -1.
    # Cut = 0.5 * sum_{i,j} adj_ij * (1 - s_i*s_j) = 0.5 * (TotalEdges - sum(adj_ij * s_i*s_j))
    # Let's just compute it directly for initialization.
    current_cut = _cut_size(labels, adj)
    
    # Initialize gains
    gains = _initial_gains(labels, adj)
    
    # Get neighbors list for fast updates
    # adj is a bool matrix. We need list of neighbors for each node.
    # Converting to CSR helps if we iterate, but we want vectorized max.
    # For incremental update, we need to update gains of neighbors of v.
    # neighbors[v] = indices where adj[v] is True
    
    # Precompute neighbor indices
    neighbors = [np.where(adj[i])[0] for i in range(n)]
    degrees = np.array([len(neigh) for neigh in neighbors])
    
    improved = True
    best_cut = current_cut
    best_labels = labels.copy()
    
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        # Find vertex with max positive gain
        max_gain_idx = np.argmax(gains)
        max_gain = gains[max_gain_idx]
        
        if max_gain <= 0:
            break
            
        v = max_gain_idx
        old_s_v = s[v]
        
        # Flip vertex v
        s[v] = -old_s_v
        labels[v] = 1 - labels[v]
        
        # Update cut
        # The change in cut is exactly max_gain (which is an integer)
        # Note: gains are integers because they are counts of edges.
        current_cut += int(max_gain)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
        # Update gains for neighbors of v
        # For each neighbor u of v:
        # The contribution of edge (v,u) to gain[u] changes.
        # Before flip: edge (v,u) contributed to gain[u] based on s_v.
        # After flip: s_v becomes -s_v.
        # gain[u] = sum_{w in N(u)} s_w * adj[u,w] * (-s_u) ... wait.
        # Let's derive the update rule.
        # gain[u] = -s_u * sum_{w in N(u)} s_w * adj[u,w]
        # When s_v changes by delta_s = -2 * old_s_v, the sum for u changes by delta_s * adj[u,v].
        # So sum_u changes by delta_s * adj[u,v].
        # Therefore gain[u] changes by -s_u * (delta_s * adj[u,v]).
        
        delta_s = -2.0 * old_s_v
        neigh_v = neighbors[v]
        if len(neigh_v) > 0:
            # gains[neigh_v] -= s[neigh_v] * delta_s
            # s[neigh_v] is the current spin of the neighbors.
            gains[neigh_v] -= s[neigh_v] * delta_s
            
            # The gain of v itself also changes?
            # gain[v] = -s_v * sum_{w in N(v)} s_w
            # s_v changed. The sum for v did NOT change (neighbors didn't flip).
            # So gain[v] = -(-old_s_v) * sum_v = old_s_v * sum_v.
            # Previous gain[v] = -old_s_v * sum_v.
            # New gain[v] = -1 * Old gain[v].
            # Let's verify:
            # Old: gain_v = -s_v * S_v
            # New: s_v' = -s_v. S_v is same.
            # New gain_v = -(-s_v) * S_v = s_v * S_v = -(-s_v * S_v) = -Old_gain_v.
            # So gains[v] = -gains[v]
            
            gains[v] = -gains[v]
            
        # Note: We don't need to update gains for non-neighbors.
        # We also don't need to update gain[v] using the neighbor loop logic, 
        # but the formula gains[neigh_v] -= s[neigh_v] * delta_s is for the neighbors.
        # Does the neighbor loop affect v? No, v is not in its own neighbor list (simple graph).
        
    return best_labels, best_cut

def _spectral_init(adj, n, rng):
    """
    Generate a good initial partition using spectral methods.
    """
    try:
        L = csgraph_laplacian(adj, normed=False)
        # Get smallest eigenvectors
        vals, vecs = spla.eigsh(L, k=min(4, n-1), which='SM')
        idx = np.argsort(vals)
        # Use the Fiedler vector (2nd smallest) or a combination
        # Random linear combination of the first few eigenvectors can help
        if len(idx) > 1:
            vec = vecs[:, idx[1]]
        else:
            vec = vecs[:, 0]
        
        # Add some noise to avoid symmetric local optima
        noise = rng.normal(0, 0.1, size=n)
        vec_noisy = vec + noise
        
        labels = (vec_noisy >= 0).astype(int)
        return labels
    except Exception:
        return rng.integers(0, 2, size=n)

def _perturb(labels, n, k, rng):
    """
    Perturb the labels by flipping k random vertices.
    """
    new_labels = labels.copy()
    if k > 0:
        flip_indices = rng.choice(n, size=k, replace=False)
        new_labels[flip_indices] = 1 - new_labels[flip_indices]
    return new_labels

def partition(adjacency):
    """
    Partition vertices into two sets to maximize the number of crossing edges.
    Uses Iterated Local Search with spectral initialization and incremental updates.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    rng = np.random.default_rng(42)
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + Local Search
    # Try a few different spectral seeds
    for i in range(3):
        if time.time() > deadline - 10.0:
            break
        labels = _spectral_init(adj, n, rng)
        labels, cut = _local_search(labels, adj, deadline)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
            
    # Phase 2: Iterated Local Search (ILS)
    # Start from the best solution found so far
    if best_labels is None:
        best_labels = rng.integers(0, 2, size=n)
        best_labels, best_cut = _local_search(best_labels, adj, deadline)
        
    current_labels = best_labels.copy()
    current_cut = best_cut
    
    # Adaptive perturbation strength
    base_k = 2
    max_k = 10
    stagnation = 0
    
    while time.time() < deadline - 5.0:
        # Determine perturbation size
        # Increase k if stuck for a while
        k = base_k + min(stagnation // 5, max_k - base_k)
        
        perturbed_labels = _perturb(current_labels, n, k, rng)
        
        # Local search from perturbed state
        new_labels, new_cut = _local_search(perturbed_labels, adj, deadline)
        
        if new_cut > best_cut:
            best_cut = new_cut
            best_labels = new_labels.copy()
            current_labels = new_labels.copy()
            stagnation = 0
        else:
            stagnation += 1
            # Occasionally accept worse solutions to escape deep local minima?
            # Simple ILS usually just rejects. 
            # Let's try a simple acceptance criterion: accept if close to best
            if new_cut > best_cut - 5:
                current_labels = new_labels.copy()
            else:
                # Revert to current best
                current_labels = best_labels.copy()
                
    return best_labels
# EVOLVE-BLOCK-END
