"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Max-Cut using Spectral Initialization + Local Search.
    
    Approach:
    1. Compute the leading eigenvector of the adjacency matrix.
    2. Use the sign of the eigenvector components as initial partition labels.
    3. Refine the partition using a local search (flipping vertices if it improves the cut).
    4. Repeat with a few random restarts to ensure robustness.
    
    This approach leverages the spectral gap to find a good starting point,
    which is often much better than random initialization, and then uses
    fast deterministic local search to polish the result.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Precompute degrees for efficient delta calculation
    degree = adj.sum(axis=1)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # Cut size is the sum of edges where u=0 and v=1 (or vice versa).
        # Since the graph is undirected and adjacency is symmetric, 
        # sum(adj[labels==0][:, labels==1]) gives the correct count.
        return np.sum(adj[labels == 0][:, labels == 1])
    
    def local_search(labels, max_passes=100):
        """
        Refine labels by flipping vertices if it increases the cut size.
        Returns the best labels found during the process.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_cut = current_cut
        best_labels = labels.copy()
        
        # We track the gain of flipping each vertex.
        # Gain(v) = 2 * (neighbors in opposite partition) - degree(v)
        # If Gain(v) > 0, flipping v increases the cut.
        
        for _ in range(max_passes):
            improved = False
            # Calculate gains for all vertices
            # neighbors_in_opposite = adj * (labels != labels[:, None])
            # This is O(N^2). For N=100, this is very fast.
            
            # Efficient calculation:
            # For each vertex v, gain = 2 * sum(adj[v][labels != labels[v]]) - degree[v]
            
            gains = np.zeros(n)
            for v in range(n):
                # Count neighbors in opposite partition
                opp_count = np.sum(adj[v][labels != labels[v]])
                gains[v] = 2.0 * opp_count - degree[v]
            
            # Find the vertex with the highest positive gain
            # If we just want *any* improvement, we can pick the first positive one.
            # Picking the best one (steepest ascent) often converges faster in terms of passes,
            # though first improvement is often better in terms of total operations.
            # Given N=100, steepest ascent is fine.
            
            max_gain_idx = np.argmax(gains)
            max_gain = gains[max_gain_idx]
            
            if max_gain <= 0:
                break # No improvement possible
            
            # Flip the vertex
            labels[max_gain_idx] = 1 - labels[max_gain_idx]
            current_cut += max_gain
            improved = True
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
        return best_labels, best_cut
    
    def spectral_init():
        """
        Generate initial labels based on the leading eigenvector of the adjacency matrix.
        """
        try:
            # Use scipy.linalg.eigh for symmetric matrices
            from scipy.linalg import eigh
            eigenvalues, eigenvectors = eigh(adj)
            # The eigenvector corresponding to the largest eigenvalue
            # is the last column in the returned array (sorted ascending).
            v_max = eigenvectors[:, -1]
            
            # Use the sign of the eigenvector components as labels
            # 0 if negative, 1 if positive
            labels = (v_max > 0).astype(np.int64)
            
            # Handle zeros (rare but possible)
            labels[v_max == 0] = 0
            
            return labels
        except Exception:
            # Fallback to random if spectral fails
            return np.random.randint(0, 2, n)

    best_labels = None
    best_cut = -1
    
    # Strategy: 
    # 1. Try spectral initialization
    # 2. Try a few random initializations
    # We will run local search on each candidate.
    
    candidates = []
    
    # Spectral candidate
    labels_spec = spectral_init()
    candidates.append(labels_spec)
    
    # Random candidates
    num_random_restarts = 5
    for _ in range(num_random_restarts):
        candidates.append(np.random.randint(0, 2, n))
    
    # Also try the complement of spectral, just in case
    candidates.append(1 - labels_spec)
    
    for init_labels in candidates:
        refined_labels, refined_cut = local_search(init_labels)
        if refined_cut > best_cut:
            best_cut = refined_cut
            best_labels = refined_labels
            
    return best_labels
# EVOLVE-BLOCK-END
