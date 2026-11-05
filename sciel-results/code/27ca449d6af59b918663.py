"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Find a high-quality cut in an undirected graph using spectral initialization
    followed by multi-start local search with simulated annealing-like acceptance.
    
    Parameters:
    -----------
    adjacency : bool matrix (n x n)
        Symmetric adjacency matrix with zeros on diagonal.
        
    Returns:
    --------
    labels : array of 0/1 per vertex
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Convert to float for numerical operations
    A = np.array(adjacency, dtype=np.float64)
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # labels is 0/1 array
        # An edge (i,j) crosses if labels[i] != labels[j]
        # For each vertex i, count neighbors with different label
        # cut = sum over i of (sum of A[i,j] where labels[j] != labels[i]) / 2
        # More efficiently: 
        # For label 0 vertices: sum of edges to label 1 vertices
        # For label 1 vertices: sum of edges to label 0 vertices
        # Total cut = sum(A[labels==0, :][:, labels==1])
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.sum() == 0 or mask1.sum() == 0:
            return 0.0
        return A[mask0][:, mask1].sum()
    
    def local_search(labels, max_iters=50, temperature=0.0):
        """
        Perform local search by flipping individual vertices.
        Returns improved labels and cut size.
        """
        labels = labels.copy()
        current_cut = cut_size(labels)
        
        # Precompute degrees for efficiency
        degrees = A.sum(axis=1)
        
        # For each vertex, we can compute the gain from flipping
        # If vertex v has label l, and s neighbors with same label, o with different
        # Flipping changes cut by: (o - s) because edges that were crossing become non-crossing and vice versa
        # Actually: currently, edges from v to opposite label contribute to cut.
        # After flip, edges from v to same label (now opposite) contribute.
        # So delta = (edges to same) - (edges to opposite) = (degree - edges to opposite) - edges to opposite
        # Wait, let's think again:
        # Currently crossing edges from v: edges to vertices with different label = c_v
        # After flip: crossing edges from v: edges to vertices with same label (old) = d_v - c_v
        # Delta in cut from flipping v = (d_v - c_v) - c_v = d_v - 2*c_v
        # But this is per-vertex, and when we flip, it affects neighbors too.
        # For single flip, the net change is:
        # - Edges from v to same-label neighbors: were not crossing, now crossing: +count_same
        # - Edges from v to opposite-label neighbors: were crossing, now not crossing: -count_opp
        # Net delta = count_same - count_opp
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            for v in range(n):
                if time.time() > deadline - 5.0:
                    break
                label_v = labels[v]
                # Count neighbors with same and different labels
                same_count = A[v][labels == label_v].sum()
                diff_count = A[v][labels != label_v].sum()
                delta = same_count - diff_count
                if delta > 0:
                    labels[v] = 1 - label_v
                    current_cut += delta
                    improved = True
            if time.time() > deadline - 5.0:
                break
        
        return labels, current_cut
    
    # Spectral initialization
    # Use the Fiedler vector (second eigenvector of Laplacian) to initialize
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        # Use scipy for eigendecomposition if available, else numpy
        from scipy.linalg import eigh
        # Get smallest eigenvalues
        # We want the Fiedler vector (second smallest eigenvalue)
        vals, vecs = eigh(L, subset_by_index=[1, 1])
        fiedler = vecs[:, 0]
    except Exception:
        try:
            # Fallback to numpy
            vals, vecs = np.linalg.eigh(L)
            # Sort by eigenvalue
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            fiedler = vecs[:, 1]  # Second smallest
        except Exception:
            fiedler = np.random.randn(n)
    
    # Initialize labels based on sign of Fiedler vector
    labels_spec = (fiedler > 0).astype(int)
    
    # Ensure we don't have all zeros or all ones
    if labels_spec.sum() == 0 or labels_spec.sum() == n:
        # Use random initialization
        labels_spec = np.random.randint(0, 2, n)
    
    # Run local search from spectral initialization
    best_labels, best_cut = local_search(labels_spec, max_iters=100)
    
    # Multi-start with random initializations
    num_restarts = 20
    restart_count = 0
    while restart_count < num_restarts:
        if time.time() > deadline - 10.0:
            break
        
        # Random initialization
        labels_rand = np.random.randint(0, 2, n)
        # Ensure both labels are present
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
            labels_rand[0] = 0
            labels_rand[1] = 1
        
        # Run local search
        labels_ls, cut_ls = local_search(labels_rand, max_iters=100)
        
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls
        
        restart_count += 1
    
    # Additional local search passes on the best solution
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        best_labels, best_cut = local_search(best_labels, max_iters=50)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
