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
        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.sum() == 0 or mask1.sum() == 0:
            return 0.0
        return A[mask0][:, mask1].sum()
    
    def _vectorized_gains(labels):
        """
        Compute flip gain for every vertex in a vectorized way.
        gain[v] = same_side_neighbors[v] - other_side_neighbors[v]
        """
        # For vertex v with label l_v:
        #   same_side = neighbors w with labels[w] == labels[v]
        #   other_side = neighbors w with labels[w] != labels[v]
        #   gain[v] = same_side - other_side
        #
        # Let s1[v] = sum of labels of neighbors of v = (A @ labels)[v]
        # If labels[v] == 0: same_side = degree[v] - s1[v], other_side = s1[v]
        #   gain = (degree - s1) - s1 = degree - 2*s1
        # If labels[v] == 1: same_side = s1[v], other_side = degree[v] - s1[v]
        #   gain = s1 - (degree - s1) = 2*s1 - degree
        #
        # Unified: gain[v] = (2*labels[v] - 1) * (2*s1[v] - degree[v])
        # Wait, let's verify:
        #   labels[v]=0: (2*0-1)*(2*s1 - deg) = -1*(2*s1 - deg) = deg - 2*s1 ✓
        #   labels[v]=1: (2*1-1)*(2*s1 - deg) = 1*(2*s1 - deg) = 2*s1 - deg ✓
        
        deg = A.sum(axis=1)
        s1 = A @ labels  # sum of neighbor labels for each vertex
        gains = (2.0 * labels - 1.0) * (2.0 * s1 - deg)
        return gains
    
    def local_search(labels, max_iters=50):
        """
        Perform local search by flipping individual vertices with positive gain.
        Uses vectorized gain computation for speed.
        Returns improved labels and cut size.
        """
        labels = labels.copy().astype(np.float64)
        current_cut = cut_size(labels.astype(int))
        
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            if time.time() > deadline - 5.0:
                break
            
            gains = _vectorized_gains(labels)
            
            # Find vertices with positive gain
            positive_mask = gains > 1e-9
            if not np.any(positive_mask):
                break
            
            # Flip all vertices with positive gain simultaneously
            labels[positive_mask] = 1.0 - labels[positive_mask]
            current_cut += gains[positive_mask].sum()
            improved = True
        
        return labels.astype(int), current_cut
    
    def simulated_annealing(labels, steps=5000, T_start=2.0, T_end=0.1):
        """
        Simulated annealing to escape local optima.
        """
        labels = labels.copy().astype(np.float64)
        current_cut = cut_size(labels.astype(int))
        best_cut = current_cut
        best_labels = labels.copy()
        
        n_steps = steps
        for i in range(n_steps):
            if time.time() > deadline - 10.0:
                break
            
            T = T_start * (T_end / T_start) ** (i / max(n_steps - 1, 1))
            
            # Pick a random vertex to flip
            v = np.random.randint(0, n)
            labels[v] = 1.0 - labels[v]
            
            new_cut = cut_size(labels.astype(int))
            delta = new_cut - current_cut
            
            if delta >= 0 or np.random.rand() < np.exp(delta / T):
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                labels[v] = 1.0 - labels[v]  # revert
        
        return best_labels.astype(int), best_cut
    
    # Spectral initialization
    D = np.diag(A.sum(axis=1))
    L = D - A
    
    try:
        from scipy.linalg import eigh
        vals, vecs = eigh(L, subset_by_index=[1, 1])
        fiedler = vecs[:, 0]
    except Exception:
        try:
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            vals = vals[idx]
            vecs = vecs[:, idx]
            fiedler = vecs[:, 1]
        except Exception:
            fiedler = np.random.randn(n)
    
    # Initialize labels based on sign of Fiedler vector
    labels_spec = (fiedler > 0).astype(int)
    
    if labels_spec.sum() == 0 or labels_spec.sum() == n:
        labels_spec = np.random.randint(0, 2, n)
    
    # Run local search from spectral initialization
    best_labels, best_cut = local_search(labels_spec, max_iters=100)
    
    # Multi-start with random initializations
    num_restarts = 30
    restart_count = 0
    while restart_count < num_restarts:
        if time.time() > deadline - 20.0:
            break
        
        # Random initialization
        labels_rand = np.random.randint(0, 2, n)
        if labels_rand.sum() == 0 or labels_rand.sum() == n:
            labels_rand[0] = 0
            labels_rand[1] = 1
        
        # Run local search
        labels_ls, cut_ls = local_search(labels_rand, max_iters=100)
        
        if cut_ls > best_cut:
            best_cut = cut_ls
            best_labels = labels_ls
        
        restart_count += 1
    
    # Simulated annealing from best solution
    if time.time() < deadline - 15.0:
        labels_sa, cut_sa = simulated_annealing(best_labels, steps=10000, T_start=3.0, T_end=0.05)
        if cut_sa > best_cut:
            best_cut = cut_sa
            best_labels = labels_sa
        # Polish after SA
        best_labels, best_cut = local_search(best_labels, max_iters=50)
    
    # Final polish
    for _ in range(5):
        if time.time() > deadline - 5.0:
            break
        best_labels, best_cut = local_search(best_labels, max_iters=50)
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
