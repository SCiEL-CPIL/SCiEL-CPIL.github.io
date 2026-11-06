"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def _local_search(labels, adjacency, time_limit, rng):
    """
    Perform local search (flip individual vertices) to improve cut size.
    Uses a time limit to stop before exceeding the budget.
    Optimized: maintain cut value incrementally.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int32)
    
    # Compute initial cut size
    diff = labels[:, None] != labels[None, :]
    current_cut = int((adj * diff).sum() // 2)
    
    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Track number of edges from each vertex to the opposite set
    opp_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_count[v] = adj[v][labels != labels[v]].sum()
    
    # Precompute neighbor lists for faster access
    neighbor_lists = [np.where(adj[v] > 0)[0] for v in range(n)]
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time.time() - start_time > time_limit:
                break
            # Gain from flipping v
            gain = degrees[v] - 2 * opp_count[v]
            if gain > 0:
                # Flip vertex v
                labels[v] = 1 - labels[v]
                current_cut += gain
                
                # Update opp_count for v and its neighbors
                opp_count[v] = degrees[v] - opp_count[v]
                
                neighbors = neighbor_lists[v]
                for u in neighbors:
                    if labels[u] == labels[v]:
                        opp_count[u] -= 1
                    else:
                        opp_count[u] += 1
                
                improved = True
        if time.time() - start_time > time_limit:
            break
    
    return labels, current_cut


def _simulated_annealing(labels, adjacency, time_limit, rng):
    """
    Simulated annealing to escape local optima.
    """
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int32)
    
    # Compute initial cut size
    diff = labels[:, None] != labels[None, :]
    current_cut = int((adj * diff).sum() // 2)
    
    # Precompute degrees
    degrees = adj.sum(axis=1).astype(np.int32)
    
    # Track number of edges from each vertex to the opposite set
    opp_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        opp_count[v] = adj[v][labels != labels[v]].sum()
    
    # Precompute neighbor lists
    neighbor_lists = [np.where(adj[v] > 0)[0] for v in range(n)]
    
    start_time = time.time()
    
    # Temperature schedule
    T0 = max(1.0, float(degrees.max()) if n > 0 else 1.0)
    T_min = 0.01
    cooling_rate = 0.995
    
    T = T0
    best_labels = labels.copy()
    best_cut = current_cut
    
    while T > T_min:
        if time.time() - start_time > time_limit:
            break
        
        # Pick a random vertex to flip
        v = rng.integers(0, n)
        
        # Compute gain from flipping v
        gain = degrees[v] - 2 * opp_count[v]
        
        # Decide whether to accept the flip
        if gain >= 0:
            accept = True
        else:
            accept_prob = np.exp(gain / T)
            accept = rng.random() < accept_prob
        
        if accept:
            labels[v] = 1 - labels[v]
            current_cut += gain
            
            opp_count[v] = degrees[v] - opp_count[v]
            
            neighbors = neighbor_lists[v]
            for u in neighbors:
                if labels[u] == labels[v]:
                    opp_count[u] -= 1
                else:
                    opp_count[u] += 1
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
        
        T *= cooling_rate
    
    return best_labels, best_cut


def _spectral_init(adjacency, rng):
    """
    Generate initial 0/1 labels using the Fiedler vector (second-smallest
    eigenvector of the graph Laplacian).
    """
    n = len(adjacency)
    if n <= 1:
        return np.zeros(n, dtype=int)
    
    adj = adjacency.astype(np.float64)
    degrees = adj.sum(axis=1)
    
    # Build the Laplacian L = D - A
    L = np.diag(degrees) - adj
    
    # Use eigsh for the smallest eigenvalues
    try:
        # k=2: we want the second-smallest eigenvalue's eigenvector
        k = min(2, n - 1)
        if k < 2:
            # Fallback for very small graphs
            return rng.integers(0, 2, size=n)
        
        # Use shift-invert for smallest eigenvalues
        vals, vecs = eigsh(L, k=k, which='SM', sigma=0, return_eigenvectors=True)
        
        # The Fiedler vector is the eigenvector for the second-smallest eigenvalue
        # eigsh returns eigenvalues in ascending order when which='SM'
        fiedler = vecs[:, 1]
    except Exception:
        # Fallback to random if spectral method fails
        return rng.integers(0, 2, size=n)
    
    # Threshold the Fiedler vector at its mean to get a 0/1 labeling
    threshold = np.mean(fiedler)
    labels = (fiedler > threshold).astype(int)
    
    # Ensure both labels are present (avoid degenerate all-0 or all-1)
    if labels.sum() == 0 or labels.sum() == n:
        # Flip the vertex with the largest magnitude deviation from threshold
        idx = np.argmax(np.abs(fiedler - threshold))
        labels[idx] = 1 - labels[idx]
    
    return labels


def partition(adjacency):
    n = len(adjacency)
    global_deadline = time.time() + 110
    rng = np.random.default_rng(seed=42)
    
    best_labels = None
    best_cut = -1
    
    num_restarts = 0
    max_restarts = 100
    
    # Phase 1: Spectral initialization + local search
    # First, try spectral initialization
    if n > 1:
        spectral_labels = _spectral_init(adjacency, rng)
        remaining_time = max(1.0, (global_deadline - 30) - time.time())
        ls_time_limit = max(0.05, remaining_time / 15)
        refined_labels, cut = _local_search(spectral_labels, adjacency, ls_time_limit, rng)
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        num_restarts += 1
    
    # Then try other initializations
    while time.time() < global_deadline - 30 and num_restarts < max_restarts:
        if num_restarts == 1:
            labels = np.arange(n) % 2
        elif num_restarts == 2:
            labels = np.zeros(n, dtype=int)
        elif num_restarts == 3:
            labels = np.ones(n, dtype=int)
        else:
            labels = rng.integers(0, 2, size=n)
        
        remaining_time = max(1.0, (global_deadline - 30) - time.time())
        ls_time_limit = max(0.05, remaining_time / (num_restarts + 10))
        
        refined_labels, cut = _local_search(labels, adjacency, ls_time_limit, rng)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = refined_labels.copy()
        
        num_restarts += 1
    
    # Phase 2: Simulated annealing
    if best_labels is not None and time.time() < global_deadline - 5:
        sa_time_limit = max(0.1, global_deadline - 5 - time.time())
        sa_labels, sa_cut = _simulated_annealing(best_labels, adjacency, sa_time_limit, rng)
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels.copy()
    
    if best_labels is None:
        best_labels = np.zeros(n, dtype=int)
    
    return best_labels
# EVOLVE-BLOCK-END
