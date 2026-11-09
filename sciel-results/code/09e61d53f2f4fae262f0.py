"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """Initialize using the sign of the Fiedler vector (2nd smallest eigenvector of Laplacian)."""
    n = adjacency.shape[0]
    # Compute Laplacian L = D - A
    # Since adjacency is boolean, degrees are sums of rows
    degrees = adjacency.sum(axis=1)
    # L = D - A
    # We need the 2nd smallest eigenvalue/eigenvector.
    # For sparse or medium graphs, scipy.sparse.linalg.eigsh is efficient.
    # But standard numpy is fine for n=100.
    L = np.diag(degrees) - adjacency.astype(float)
    
    # Use np.linalg.eigh for symmetric matrix
    try:
        vals, vecs = np.linalg.eigh(L)
        # The Fiedler vector is the eigenvector corresponding to the second smallest eigenvalue
        # Eigenvalues are sorted in ascending order by eigh
        fiedler = vecs[:, 1]
        # Assign 0 where fiedler < 0, 1 otherwise
        labels = (fiedler >= 0).astype(int)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n)

def _simulated_annealing(adjacency, labels, start_temp, end_temp, max_iters, deadline):
    """
    Vectorized Simulated Annealing for Max Cut.
    Uses numpy to calculate energy differences for all vertices at once 
    (or in large batches) to speed up the inner loop.
    """
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute adjacency as float for matrix ops
    adj_float = adjacency.astype(float)
    
    # To speed up SA, we can pre-compute the "gain" of flipping each node.
    # Gain(v) = (neighbors with opposite label) - (neighbors with same label)
    # If we flip v, the cut changes by Gain(v).
    # Actually, if we flip v, edges (v, u) where label[u] != label[v] become non-cut,
    # and edges (v, u) where label[u] == label[v] become cut.
    # So Delta = (# same) - (# diff).
    # We want to maximize cut, so we want Delta > 0.
    
    # We will update gains incrementally to avoid O(N^2) per step.
    # However, for N=100, O(N^2) per step is 10,000 ops. 
    # 100,000 steps * 10,000 ops = 10^9 ops. This might be too slow in pure Python loops.
    # Vectorized approach:
    # Let's try a hybrid: Calculate all gains at start of each "batch" or just use vectorized ops for the whole matrix if possible?
    # No, labels change.
    
    # Alternative: Use the fact that N is small (100).
    # A single flip changes gains of neighbors.
    # Let's stick to a simple loop but optimized with local variables.
    
    start_time = time.time()
    
    # Temperature schedule: Geometric cooling
    if max_iters <= 0:
        return best_labels, best_cut
        
    alpha = (end_temp / start_temp) ** (1.0 / max_iters)
    
    # For speed, we will try to flip multiple nodes? No, standard SA is sequential.
    # But we can parallelize the "decision" part? No, decisions depend on current state.
    
    # Let's try to optimize the gain calculation.
    # current_gain[v] = sum(adj[v, u] * (1 if labels[u] != labels[v] else -1))
    # Wait, if labels[u] != labels[v], edge is cut. Flipping v makes it NOT cut. Loss = 1.
    # If labels[u] == labels[v], edge is NOT cut. Flipping v makes it cut. Gain = 1.
    # So Delta(v) = (# same neighbors) - (# diff neighbors)
    
    # Initialize gains
    # same = (adjacency & (labels == labels[:, None])).sum(axis=1)
    # diff = (adjacency & (labels != labels[:, None])).sum(axis=1)
    # This is O(N^2).
    
    # Let's just do the standard loop, it's N=100.
    # 100 nodes. 100k iterations. 10^7 node-visits.
    # In Python, 10^7 simple ops is ~1s. Should be fine.
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        T = start_temp * (alpha ** i)
        
        # Pick a random vertex
        v = np.random.randint(0, n)
        
        # Calculate gain of flipping v
        # neighbors = adjacency[v]
        # same_count = np.sum(neighbors & (labels == labels[v]))
        # diff_count = np.sum(neighbors & (labels != labels[v]))
        # delta = same_count - diff_count
        
        # Vectorized for single v:
        lv = labels[v]
        # neighbors with same label
        same = np.sum(adjacency[v] & (labels == lv))
        # neighbors with diff label
        diff = np.sum(adjacency[v] & (labels != lv))
        delta = same - diff
        
        # Acceptance probability
        if delta > 0:
            accept = True
        else:
            # P(accept) = exp(-delta / T)
            # delta is negative here, so -delta is positive
            # If T is very small, exp is 0
            if T > 0:
                prob = np.exp(-delta / T)
                accept = np.random.rand() < prob
            else:
                accept = False
        
        if accept:
            labels[v] = 1 - lv
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function.
    1. Spectral Init (Fiedler vector)
    2. Simulated Annealing to polish and escape local optima
    3. If time remains, try a few random restarts with SA
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + SA
    labels = _spectral_init(adjacency)
    # Run SA from this good start
    # Start temp high enough to allow exploration, end temp low for refinement
    # For N=100, typical cut size is around N^2/4 * density. 
    # Max possible cut is E.
    # Let's estimate max gain.
    max_deg = adjacency.sum(axis=1).max()
    # A reasonable start temp is related to the max possible delta
    start_temp = max_deg * 1.5
    end_temp = 0.01
    
    # How many iterations?
    # Let's try to use 80% of the time for this.
    est_time_per_iter = 0.0001 # Rough guess for N=100 in Python
    max_iters = int((deadline - time.time() - 10.0) / est_time_per_iter)
    max_iters = max(1000, min(max_iters, 500000)) # Cap it
    
    labels, cut = _simulated_annealing(adjacency, labels, start_temp, end_temp, max_iters, deadline - 10.0)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels
    
    # Phase 2: Random Restarts if time allows
    # Spectral init is usually very strong, but random restarts might find a better basin.
    remaining = deadline - time.time() - 5.0
    if remaining > 5.0:
        # Do a few quick random restarts
        num_restarts = 5
        time_per_restart = remaining / num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline - 5.0:
                break
            labels = np.random.randint(0, 2, size=n)
            # Shorter SA for restarts
            max_iters_r = int(time_per_restart / est_time_per_iter)
            labels, cut = _simulated_annealing(adjacency, labels, start_temp, end_temp, max_iters_r, deadline - 5.0)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels
    
    if best_labels is None:
        best_labels = _spectral_init(adjacency)
        
    return best_labels
# EVOLVE-BLOCK-END
