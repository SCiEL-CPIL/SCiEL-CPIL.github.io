# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gain(adjacency, labels, v):
    """Compute the change in cut size if we flip vertex v."""
    same_count = adjacency[v][labels == labels[v]].sum()
    opp_count = adjacency[v][labels != labels[v]].sum()
    return int(same_count - opp_count)


def _all_flip_gains(adjacency, labels):
    """Compute flip gains for all vertices at once."""
    n = len(labels)
    # For each vertex v, gain = (neighbors with same label) - (neighbors with opposite label)
    # same_count[v] = sum of adjacency[v] where labels == labels[v]
    # opp_count[v] = sum of adjacency[v] where labels != labels[v]
    
    # Vectorized: for each vertex, count neighbors with same label
    # labels[:, None] != labels[None, :] gives the diff matrix
    diff = (labels[:, None] != labels[None, :])
    
    # opp_counts[v] = number of neighbors of v with opposite label
    opp_counts = (adjacency & diff).sum(axis=1)
    
    # total_degrees[v] = total number of neighbors of v
    total_degrees = adjacency.sum(axis=1)
    
    # same_counts[v] = total_degrees[v] - opp_counts[v]
    same_counts = total_degrees - opp_counts
    
    # gain = same_count - opp_count
    gains = same_counts - opp_counts
    return gains


def _local_search(adjacency, labels, deadline):
    """Iterative local search: flip vertices if it improves cut size.
    Uses vectorized gain computation and shuffled order for efficiency."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    improved = True
    while improved:
        improved = False
        if time.time() > deadline:
            break
        
        # Shuffle the order to avoid getting stuck in deterministic cycles
        order = np.random.permutation(n)
        
        for v in order:
            if time.time() > deadline:
                break
            gain = _flip_gain(adjacency, labels, v)
            if gain > 0:
                labels[v] = 1 - labels[v]
                current_cut += gain
                improved = True
    
    return labels, current_cut


def _random_labels(n, seed):
    """Generate random 0/1 labels."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 2, size=n)


def _spectral_partition(adjacency, n):
    """Spectral partition using the Fiedler vector."""
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import eigsh
        
        # Build Laplacian L = D - A
        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get smallest eigenvalue (should be 0) and second smallest (Fiedler)
        # Use eigsh for the smallest eigenvalues
        try:
            vals, vecs = eigsh(L, k=2, which='SA')
            # Sort by eigenvalue
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        except Exception:
            # Fallback to dense
            vals, vecs = np.linalg.eigh(L)
            idx = np.argsort(vals)
            fiedler_vec = vecs[:, idx[1]]
        
        # Partition: vertices with positive Fiedler component -> 1, negative -> 0
        labels = (fiedler_vec > 0).astype(np.int8)
        
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            # If all same, use median split
            median = np.median(fiedler_vec)
            labels = (fiedler_vec > median).astype(np.int8)
            if labels.sum() == 0 or labels.sum() == n:
                labels = np.arange(n) % 2
        
        return labels
    except Exception:
        return np.arange(n) % 2


def _goemans_williamson_cut(adjacency, n, rng=None):
    """Goemans-Williamson style: use eigenvectors of Laplacian to generate
    random hyperplane cuts. This leverages the geometric separation of vertices."""
    if rng is None:
        rng = np.random.RandomState(42)
    
    try:
        A = adjacency.astype(np.float64)
        D = np.diag(A.sum(axis=1))
        L = D - A
        
        # Get several eigenvectors corresponding to smallest eigenvalues
        # These capture the low-frequency structure of the graph
        k = min(10, n - 1)
        
        try:
            from scipy.sparse import csr_matrix
            from scipy.sparse.linalg import eigsh
            L_sparse = csr_matrix(L)
            vals, vecs = eigsh(L_sparse, k=k, which='SA')
        except Exception:
            vals, vecs = np.linalg.eigh(L)
            vecs = vecs[:, :k]
        
        # For each random direction in the eigenvector space, project and cut
        best_labels = None
        best_cut = -1
        
        num_hyperplanes = 20
        for _ in range(num_hyperplanes):
            # Random unit vector in the space spanned by eigenvectors
            coeffs = rng.randn(k)
            coeffs /= (np.linalg.norm(coeffs) + 1e-10)
            
            # Project vertices onto this direction
            projection = vecs @ coeffs
            
            # Cut based on sign
            labels = (projection > 0).astype(np.int8)
            
            # Ensure both partitions are non-empty
            if labels.sum() == 0 or labels.sum() == n:
                continue
            
            cut = _cut_size(adjacency, labels)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        if best_labels is not None:
            return best_labels
        else:
            return np.arange(n) % 2
    
    except Exception:
        return np.arange(n) % 2


def _walksat(adjacency, labels, deadline, max_iters=20000):
    """WalkSAT-style local search that can escape local optima."""
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    
    # Precompute total degrees
    total_degrees = adjacency.sum(axis=1)
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
            
        # Calculate gains for all vertices
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        best_gain = np.max(gains)
        
        if best_gain > 0:
            # Flip the vertex with the best gain
            v = np.argmax(gains)
            labels[v] = 1 - labels[v]
            cut += best_gain
        else:
            # Local optimum. Pick a vertex involved in an uncut edge.
            uncut_mask = adjacency & (~diff)
            uncut_edges = np.argwhere(uncut_mask)
            
            if len(uncut_edges) == 0:
                break
                
            # Pick a random uncut edge
            edge_idx = np.random.randint(0, len(uncut_edges))
            u, v = uncut_edges[edge_idx]
            
            # Flip one of them randomly
            if np.random.rand() < 0.5:
                flip_v = u
            else:
                flip_v = v
                
            # Update cut
            cut += gains[flip_v]
            labels[flip_v] = 1 - labels[flip_v]
            
    return labels, cut


def _simulated_annealing(adjacency, labels, deadline, max_iters=30000):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    cut = _cut_size(adjacency, labels)
    best_labels = labels.copy()
    best_cut = cut
    
    total_degrees = adjacency.sum(axis=1)
    current_temp = 5.0
    cooling_rate = 0.9995
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        # Calculate gains for all vertices
        diff = (labels[:, None] != labels[None, :])
        opp_counts = (adjacency & diff).sum(axis=1)
        gains = total_degrees - 2 * opp_counts
        
        # Choose a random vertex to flip
        v = np.random.randint(n)
        gain = gains[v]
        
        if gain > 0 or np.random.rand() < np.exp(gain / max(current_temp, 1e-8)):
            labels[v] = 1 - labels[v]
            cut += gain
            
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
        
        current_temp *= cooling_rate
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    adjacency = np.asarray(adjacency, dtype=bool)
    
    start_time = time.time()
    deadline = start_time + 145  # Safety margin: 145s out of 150s limit
    
    best_labels = None
    best_cut = -1
    
    def _update_best(labels, cut):
        nonlocal best_labels, best_cut
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Strategy 1: Goemans-Williamson spectral initialization
    if n >= 2:
        gw_labels = _goemans_williamson_cut(adjacency, n)
        gw_labels, gw_cut = _local_search(adjacency, gw_labels, deadline)
        _update_best(gw_labels, gw_cut)
    
    # Strategy 2: Standard spectral initialization
    if n >= 2:
        spec_labels = _spectral_partition(adjacency, n)
        spec_labels, spec_cut = _local_search(adjacency, spec_labels, deadline)
        _update_best(spec_labels, spec_cut)
    
    # Strategy 3: Alternating start
    alt_labels = np.arange(n) % 2
    alt_labels, alt_cut = _local_search(adjacency, alt_labels, deadline)
    _update_best(alt_labels, alt_cut)
    
    # Strategy 4: Multiple random restarts with WalkSAT
    base_seed = 42
    max_restarts = 15
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        seed = base_seed + restart
        labels = _random_labels(n, seed)
        
        # Run WalkSAT first
        labels, cut = _walksat(adjacency, labels, deadline)
        _update_best(labels, cut)
        
        # If we have time, run simulated annealing
        if time.time() < deadline - 5:
            sa_labels, sa_cut = _simulated_annealing(adjacency, labels, deadline)
            _update_best(sa_labels, sa_cut)
    
    # Strategy 5: Perturbation-based restarts from best
    if best_labels is not None and time.time() < deadline - 5:
        for _ in range(10):
            if time.time() > deadline:
                break
            # Perturb the best solution
            perturbed = best_labels.copy()
            # Flip 5% of random vertices
            num_flips = max(1, n // 20)
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            
            # Run local search on perturbed solution
            perturbed, perturbed_cut = _local_search(adjacency, perturbed, deadline)
            _update_best(perturbed, perturbed_cut)
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(np.int8)
# EVOLVE-BLOCK-END
