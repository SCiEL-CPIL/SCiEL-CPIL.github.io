"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # For each edge (i,j), it crosses if labels[i] != labels[j]
    # Since adjacency is symmetric, we count each edge twice, so divide by 2
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _cut_size_fast(adjacency_int, labels):
    """Fast cut size computation using precomputed int adjacency."""
    # labels: 0/1 int array
    # An edge (i,j) crosses if labels[i] != labels[j]
    # Use: cut = sum(A[i,j] * (labels[i] != labels[j])) / 2
    # Efficient: (labels @ A @ (1 - labels)) + (labels @ A @ labels) ... no
    # Better: diff = labels[:,None] != labels[None,:]
    # But we can use: for each i, sum over j where A[i,j]=1 and labels[j]!=labels[i]
    # Vectorized: for each row i, count neighbors with different label
    # diff_matrix = labels[:,None] != labels[None,:]  # O(n^2) memory
    # Instead: cut = 0.5 * sum_i sum_j A[i,j] * (1 - 2*labels[i]*(1-labels[j]) - 2*(1-labels[i])*labels[j])
    # Simpler: let s = labels (0/1). Edge (i,j) crosses iff s[i]+s[j] == 1
    # cut = 0.5 * sum_{i,j} A[i,j] * (s[i] + s[j] - 2*s[i]*s[j])
    #     = 0.5 * (s @ A @ ones + ones @ A @ s - 2 * s @ A @ s)
    # Since A is symmetric: s @ A @ ones = s @ (A @ ones) = s @ degrees
    # And ones @ A @ s = (A @ s) @ ones = sum(A @ s)
    # So cut = 0.5 * (s @ d + sum(A @ s) - 2 * s @ A @ s)
    # where d = A @ ones = degrees
    s = labels.astype(np.float64)
    d = adjacency_int.sum(axis=1).astype(np.float64)
    As = adjacency_int @ s
    cut = 0.5 * (s @ d + As.sum() - 2.0 * (s @ As))
    return int(round(cut))


def _spectral_partition(adjacency):
    """Use spectral method (Fiedler vector) to get initial partition."""
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    # Degree matrix
    d = A.sum(axis=1)
    # Laplacian L = D - A
    L = np.diag(d) - A
    
    # Compute eigenvalues and eigenvectors
    # We want the second smallest eigenvalue's eigenvector (Fiedler vector)
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        # Fallback to random
        return np.random.randint(0, 2, size=n)
    
    # Sort by eigenvalues
    idx = np.argsort(eigvals)
    # Fiedler vector is the eigenvector corresponding to second smallest eigenvalue
    fiedler = eigvecs[:, idx[1]]
    
    # Partition based on sign of Fiedler vector
    labels = (fiedler >= 0).astype(np.int32)
    
    # Ensure both partitions are non-empty
    if labels.sum() == 0 or labels.sum() == n:
        labels = np.random.randint(0, 2, size=n)
    
    return labels


def _gwg_spectral_partition(adjacency, n_hyperplanes=50):
    """
    Generate high-quality initial partitions using Goemans-Williamson style
    random hyperplane cuts from Laplacian eigenvectors.
    
    The idea: embed vertices in R^k using the first k eigenvectors of the
    graph Laplacian, then generate random hyperplanes to cut this embedding.
    This leverages the geometric structure of the graph.
    """
    n = len(adjacency)
    A = adjacency.astype(np.float64)
    d = A.sum(axis=1)
    L = np.diag(d) - A
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except np.linalg.LinAlgError:
        return np.random.randint(0, 2, size=n)
    
    # Sort by eigenvalues
    idx = np.argsort(eigvals)
    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]
    
    # Use the first few eigenvectors (skip the trivial constant vector)
    # The Fiedler vector is eigvecs_sorted[:, 1]
    # Use eigenvectors corresponding to small eigenvalues for better geometric separation
    k = min(10, n - 1)  # Use up to 10 eigenvectors (excluding the constant one)
    embedding = eigvecs_sorted[:, 1:k+1]  # Shape: (n, k)
    
    # Normalize rows to unit length for better geometric interpretation
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embedding_normalized = embedding / norms
    
    best_labels = None
    best_cut = -1
    
    # Generate multiple random hyperplanes
    for _ in range(n_hyperplanes):
        # Generate a random hyperplane normal vector in R^k
        normal = np.random.randn(k)
        normal /= (np.linalg.norm(normal) + 1e-10)
        
        # Project vertices onto the hyperplane normal
        projections = embedding_normalized @ normal
        
        # Partition based on sign of projection
        labels = (projections >= 0).astype(np.int32)
        
        # Ensure both partitions are non-empty
        if labels.sum() == 0 or labels.sum() == n:
            continue
        
        cut = _cut_size(adjacency, labels)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels
    
    # If no valid partition was found, fall back to Fiedler vector
    if best_labels is None:
        fiedler = eigvecs_sorted[:, 1]
        best_labels = (fiedler >= 0).astype(np.int32)
        if best_labels.sum() == 0 or best_labels.sum() == n:
            best_labels = np.random.randint(0, 2, size=n)
    
    return best_labels


def _random_labels(n):
    """Generate random initial labels."""
    return np.random.randint(0, 2, size=n)


def _local_search_fast(adjacency, labels, deadline, max_rounds=100):
    """
    Vectorized iterative local search using gain calculations.
    For vertex v with label lv, the gain from flipping is:
      gain = 2 * (num_opp_neighbors) - degree[v]
    where num_opp_neighbors = count of neighbors with label != lv
    Flip if gain > 0.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int8)
    
    # Precompute degrees
    degrees = A.sum(axis=1).astype(np.int32)
    
    # Precompute neighbor lists for O(1) access
    neighbors = [np.where(A[i] == 1)[0] for i in range(n)]
    
    # Track which vertices have been flipped to avoid oscillation
    # Use a simple approach: in each pass, compute all gains and flip all positive ones
    
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        
        # Compute gain for each vertex
        # gain[v] = 2 * opp_count[v] - degrees[v]
        # opp_count[v] = number of neighbors of v with label != labels[v]
        
        improved = False
        # Process vertices in random order to break ties
        order = np.random.permutation(n)
        
        for v in order:
            if time.time() > deadline:
                break
            
            lv = labels[v]
            deg = degrees[v]
            if deg == 0:
                continue
            
            nbs = neighbors[v]
            # Count neighbors with opposite label
            opp_count = (labels[nbs] != lv).sum()
            same_count = deg - opp_count
            
            # Flipping v: same->crossing (+same_count), crossing->same (-opp_count)
            # Net change in cut = same_count - opp_count
            gain = same_count - opp_count
            
            if gain > 0:
                labels[v] = 1 - lv
                improved = True
        
        if not improved:
            break
    
    return labels


def _simulated_annealing_fast(adjacency, labels, deadline, max_iterations=10000, 
                                T_start=8.0, T_end=0.1, adaptive=True):
    """
    Simulated annealing with adaptive cooling schedule.
    Uses vectorized gain calculation for speed.
    """
    n = len(labels)
    labels = labels.copy().astype(np.int32)
    A = adjacency.astype(np.int8)
    
    current_cut = _cut_size_fast(A, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    # Precompute neighbor indices and degrees
    neighbors = [np.where(A[i] == 1)[0] for i in range(n)]
    degrees = A.sum(axis=1).astype(np.int32)
    
    T = T_start
    non_improving_count = 0
    patience = 500  # Reduce temperature faster if no improvement
    
    for it in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Adaptive temperature update
        if adaptive:
            if non_improving_count >= patience:
                T = max(T * 0.9, T_end)  # Cool faster
                non_improving_count = 0
            else:
                progress = it / max_iterations
                T = T_start * (T_end / T_start) ** progress
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        deg = degrees[v]
        
        if deg == 0:
            continue
        
        nbs = neighbors[v]
        same_count = (labels[nbs] == old_label).sum()
        opp_count = deg - same_count
        delta = same_count - opp_count
        
        # Accept if improvement or with probability exp(delta/T)
        if delta > 0:
            accept = True
        else:
            if T > T_end * 0.5:
                accept = np.random.random() < np.exp(delta / max(T, 1e-10))
            else:
                accept = False
        
        if accept:
            labels[v] = 1 - old_label
            current_cut += delta
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
                non_improving_count = 0
            else:
                non_improving_count += 1
        else:
            non_improving_count += 1
    
    return best_labels, best_cut


def partition(adjacency):
    """
    Main partition function using multi-start strategy with:
    - Spectral initialization
    - Vectorized local search
    - Multi-start simulated annealing with adaptive cooling
    - Final polish pass
    """
    n = len(adjacency)
    
    # Set a deadline with safety margin (150s limit, use 125s to be safe)
    start_time = time.time()
    deadline = start_time + 125.0
    
    # Precompute int adjacency for fast operations
    A_int = adjacency.astype(np.int8)
    
    # Generate diverse initial candidates
    candidates = []
    
    # 1. Standard spectral partition (Fiedler vector)
    spec_labels = _spectral_partition(adjacency)
    candidates.append(spec_labels)
    
    # 2. Goemans-Williamson style spectral partition with random hyperplanes
    gw_labels = _gwg_spectral_partition(adjacency, n_hyperplanes=30)
    candidates.append(gw_labels)
    
    # 3. Random partitions (a few)
    for _ in range(3):
        rand_labels = _random_labels(n)
        candidates.append(rand_labels)
    
    # 4. Alternating partition
    alt_labels = np.arange(n) % 2
    candidates.append(alt_labels)
    
    # Phase 1: Local search on each candidate
    best_labels = None
    best_cut = -1
    
    for cand in candidates:
        if time.time() > deadline:
            break
        
        # Run fast local search on this candidate
        ls_labels = _local_search_fast(adjacency, cand, deadline)
        ls_cut = _cut_size_fast(A_int, ls_labels)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
    
    # Phase 2: Multi-start simulated annealing
    if time.time() < deadline - 20.0 and best_labels is not None:
        # Run multiple SA restarts from different starting points
        sa_starts = [best_labels.copy()]
        
        # Add a few perturbed versions of the best solution
        for _ in range(2):
            perturbed = best_labels.copy()
            # Flip 5-10% of vertices randomly
            n_flips = max(1, int(0.075 * n))
            flip_indices = np.random.choice(n, size=n_flips, replace=False)
            perturbed[flip_indices] = 1 - perturbed[flip_indices]
            sa_starts.append(perturbed)
        
        # Also try SA from spectral solution directly
        sa_starts.append(spec_labels.copy())
        
        for start_labels in sa_starts:
            if time.time() > deadline - 10.0:
                break
            
            # Run adaptive simulated annealing
            sa_labels, sa_cut = _simulated_annealing_fast(
                adjacency, start_labels, deadline, 
                max_iterations=8000, T_start=8.0, T_end=0.1, adaptive=True
            )
            
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels
    
    # Phase 3: Final high-precision polish pass
    if time.time() < deadline - 5.0 and best_labels is not None:
        # Run additional local search rounds with tighter convergence
        final_labels = _local_search_fast(adjacency, best_labels, deadline, max_rounds=50)
        final_cut = _cut_size_fast(A_int, final_labels)
        
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
        
        # One more SA polish from the best result if time permits
        if time.time() < deadline - 2.0:
            polish_labels, polish_cut = _simulated_annealing_fast(
                adjacency, best_labels, deadline,
                max_iterations=3000, T_start=3.0, T_end=0.05, adaptive=True
            )
            if polish_cut > best_cut:
                best_cut = polish_cut
                best_labels = polish_labels
                
                # Final local search after polish
                if time.time() < deadline - 0.5:
                    final2 = _local_search_fast(adjacency, best_labels, deadline, max_rounds=30)
                    final2_cut = _cut_size_fast(A_int, final2)
                    if final2_cut > best_cut:
                        best_cut = final2_cut
                        best_labels = final2
    
    # Ensure we return a valid label array
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
