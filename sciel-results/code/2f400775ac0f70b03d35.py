"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :])
    return int((adjacency & diff).sum() // 2)


def _flip_gains(adjacency, labels):
    """Compute flip gain for all vertices simultaneously.
    Positive gain means flipping improves the cut.
    """
    s = 2 * labels - 1  # +1 or -1
    gains = s * (adjacency @ s)
    return gains


def _local_search(adjacency, labels, max_time=None):
    """Local search with single-flip improvements."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > max_time:
            break
        
        # Compute all flip gains at once
        gains = _flip_gains(adjacency, labels)
        
        # Find vertices with positive gain
        positive = np.where(gains > 0)[0]
        
        if len(positive) == 0:
            break
        
        # Flip all vertices with positive gain
        for v in positive:
            if time.time() - start_time > max_time:
                return labels, current_cut
            labels[v] = 1 - labels[v]
            current_cut += int(gains[v]) // 2
            improved = True
    
    return labels, current_cut


def _spectral_partition(adjacency):
    """Initialize labels using the Fiedler vector (eigenvector of second smallest eigenvalue)."""
    n = len(adjacency)
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        L = np.diag(degrees) - adjacency
        vals, vecs = eigsh(L, k=2, which='SM')
        fiedler = vecs[:, 1]
        labels = (fiedler > 0).astype(int)
        if labels.sum() > n / 2:
            labels = 1 - labels
        return labels
    except Exception:
        try:
            degrees = np.array(adjacency.sum(axis=1)).flatten()
            L = np.diag(degrees) - adjacency
            vals, vecs = np.linalg.eigh(L)
            fiedler = vecs[:, 1]
            labels = (fiedler > 0).astype(int)
            if labels.sum() > n / 2:
                labels = 1 - labels
            return labels
        except Exception:
            return np.random.randint(0, 2, size=n)


def _goemans_williamson_init(adjacency, n_hyperplanes=5):
    """Generate initial partitions using Goemans-Williamson style random hyperplane cuts
    from the eigenvectors of the graph Laplacian."""
    n = len(adjacency)
    
    try:
        from scipy.sparse.linalg import eigsh
        from scipy.sparse import csr_matrix
        
        # Compute Laplacian
        degrees = np.array(adjacency.sum(axis=1)).flatten()
        L = np.diag(degrees) - adjacency
        
        # Get several eigenvectors (not just the Fiedler vector)
        k = min(10, n - 1)
        vals, vecs = eigsh(L, k=k, which='SM')
        
        # Use the eigenvectors (skip the first one which is constant)
        eigenvectors = vecs[:, 1:]  # Shape: (n, k-1)
        
        # Normalize columns
        norms = np.linalg.norm(eigenvectors, axis=0, keepdims=True)
        norms[norms == 0] = 1
        eigenvectors = eigenvectors / norms
        
        best_labels = None
        best_cut = -1
        
        for _ in range(n_hyperplanes):
            # Generate a random hyperplane in the eigenspace
            random_vector = np.random.randn(eigenvectors.shape[1])
            projections = eigenvectors @ random_vector
            labels = (projections > 0).astype(int)
            
            # Ensure balanced partition
            if labels.sum() > n / 2:
                labels = 1 - labels
            
            # Quick local search to evaluate quality
            refined_labels, cut = _local_search(adjacency, labels, max_time=2.0)
            if cut > best_cut:
                best_cut = cut
                best_labels = refined_labels.copy()
        
        if best_labels is not None:
            return best_labels
    except Exception:
        pass
    
    # Fallback to spectral
    return _spectral_partition(adjacency)


def _walksat(adjacency, labels, max_time=None):
    """WalkSAT-style local search for Max-Cut."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    # Precompute neighbor lists
    neighbors = [np.where(adjacency[i])[0] for i in range(n)]
    
    while time.time() - start_time < max_time:
        # Compute gains
        gains = _flip_gains(adjacency, labels)
        
        # Find best flip
        best_gain = np.max(gains)
        if best_gain <= 0:
            # No improving flip; do a random walk
            # Pick a random vertex and flip it
            v = np.random.randint(n)
            labels[v] = 1 - labels[v]
            current_cut += int(gains[v]) // 2
            continue
        
        # Flip the vertex with the best gain
        v = np.argmax(gains)
        labels[v] = 1 - labels[v]
        current_cut += int(gains[v]) // 2
    
    return labels, current_cut


def _two_flip_local_search(adjacency, labels, max_time=None):
    """Local search that considers pairs of flips."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        if time.time() - start_time > max_time:
            break
        
        # First try single flips
        gains = _flip_gains(adjacency, labels)
        positive = np.where(gains > 0)[0]
        
        if len(positive) > 0:
            for v in positive:
                labels[v] = 1 - labels[v]
                current_cut += int(gains[v]) // 2
                improved = True
            continue
        
        # Try two-flip moves
        # Sample random pairs
        for _ in range(100):
            if time.time() - start_time > max_time:
                break
            i, j = np.random.choice(n, size=2, replace=False)
            if labels[i] == labels[j]:
                continue
            
            # Compute gain of flipping both i and j
            # This is complex; approximate by checking if it helps
            old_labels = labels.copy()
            labels[i] = 1 - labels[i]
            labels[j] = 1 - labels[j]
            new_cut = _cut_size(adjacency, labels)
            
            if new_cut > current_cut:
                current_cut = new_cut
                improved = True
            else:
                labels = old_labels
    
    return labels, current_cut


def _simulated_annealing(adjacency, labels, max_time=None, initial_temp=10.0, cooling=0.95):
    """Simulated annealing for Max-Cut."""
    n = len(adjacency)
    labels = labels.copy()
    current_cut = _cut_size(adjacency, labels)
    
    if max_time is None:
        max_time = 145.0
    
    start_time = time.time()
    temp = initial_temp
    best_labels = labels.copy()
    best_cut = current_cut
    
    while time.time() - start_time < max_time and temp > 0.01:
        # Pick a random vertex to flip
        v = np.random.randint(n)
        
        # Compute gain
        gains = _flip_gains(adjacency, labels)
        gain = gains[v]
        
        # Accept if improvement, or with probability exp(-delta/temp)
        if gain > 0:
            labels[v] = 1 - labels[v]
            current_cut += int(gain) // 2
        else:
            if np.random.random() < np.exp(-abs(int(gain)) / temp):
                labels[v] = 1 - labels[v]
                current_cut += int(gain) // 2
        
        # Update best
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
        
        # Cool down
        temp *= cooling
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    best_labels = None
    best_cut = -1
    
    def _try_labels(labels, time_limit):
        nonlocal best_labels, best_cut
        labels, cut = _local_search(adjacency, labels, max_time=time_limit)
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        return cut
    
    # Phase 1: Goemans-Williamson style initialization with multiple hyperplanes
    remaining = deadline - time.time()
    if remaining > 30:
        gw_labels = _goemans_williamson_init(adjacency, n_hyperplanes=8)
        _try_labels(gw_labels, min(remaining - 10, 20))
    
    # Phase 2: Spectral initialization
    remaining = deadline - time.time()
    if remaining > 20:
        spectral_labels = _spectral_partition(adjacency)
        _try_labels(spectral_labels, min(remaining - 10, 15))
    
    # Phase 3: Multiple random restarts with different strategies
    num_restarts = 30
    for restart in range(num_restarts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5:
            break
        
        # Choose initialization strategy
        if best_labels is not None and restart % 4 == 0:
            # Perturbed best solution
            current_labels = best_labels.copy()
            num_flips = max(1, int(n * 0.1))
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            current_labels[flip_indices] = 1 - current_labels[flip_indices]
        elif restart % 4 == 1:
            # Random start
            current_labels = np.random.randint(0, 2, size=n)
        elif restart % 4 == 2:
            # Alternating start
            current_labels = np.arange(n) % 2
            if np.random.random() < 0.5:
                current_labels = 1 - current_labels
        else:
            # Spectral-based start
            current_labels = _spectral_partition(adjacency)
        
        # Time budget for this restart
        time_per_restart = max(1, (remaining - 3) / max(1, num_restarts - restart))
        
        # Choose search method
        if restart % 3 == 0:
            labels, cut = _local_search(adjacency, current_labels, max_time=time_per_restart)
        elif restart % 3 == 1:
            labels, cut = _walksat(adjacency, current_labels, max_time=time_per_restart)
        else:
            labels, cut = _simulated_annealing(adjacency, current_labels, max_time=time_per_restart, initial_temp=5.0)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # Phase 4: Final polish with two-flip local search
    remaining = deadline - time.time()
    if remaining > 5 and best_labels is not None:
        final_labels, final_cut = _two_flip_local_search(adjacency, best_labels, max_time=min(remaining - 2, 10))
        if final_cut > best_cut:
            best_cut = final_cut
            best_labels = final_labels
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels.astype(int)
# EVOLVE-BLOCK-END
