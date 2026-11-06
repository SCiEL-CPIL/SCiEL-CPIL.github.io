"""
Strategy: Multi-Start Spectral Initialization with Vectorized Simulated Annealing.
Instead of relying on a single Fiedler vector initialization followed by greedy local search (which gets trapped in local optima),
this approach uses multiple diverse spectral initializations (Fiedler vector, top eigenvector, random) and runs a
vectorized Simulated Annealing (SA) process that accepts temporary quality drops to escape local optima.
The SA is fully vectorized: at each step, we compute gains for ALL vertices simultaneously, then select a move
(stochastic or greedy) based on the temperature. This allows the search to explore the solution space more broadly
while still converging to high-quality cuts. Multiple restarts with different seeds/initializations ensure robustness.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def partition(adjacency):
    """
    Assign each vertex a 0/1 label to maximize the number of cut edges.
    Uses multi-start spectral initialization followed by vectorized simulated annealing.
    """
    adj = np.asarray(adjacency, dtype=np.float64)
    n = len(adj)
    
    if n <= 1:
        return np.zeros(n, dtype=np.int8)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Check if A is all zeros
    if not np.any(adj):
        return np.zeros(n, dtype=np.int8)
    
    # Precompute degrees
    degrees = np.sum(adj, axis=1)
    
    # Build Laplacian for spectral methods
    D = np.diag(degrees)
    L = D - adj
    
    # Generate multiple diverse initializations
    initializations = []
    
    # 1. Fiedler vector initialization
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        fiedler_vec = eigenvectors[:, 1]
        labels_fiedler = (fiedler_vec > 0).astype(np.int8)
        initializations.append(labels_fiedler.copy())
    except Exception:
        pass
    
    # 2. Top eigenvector of adjacency matrix
    try:
        eigenvalues_a, eigenvectors_a = np.linalg.eigh(adj)
        idx_a = np.argsort(eigenvalues_a)[::-1]
        top_eigvec = eigenvectors_a[:, idx_a[0]]
        labels_top = (top_eigvec > 0).astype(np.int8)
        initializations.append(labels_top.copy())
    except Exception:
        pass
    
    # 3. Random initialization (a few variants)
    rng = np.random.RandomState(42)
    for _ in range(3):
        labels_rand = rng.randint(0, 2, n).astype(np.int8)
        initializations.append(labels_rand.copy())
    
    # 4. Balanced random initialization
    labels_balanced = np.zeros(n, dtype=np.int8)
    half = n // 2
    labels_balanced[:half] = 1
    perm = rng.permutation(n)
    labels_balanced = labels_balanced[perm]
    initializations.append(labels_balanced.copy())
    
    def compute_cut_size(labels):
        """Compute the number of cut edges for a given labeling."""
        # Cut edges: edges (i,j) where labels[i] != labels[j]
        # cut = sum_{i<j} adj[i,j] * (1 - labels[i]*labels[j]) for 0/1 labels
        # More efficiently: cut = sum_i sum_j adj[i,j] * (labels[i] != labels[j]) / 2
        diff_mask = (labels[:, None] != labels[None, :])
        return 0.5 * np.sum(adj * diff_mask)
    
    def vectorized_sa(labels, adj, degrees, start_time, deadline, max_iter=50000):
        """
        Vectorized Simulated Annealing.
        At each step, compute gains for all vertices, then select a move.
        """
        labels = labels.copy()
        current_cut = compute_cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        
        # SA parameters
        T0 = 10.0  # Initial temperature
        T_min = 0.01
        cooling = 0.9995  # Cooling rate
        
        T = T0
        rng = np.random.RandomState(int(start_time * 1000) % 10000)
        
        # Precompute for efficiency
        # We'll recompute gains each iteration (vectorized)
        
        for it in range(max_iter):
            if time.time() > deadline - 5.0:
                break
            
            # Compute gains for flipping each vertex
            # Gain for flipping vertex i = 2 * (number of neighbors on same side) - degree[i]
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            # Select a move: with some probability, pick the best move; otherwise, pick randomly
            # This balances exploitation and exploration
            if rng.random() < 0.3:
                # Exploitation: pick the vertex with max positive gain
                max_gain = np.max(gains)
                if max_gain <= 1e-9:
                    # No improving move, try random move
                    flip_idx = rng.randint(n)
                else:
                    flip_idx = np.argmax(gains)
            else:
                # Exploration: pick a random vertex
                flip_idx = rng.randint(n)
            
            gain = gains[flip_idx]
            
            # Accept or reject based on SA criterion
            if gain > 0:
                accept = True
            else:
                # Accept with probability exp(gain / T)
                if T > T_min:
                    prob = np.exp(gain / T)
                    accept = (rng.random() < prob)
                else:
                    accept = False
            
            if accept:
                labels[flip_idx] = 1 - labels[flip_idx]
                current_cut += gain
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            
            # Cool down
            T *= cooling
            if T < T_min:
                T = T_min
        
        return best_labels, best_cut
    
    # Run SA from each initialization, keeping the best result
    best_labels = None
    best_cut = -1
    
    for init_labels in initializations:
        if time.time() > deadline - 15.0:
            break
        
        sa_labels, sa_cut = vectorized_sa(
            init_labels, adj, degrees, start_time, deadline,
            max_iter=20000
        )
        
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_labels
    
    # Final polish: greedy local search on the best result
    if best_labels is not None and time.time() < deadline - 10.0:
        labels = best_labels.copy()
        improved = True
        while improved:
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            same_mask = (labels[:, None] == labels[None, :])
            same_counts = np.sum(adj * same_mask, axis=1)
            gains = 2 * same_counts - degrees
            
            max_gain = np.max(gains)
            if max_gain > 1e-9:
                flip_idx = np.argmax(gains)
                labels[flip_idx] = 1 - labels[flip_idx]
                improved = True
        best_labels = labels
    
    if best_labels is None:
        best_labels = np.zeros(n, dtype=np.int8)
    
    return best_labels
# EVOLVE-BLOCK-END
