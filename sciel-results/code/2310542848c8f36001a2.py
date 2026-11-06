"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Maximum Cut via spectral initialization + simulated annealing with multiple restarts.
    Uses the Fiedler vector (second eigenvector of the graph Laplacian) for a high-quality
    starting partition, then refines with simulated annealing and greedy local search.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=int)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    def cut_size(labels):
        """Compute the number of edges crossing the partition."""
        # An edge (i,j) crosses if labels[i] != labels[j]
        # Using: sum over all edges of adj[i,j] * (1 if labels[i]!=labels[j] else 0)
        # Efficient: 0.5 * (sum(adj) - sum of adj[i,j] where labels[i]==labels[j])
        # Or: for 0/1 labels, crossing = 0.5 * sum_i sum_j adj[i,j] * |labels[i]-labels[j]|
        # Even simpler: let L be vector of labels (0/1). 
        # Number of edges within same side = sum over i<j where labels[i]==labels[j] of adj[i,j]
        # Total edges = 0.5 * sum(adj)
        # Cut = total edges - edges within same side
        # Edges within same side: for side 0: sum of adj[i,j] for i,j in side0, /2
        #                        for side 1: sum of adj[i,j] for i,j in side1, /2
        side0_mask = (labels == 0)
        side1_mask = (labels == 1)
        within_0 = adj[side0_mask][:, side0_mask].sum() / 2.0
        within_1 = adj[side1_mask][:, side1_mask].sum() / 2.0
        total_edges = adj.sum() / 2.0
        return total_edges - within_0 - within_1
    
    def greedy_local_search(labels):
        """Single pass of greedy local improvement."""
        labels = labels.copy()
        for v in range(n):
            same = adj[v, labels == labels[v]].sum()
            other = adj[v, labels != labels[v]].sum()
            if same > other:
                labels[v] = 1 - labels[v]
        return labels
    
    def simulated_annealing(labels, T_start=1.0, T_end=0.01, cooling=0.999):
        """Simulated annealing to refine the partition."""
        labels = labels.copy()
        current_cut = cut_size(labels)
        best_labels = labels.copy()
        best_cut = current_cut
        T = T_start
        
        while T > T_end and time.time() < deadline:
            # Pick a random vertex to flip
            v = np.random.randint(n)
            labels[v] = 1 - labels[v]
            new_cut = cut_size(labels)
            delta = new_cut - current_cut
            
            if delta > 0 or np.random.random() < np.exp(delta / T):
                current_cut = new_cut
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = labels.copy()
            else:
                labels[v] = 1 - labels[v]  # revert
            
            T *= cooling
        
        return best_labels, best_cut
    
    # Spectral initialization using the Fiedler vector
    # Compute the graph Laplacian L = D - A
    degrees = adj.sum(axis=1)
    L = np.diag(degrees) - adj
    
    # Get eigenvalues and eigenvectors
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        # The second smallest eigenvalue's eigenvector is the Fiedler vector
        # Sort eigenvalues (eigh already returns in ascending order)
        fiedler = eigenvectors[:, 1]
        # Create initial partition based on median split of Fiedler vector
        median_val = np.median(fiedler)
        labels_spectral = (fiedler > median_val).astype(np.int64)
    except Exception:
        # Fallback to alternating start
        labels_spectral = np.arange(n) % 2
    
    # Run greedy local search on spectral initialization
    labels_greedy = greedy_local_search(labels_spectral)
    best_labels = labels_greedy
    best_cut = cut_size(best_labels)
    
    # Try simulated annealing with multiple restarts
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        # Start from the best found so far or a perturbed version
        if num_restarts == 1:
            sa_start = labels_spectral.copy()
        else:
            # Perturb the best solution
            sa_start = best_labels.copy()
            # Flip a few random vertices
            num_flips = max(1, n // 20)
            flip_indices = np.random.choice(n, size=num_flips, replace=False)
            sa_start[flip_indices] = 1 - sa_start[flip_indices]
        
        sa_result, sa_cut = simulated_annealing(sa_start)
        if sa_cut > best_cut:
            best_cut = sa_cut
            best_labels = sa_result
        
        # Also try greedy local search on the SA result
        greedy_result = greedy_local_search(sa_result)
        greedy_cut = cut_size(greedy_result)
        if greedy_cut > best_cut:
            best_cut = greedy_cut
            best_labels = greedy_result
        
        # Check if we've hit the deadline
        if time.time() > deadline - 5.0:
            break
    
    return best_labels.astype(np.int64)
# EVOLVE-BLOCK-END
