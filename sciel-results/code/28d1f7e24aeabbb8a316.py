# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    # labels is 0/1 array
    # An edge (i,j) crosses if labels[i] != labels[j]
    # For undirected graph with symmetric adjacency, each crossing edge is counted once in the sum
    # but we want the count of crossing edges.
    # Using the formula: cut = sum_{i,j} A[i,j] * (labels[i] != labels[j]) / 2
    # since each crossing edge appears twice in the double sum (i,j) and (j,i)
    diff = (labels[:, None] != labels[None, :])
    return int(np.sum(adj * diff) / 2)


def _spectral_init(adj, n, eig_idx):
    """Create a partition from the eigenvector of the graph Laplacian."""
    # Build Laplacian L = D - A
    deg = np.sum(adj, axis=1)
    D = np.diag(deg)
    L = D - adj
    
    # Use scipy to compute eigenvectors
    # For small n (100), we can use dense eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # eigvecs columns are eigenvectors, sorted by eigenvalue
        # The Fiedler vector is the second smallest eigenvalue's eigenvector (index 1)
        # We want eig_idx-th eigenvector (0-indexed), so we pick eigvecs[:, eig_idx]
        if eig_idx < eigvecs.shape[1]:
            vec = eigvecs[:, eig_idx]
        else:
            vec = eigvecs[:, -1]
        
        # Create partition: split by median of the eigenvector values
        median_val = np.median(vec)
        labels = (vec > median_val).astype(np.int64)
        
        # Ensure both partitions are non-empty
        if np.all(labels == 0) or np.all(labels == 1):
            # Fallback: use sign
            labels = (vec >= 0).astype(np.int64)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.arange(n) % 2
    except Exception:
        labels = np.arange(n) % 2
    
    return labels


def _greedy_refine(adj, labels, max_time=100.0):
    """Greedy local search: flip vertices to improve cut size."""
    n = len(labels)
    labels = labels.copy()
    
    # Precompute row sums for efficiency
    current_cut = _cut_size(adj, labels)
    
    deadline = time.time() + max_time
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            
            # Count edges from v to same side and other side
            same_side = (labels == labels[v])
            other_side = (labels != labels[v])
            
            same_count = int(np.sum(adj[v, same_side]))
            other_count = int(np.sum(adj[v, other_side]))
            
            # If flipping v increases the cut, do it
            # Current contribution: other_count edges cross
            # After flip: same_count edges cross
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                current_cut += same_count - other_count
                improved = True
    
    return labels, current_cut


def _simulated_annealing(adj, labels, start_temp=10.0, end_temp=0.1, 
                          n_iterations=5000, max_time=100.0):
    """Simulated annealing to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    best_labels = labels.copy()
    best_cut = current_cut
    
    deadline = time.time() + max_time
    T = start_temp
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        # Update temperature
        T = start_temp * (end_temp / start_temp) ** (it / max(n_iterations - 1, 1))
        
        # Pick a random vertex to flip
        v = np.random.randint(n)
        old_label = labels[v]
        
        # Compute change in cut
        same_side = (labels == old_label)
        other_side = (labels != old_label)
        
        same_count = int(np.sum(adj[v, same_side]))
        other_count = int(np.sum(adj[v, other_side]))
        
        # If we flip v, the change in cut is:
        # Currently: other_count edges from v cross
        # After flip: same_count edges from v cross
        # Delta = same_count - other_count
        delta = same_count - other_count
        
        # Accept if delta > 0 or with probability exp(delta / T)
        if delta > 0 or (T > 0 and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def partition(adjacency):
    """Find a high-quality 2-partition of the graph using spectral initialization
    and local search with simulated annealing."""
    n = len(adjacency)
    
    # Convert to numpy array if needed
    adj = np.asarray(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Try multiple spectral initializations
    for eig_idx in [1, 2, 3]:  # 2nd, 3rd, 4th eigenvectors
        if time.time() > deadline:
            break
        
        labels = _spectral_init(adj, n, eig_idx)
        
        # Greedy refinement
        refined_labels, cut_val = _greedy_refine(adj, labels, max_time=30.0)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        # Simulated annealing from the greedy result
        if time.time() < deadline - 20.0:
            sa_labels, sa_cut = _simulated_annealing(adj, refined_labels, 
                                                       start_temp=5.0, 
                                                       end_temp=0.1,
                                                       n_iterations=3000,
                                                       max_time=30.0)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    # Also try a simple alternating start as fallback
    if time.time() < deadline - 10.0:
        alt_labels = np.arange(n) % 2
        refined_alt, alt_cut = _greedy_refine(adj, alt_labels, max_time=20.0)
        if alt_cut > best_cut:
            best_cut = alt_cut
            best_labels = refined_alt.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
