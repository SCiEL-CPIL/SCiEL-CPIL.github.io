# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _cut_size(adj, labels):
    """Compute the number of edges crossing the 0/1 partition."""
    diff = (labels[:, None] != labels[None, :])
    return int(np.sum(adj * diff) / 2)


def _spectral_init(adj, n, eig_idx):
    """Create a partition from the eigenvector of the graph Laplacian."""
    deg = np.sum(adj, axis=1)
    D = np.diag(deg)
    L = D - adj
    
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        if eig_idx < eigvecs.shape[1]:
            vec = eigvecs[:, eig_idx]
        else:
            vec = eigvecs[:, -1]
        
        median_val = np.median(vec)
        labels = (vec > median_val).astype(np.int64)
        
        if np.all(labels == 0) or np.all(labels == 1):
            labels = (vec >= 0).astype(np.int64)
            if np.all(labels == 0) or np.all(labels == 1):
                labels = np.arange(n) % 2
    except Exception:
        labels = np.arange(n) % 2
    
    return labels


def _goemans_maxwell_init(adj, n):
    """Goemans-Williamson style randomized rounding from SDP-like matrix.
    We approximate the SDP relaxation by using the normalized adjacency
    and computing a random hyperplane rounding."""
    # Normalize adjacency to get a correlation-like matrix
    deg = np.sum(adj, axis=1)
    deg_safe = np.where(deg > 0, deg, 1)
    # D^{-1/2} A D^{-1/2}
    d_inv_sqrt = 1.0 / np.sqrt(deg_safe)
    W = adj * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
    # Symmetrize
    W = (W + W.T) / 2.0
    # Make it a valid correlation matrix by ensuring diag is 1
    # Actually W has diag = 0 for no self-loops, but let's just use it
    # Compute eigendecomposition
    try:
        eigvals, eigvecs = np.linalg.eigh(W)
        # Take top k eigenvectors for embedding
        k = min(n, 20)
        V = eigvecs[:, :k] * np.sqrt(np.maximum(eigvals[:k], 0))[:, None]
        # Normalize rows to unit length
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        V = V / norms
        
        # Random hyperplane rounding
        r = np.random.randn(k)
        proj = V @ r
        labels = (proj > 0).astype(np.int64)
        
        if np.all(labels == 0) or np.all(labels == 1):
            labels = np.arange(n) % 2
    except Exception:
        labels = np.random.randint(0, 2, n)
    
    return labels


def _greedy_refine(adj, labels, max_time=100.0):
    """Greedy local search: flip vertices to improve cut size."""
    n = len(labels)
    labels = labels.copy()
    
    current_cut = _cut_size(adj, labels)
    
    deadline = time.time() + max_time
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for v in range(n):
            if time.time() > deadline:
                break
            
            same_side = (labels == labels[v])
            other_side = (labels != labels[v])
            
            same_count = int(np.sum(adj[v, same_side]))
            other_count = int(np.sum(adj[v, other_side]))
            
            if same_count > other_count:
                labels[v] = 1 - labels[v]
                current_cut += same_count - other_count
                improved = True
    
    return labels, current_cut


def _pairwise_swap(adj, labels, max_time=50.0):
    """Try pairwise swaps to escape local optima."""
    n = len(labels)
    labels = labels.copy()
    current_cut = _cut_size(adj, labels)
    
    deadline = time.time() + max_time
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i+1, n):
                if time.time() > deadline:
                    break
                if labels[i] == labels[j]:
                    continue
                
                # Compute delta for swapping i and j
                # Edges from i to same side (label 0) minus edges to other side (label 1)
                # After swap, i gets label 1, j gets label 0
                
                # Current contribution of i: edges from i to label 1
                # Current contribution of j: edges from j to label 0
                # After swap: i contributes edges to label 0, j contributes edges to label 1
                
                # Delta = (edges i->label0 - edges i->label1) + (edges j->label1 - edges j->label0)
                # But we need to be careful about the edge between i and j
                
                li = labels[i]
                lj = labels[j]
                
                # Same side as i (before swap)
                same_i = (labels == li) & (np.arange(n) != i)
                other_i = (labels != li) & (np.arange(n) != i)
                
                same_j = (labels == lj) & (np.arange(n) != j)
                other_j = (labels != lj) & (np.arange(n) != j)
                
                # Current: i contributes sum(adj[i, other_i]), j contributes sum(adj[j, other_j])
                # After swap: i gets label lj, so same side is other_i (now), contributes sum(adj[i, same_i])
                # j gets label li, contributes sum(adj[j, same_j])
                
                cur_contrib = np.sum(adj[i, other_i]) + np.sum(adj[j, other_j])
                new_contrib = np.sum(adj[i, same_i]) + np.sum(adj[j, same_j])
                
                delta = new_contrib - cur_contrib
                
                if delta > 0:
                    labels[i] = lj
                    labels[j] = li
                    current_cut += delta
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
        
        T = start_temp * (end_temp / start_temp) ** (it / max(n_iterations - 1, 1))
        
        v = np.random.randint(n)
        old_label = labels[v]
        
        same_side = (labels == old_label)
        other_side = (labels != old_label)
        
        same_count = int(np.sum(adj[v, same_side]))
        other_count = int(np.sum(adj[v, other_side]))
        
        delta = same_count - other_count
        
        if delta > 0 or (T > 0 and np.random.random() < np.exp(delta / T)):
            labels[v] = 1 - labels[v]
            current_cut += delta
            
            if current_cut > best_cut:
                best_cut = current_cut
                best_labels = labels.copy()
    
    return best_labels, best_cut


def _multi_start_search(adj, n, deadline):
    """Run multiple initializations and local searches, return best."""
    best_labels = None
    best_cut = -1
    
    # Try spectral initializations
    for eig_idx in [1, 2, 3]:
        if time.time() > deadline:
            break
        
        labels = _spectral_init(adj, n, eig_idx)
        refined_labels, cut_val = _greedy_refine(adj, labels, max_time=20.0)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        # Pairwise swap
        if time.time() < deadline - 10.0:
            swap_labels, swap_cut = _pairwise_swap(adj, refined_labels, max_time=10.0)
            if swap_cut > best_cut:
                best_cut = swap_cut
                best_labels = swap_labels.copy()
        
        # Simulated annealing
        if time.time() < deadline - 20.0:
            sa_labels, sa_cut = _simulated_annealing(adj, refined_labels, 
                                                       start_temp=5.0, 
                                                       end_temp=0.1,
                                                       n_iterations=2000,
                                                       max_time=15.0)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    # Try Goemans-Williamson style initialization
    for _ in range(3):
        if time.time() > deadline:
            break
        
        labels = _goemans_maxwell_init(adj, n)
        refined_labels, cut_val = _greedy_refine(adj, labels, max_time=20.0)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        if time.time() < deadline - 10.0:
            swap_labels, swap_cut = _pairwise_swap(adj, refined_labels, max_time=10.0)
            if swap_cut > best_cut:
                best_cut = swap_cut
                best_labels = swap_labels.copy()
        
        if time.time() < deadline - 20.0:
            sa_labels, sa_cut = _simulated_annealing(adj, refined_labels, 
                                                       start_temp=5.0, 
                                                       end_temp=0.1,
                                                       n_iterations=2000,
                                                       max_time=15.0)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
    
    # Random starts
    for _ in range(5):
        if time.time() > deadline:
            break
        
        labels = np.random.randint(0, 2, n)
        refined_labels, cut_val = _greedy_refine(adj, labels, max_time=15.0)
        
        if cut_val > best_cut:
            best_cut = cut_val
            best_labels = refined_labels.copy()
        
        if time.time() < deadline - 10.0:
            swap_labels, swap_cut = _pairwise_swap(adj, refined_labels, max_time=10.0)
            if swap_cut > best_cut:
                best_cut = swap_cut
                best_labels = swap_labels.copy()
    
    # Alternating start
    if time.time() < deadline - 10.0:
        alt_labels = np.arange(n) % 2
        refined_alt, alt_cut = _greedy_refine(adj, alt_labels, max_time=15.0)
        if alt_cut > best_cut:
            best_cut = alt_cut
            best_labels = refined_alt.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels, best_cut


def partition(adjacency):
    """Find a high-quality 2-partition of the graph using multiple initialization
    strategies and local search."""
    n = len(adjacency)
    
    adj = np.asarray(adjacency, dtype=np.float64)
    
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Set random seed for reproducibility within a single call
    rng = np.random.default_rng()
    
    best_labels, best_cut = _multi_start_search(adj, n, deadline)
    
    return best_labels
# EVOLVE-BLOCK-END
