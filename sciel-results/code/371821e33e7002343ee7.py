"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _cut_size(adj, labels):
    """Calculate the number of edges crossing the cut."""
    # Vectorized calculation: sum of (label_i != label_j) * adj_ij
    # We can compute this efficiently.
    # cut = sum_{i<j} adj[i,j] * (1 - (labels[i] == labels[j]))
    # Or simpler: for each node i, count neighbors j where labels[i] != labels[j].
    # Sum these counts and divide by 2 (since each edge is counted twice).
    # Using matrix multiplication:
    # Let L be the label vector (0 or 1).
    # Number of edges from i to 1s: adj @ (labels == 1)
    # Number of edges from i to 0s: adj @ (labels == 0)
    # For node i with label 0, crossing edges = edges to 1s.
    # For node i with label 1, crossing edges = edges to 0s.
    # Total cut = sum(crossing_edges)
    
    ones = (labels == 1).astype(np.float64)
    zeros = (labels == 0).astype(np.float64)
    
    # adj @ ones gives for each node the number of neighbors in set 1
    # adj @ zeros gives for each node the number of neighbors in set 0
    
    # If label[i] == 0, we want adj[i, ones]
    # If label[i] == 1, we want adj[i, zeros]
    
    # We can do:
    # crossing = (labels == 0) * (adj @ ones) + (labels == 1) * (adj @ zeros)
    # Sum this up.
    
    # Note: adj is boolean or float. Ensure float for matmul.
    # The input adjacency is a bool matrix, but we can convert to float once outside.
    
    # Let's assume adj is float64 for speed.
    # This calculation is O(N^2) roughly due to matmul.
    
    res = (labels == 0) * (adj @ ones) + (labels == 1) * (adj @ zeros)
    return np.sum(res)

def _local_search(adj, labels, deadline):
    """
    Perform local search until no improvement or deadline.
    Returns best labels and best cut size.
    """
    n = len(labels)
    best_labels = labels.copy()
    best_cut = _cut_size(adj, best_labels)
    
    # Pre-compute degrees? Not strictly necessary for the improvement check.
    # Improvement for flipping node v:
    # Current contribution of v to cut: 
    #   if label[v]=0: deg_1(v)
    #   if label[v]=1: deg_0(v)
    # After flip:
    #   if label[v]=0 -> 1: new contribution is deg_0(v)
    #   if label[v]=1 -> 0: new contribution is deg_1(v)
    # Delta = New - Old
    # If label[v]=0: Delta = deg_0(v) - deg_1(v). Flip if Delta > 0 => deg_0(v) > deg_1(v)
    # If label[v]=1: Delta = deg_1(v) - deg_0(v). Flip if Delta > 0 => deg_1(v) > deg_0(v)
    # In both cases, flip if the number of neighbors in the *other* set is greater than in the *same* set.
    
    # We can vectorize the "should flip" decision for all nodes at once?
    # Yes.
    # deg_1 = adj @ (labels == 1)
    # deg_0 = adj @ (labels == 0)
    # same = np.where(labels == 0, deg_1, deg_0)  # Wait.
    # If label is 0, same-set neighbors are 0s. So deg_same = deg_0.
    # If label is 1, same-set neighbors are 1s. So deg_same = deg_1.
    # other = np.where(labels == 0, deg_1, deg_0)
    
    # Condition: other > same
    
    while True:
        if time.time() > deadline:
            break
            
        ones = (labels == 1).astype(np.float64)
        zeros = (labels == 0).astype(np.float64)
        
        deg_1 = adj @ ones
        deg_0 = adj @ zeros
        
        # deg_same: number of neighbors with same label
        # deg_other: number of neighbors with different label
        deg_same = np.where(labels == 0, deg_0, deg_1)
        deg_other = np.where(labels == 0, deg_1, deg_0)
        
        # Nodes that should flip
        to_flip = np.where(deg_other > deg_same)[0]
        
        if len(to_flip) == 0:
            break
            
        # Flip all at once
        labels[to_flip] = 1 - labels[to_flip]
        
        # Update best
        current_cut = _cut_size(adj, labels)
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels, best_cut

def partition(adjacency):
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.float64)
    
    # Deadline: 110s to be safe (limit is 120s, but overhead exists)
    deadline = time.time() + 110
    
    best_labels = None
    best_cut = -1
    
    # 1. Spectral Initialization
    # Compute Laplacian L = D - A
    degrees = adj.sum(axis=1)
    D = np.diag(degrees)
    L = D - adj
    
    # Get Fiedler vector (2nd smallest eigenvalue)
    # For N=100, full eigendecomposition is fast.
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
        # Sort by eigenvalue
        idx = np.argsort(eigvals)
        fiedler = eigvecs[:, idx[1]]
        
        # Initialize labels based on sign of Fiedler vector
        # Center it to 0 to handle bias
        fiedler_centered = fiedler - np.mean(fiedler)
        init_labels = (fiedler_centered > 0).astype(np.int32)
        
        # Refine with local search
        ls_labels, ls_cut = _local_search(adj, init_labels.copy(), deadline)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
            
    except Exception:
        pass
        
    # 2. Random Restarts
    # If we have time, do random restarts
    # Each restart: random init -> local search
    # Estimate time per restart. Local search on 100 nodes is very fast.
    # Let's try as many as possible until deadline.
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5: # Leave 5s buffer
        # Random initialization
        random_labels = rng.integers(0, 2, size=n).astype(np.int32)
        
        # Run local search
        ls_labels, ls_cut = _local_search(adj, random_labels, deadline)
        
        if ls_cut > best_cut:
            best_cut = ls_cut
            best_labels = ls_labels
            
        # Check if we converged to the same best? Hard to know without comparing labels.
        # We'll just keep going until time runs out or we hit a high confidence.
        # For N=100, random restarts are cheap.
        
    if best_labels is None:
        # Fallback: alternating
        best_labels = np.arange(n) % 2
        best_cut = _cut_size(adj, best_labels)
        
    return best_labels
# EVOLVE-BLOCK-END
