"""
Strategy: Vectorized Multi-Sweep Local Search for Maximum Cut.

The current approach uses sequential Simulated Annealing (SA) with single-node flips.
While SA is good for escaping local optima, its sequential nature in Python is slow
for N=100 if we want to perform many iterations.

This new strategy uses a "Vectorized Greedy Sweep" approach:
1. Compute the gain of flipping EVERY node simultaneously using matrix operations.
   Gain(v) = (neighbors with same label) - (neighbors with opposite label).
   If Gain(v) > 0, flipping v increases the cut size.
2. Identify all nodes where Gain(v) > 0.
3. Flip all those nodes simultaneously.
4. Repeat until no nodes have positive gain (local optimum reached).

This is effectively a parallelized version of the "flip if beneficial" heuristic.
To escape local optima, we use multiple random restarts.
Since the sweep is vectorized (O(N^2) per sweep but in C-speed NumPy), we can perform
many sweeps very quickly. For N=100, one sweep is extremely fast (<1ms).
We can afford to do thousands of sweeps or restarts within the 150s limit.

We will combine:
1. Spectral Initialization (Fiedler vector) for a strong start.
2. Vectorized Local Search (VLS) to polish.
3. Multiple Random Restarts with VLS to explore different basins.
"""

import numpy as np
import time

def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = labels[None, :] != labels[:, None]
    return int((adjacency & diff).sum() // 2)

def _spectral_init(adjacency):
    """Initialize using the sign of the Fiedler vector (2nd smallest eigenvector of Laplacian)."""
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1)
    L = np.diag(degrees) - adjacency.astype(float)
    
    try:
        vals, vecs = np.linalg.eigh(L)
        fiedler = vecs[:, 1]
        labels = (fiedler >= 0).astype(int)
        return labels
    except Exception:
        return np.random.randint(0, 2, size=n)

def _vectorized_local_search(adjacency, labels, max_sweeps=1000, deadline=None):
    """
    Perform local search by flipping all nodes that improve the cut simultaneously.
    Returns the improved labels and the cut size.
    """
    n = len(labels)
    labels = labels.copy()
    adj_bool = adjacency # Assume boolean
    
    current_cut = _cut_size(adj_bool, labels)
    best_cut = current_cut
    best_labels = labels.copy()
    
    # Precompute adjacency as float for potential weighted extensions, 
    # but here we just need boolean logic.
    # Gain calculation:
    # For each node v, gain = sum_{u ~ v} (1 if labels[u] == labels[v] else -1)
    # If gain > 0, flipping v increases cut.
    
    # Vectorized calculation of gains:
    # Let S = labels (0 or 1)
    # Let A = adjacency matrix (0 or 1)
    # We want for each v: sum_u A[v,u] * (1 if S[v]==S[u] else -1)
    # Note: S[v]==S[u] is 1 if S[v]+S[u] is even (0 or 2), -1 if odd (1).
    # Actually, it's easier to work with +/-1 representation for the math, 
    # but we can stick to 0/1 and use logical ops.
    
    # Let's use the property:
    # same = A & (S[:, None] == S[None, :])
    # diff = A & (S[:, None] != S[None, :])
    # gain = same.sum(axis=1) - diff.sum(axis=1)
    
    # This is O(N^2) per sweep. For N=100, this is 10,000 ops. 
    # NumPy handles this in microseconds.
    
    start_time = time.time()
    
    for sweep in range(max_sweeps):
        if deadline and time.time() > deadline:
            break
            
        # Calculate gains for all nodes
        # S is 1D array of 0/1
        # S_col is S[:, None], S_row is S[None, :]
        S_col = labels[:, None]
        S_row = labels[None, :]
        
        # Boolean matrix of where labels are same
        same_mask = (S_col == S_row) & adj_bool
        # Boolean matrix of where labels are diff
        diff_mask = (S_col != S_row) & adj_bool
        
        # Count same and diff neighbors for each node
        same_counts = same_mask.sum(axis=1)
        diff_counts = diff_mask.sum(axis=1)
        
        # Gain of flipping node v is: 
        # Edges that were same become diff (+1 to cut)
        # Edges that were diff become same (-1 to cut)
        # Delta = same_counts - diff_counts
        
        gains = same_counts - diff_counts
        
        # Find nodes where gain > 0
        flip_mask = gains > 0
        
        if not np.any(flip_mask):
            # No improvement possible, reached local optimum
            break
            
        # Flip all nodes in flip_mask simultaneously
        labels[flip_mask] = 1 - labels[flip_mask]
        
        # Update cut size
        # Instead of recalculating from scratch (O(N^2)), we can sum the gains
        # because the gains are independent? 
        # Wait, if we flip multiple nodes, the gains calculated for each node 
        # were based on the *previous* state. 
        # If we flip u and v, and u~v, the gain for u assumed v didn't flip.
        # So we must recalculate the cut size from scratch to be safe, 
        # or adjust.
        # Recalculating is O(N^2) and fast enough for N=100.
        current_cut = _cut_size(adj_bool, labels)
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = labels.copy()
            
    return best_labels, best_cut

def partition(adjacency):
    """
    Main partition function.
    1. Spectral Init
    2. Vectorized Local Search (VLS)
    3. Multiple Random Restarts with VLS
    """
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Phase 1: Spectral Initialization + VLS
    labels = _spectral_init(adjacency)
    labels, cut = _vectorized_local_search(adjacency, labels, max_sweeps=500, deadline=deadline - 5.0)
    if cut > best_cut:
        best_cut = cut
        best_labels = labels.copy()
    
    # Phase 2: Random Restarts with VLS
    # Since VLS is very fast, we can do many restarts.
    # Each restart involves random init + VLS.
    # Let's estimate time per restart.
    # One VLS run with 500 sweeps takes maybe 10-50ms for N=100.
    # So we can do ~100-1000 restarts in 140s.
    
    remaining_time = deadline - time.time() - 5.0
    if remaining_time > 1.0:
        # We will run restarts until time runs out.
        # To be safe, we'll limit the number of sweeps per restart in later runs
        # or just let it run until deadline.
        
        # Adaptive strategy: 
        # If a restart finds a better solution, we might want to polish it more?
        # For now, fixed max_sweeps for restarts to ensure we try many basins.
        
        restart_id = 0
        while time.time() < deadline - 2.0:
            labels = np.random.randint(0, 2, size=n)
            # Use fewer sweeps for random restarts to allow more restarts?
            # Or keep it same? 
            # Let's try 200 sweeps per restart.
            labels, cut = _vectorized_local_search(adjacency, labels, max_sweeps=200, deadline=deadline - 2.0)
            if cut > best_cut:
                best_cut = cut
                best_labels = labels.copy()
                # Optional: If we found a new best, we could do a deeper polish
                # but let's stick to the loop for now to maximize exploration.
            restart_id += 1
            
    if best_labels is None:
        best_labels = _spectral_init(adjacency)
        
    return best_labels
