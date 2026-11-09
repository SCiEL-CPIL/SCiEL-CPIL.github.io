"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    diff = (labels[:, None] != labels[None, :]).astype(np.int8)
    return int((adjacency & diff).sum() // 2)


def _compute_gains(adj, labels):
    """Compute the gain of flipping each vertex."""
    n = len(labels)
    gains = np.zeros(n, dtype=np.int32)
    for i in range(n):
        # Count neighbors with same label and different label
        same_count = int((adj[i] & (labels == labels[i])).sum())
        other_count = int((adj[i] & (labels == 1 - labels[i])).sum())
        # Gain from flipping: same_count becomes cut edges, other_count stops being cut
        gains[i] = same_count - other_count
    return gains


def _local_search(adjacency, labels, max_time=None):
    """Local search with 1-flip and 2-flip improvements."""
    n = len(labels)
    labels = labels.copy()
    adj = adjacency.astype(np.int8)
    
    start_time = time.time()
    deadline = start_time + (max_time if max_time else 100)
    
    current_cut = _cut_size(adjacency, labels)
    
    improved = True
    while improved:
        improved = False
        
        # Check deadline
        if time.time() > deadline:
            return labels, current_cut
        
        # Try 1-flip improvements
        gains = _compute_gains(adj, labels)
        best_gain = np.max(gains)
        
        if best_gain > 0:
            # Apply all positive gains (or just the best one)
            for v in range(n):
                if time.time() > deadline:
                    return labels, current_cut
                if gains[v] > 0:
                    labels[v] = 1 - labels[v]
                    current_cut += gains[v]
                    improved = True
                    # Recompute gains after each flip since they change
                    gains = _compute_gains(adj, labels)
        
        # Try 2-flip improvements if no 1-flip found
        if not improved:
            # Sample some pairs to try 2-flips
            if time.time() > deadline:
                return labels, current_cut
            
            # Try a few random pairs
            num_pairs_to_try = min(100, n * (n - 1) // 2)
            tried = set()
            
            for _ in range(num_pairs_to_try):
                if time.time() > deadline:
                    return labels, current_cut
                
                i = np.random.randint(0, n)
                j = np.random.randint(0, n)
                if i == j or (min(i, j), max(i, j)) in tried:
                    continue
                tried.add((min(i, j), max(i, j)))
                
                # Calculate gain for flipping both i and j
                # Current contribution of edges involving i or j to cut
                # After flip, edges that were same become different and vice versa
                
                li = labels[i]
                lj = labels[j]
                
                # Count edges from i to same partition (will become cut after flip)
                i_same = int((adj[i] & (labels == li)).sum())
                i_other = int((adj[i] & (labels == 1 - li)).sum())
                
                # Count edges from j to same partition (will become cut after flip)
                j_same = int((adj[j] & (labels == lj)).sum())
                j_other = int((adj[j] & (labels == 1 - lj)).sum())
                
                # Edge between i and j: if they have same label, it's not cut now, will be cut after
                # if different label, it's cut now, won't be cut after
                if li == lj:
                    # Edge i-j not cut now, will be cut after
                    edge_ij_gain = 1 if adj[i, j] else 0
                    # But we need to be careful: when we flip both, the edge between them
                    # goes from same to different, so +1 if edge exists
                    # However, in the counts above, we counted i's neighbors including j
                    # Let's recalculate more carefully
                    
                    # For vertex i: currently i_other edges are cut, i_same are not
                    # After flip: i_same edges become cut, i_other are not
                    # But edge i-j: if li==lj, it's in i_same count for i and j_same count for j
                    # After flip, it becomes cut, so it contributes +1 to cut
                    # But we counted it in both i_same and j_same, so we need to adjust
                    
                    # Net gain for i alone: i_same - i_other
                    # Net gain for j alone: j_same - j_other
                    # But edge i-j is counted in both, and after flipping both, 
                    # it goes from non-cut to cut (if li==lj)
                    # So the combined gain is:
                    # (i_same - i_other) + (j_same - j_other) + 2*edge_ij
                    # Wait, let me think again...
                    
                    # Actually, let's just compute the change directly
                    # Before: cut edges involving i or j
                    # After: cut edges involving i or j after flip
                    
                    # Simpler: just flip and check
                    old_labels = labels.copy()
                    labels[i] = 1 - li
                    labels[j] = 1 - lj
                    new_cut = _cut_size(adjacency, labels)
                    delta = new_cut - current_cut
                    labels = old_labels
                    
                    if delta > 0:
                        labels[i] = 1 - li
                        labels[j] = 1 - lj
                        current_cut = new_cut
                        improved = True
                        break
        
        # Check deadline again
        if time.time() > deadline:
            return labels, current_cut
    
    return labels, current_cut


def partition(adjacency):
    """Find a good partition using multiple restarts and local search."""
    n = len(adjacency)
    start_time = time.time()
    total_deadline = start_time + 150  # 150 second limit
    
    best_labels = None
    best_cut = -1
    
    # Number of restarts to try
    # We'll do random restarts with local search
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > total_deadline - 10:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Run local search
        labels, cut = _local_search(adjacency, labels, max_time=total_deadline - time.time() - 5)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # If we've used up most of our time, stop
        if time.time() > total_deadline - 15:
            break
    
    # If no best found, use alternating
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
