"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    n = len(labels)
    # Edge (i,j) crosses if labels[i] != labels[j]
    # Use boolean mask
    cross = (labels[:, None] != labels[None, :]) & adjacency
    return int(cross.sum())


def _local_search(adjacency, labels, time_limit=None):
    """
    Greedy local search with incremental cut updates.
    labels: array of 0/1
    Returns improved labels and cut size.
    """
    n = len(labels)
    labels = labels.copy()
    
    # Precompute adjacency as 0/1 int array for speed
    adj = adjacency.astype(np.int8)
    
    # For each vertex, track the number of neighbors in the same group
    # same_count[v] = number of neighbors of v that have the same label as v
    # When we flip v, the gain in cut = other_count - same_count
    # where other_count = degree[v] - same_count[v]
    
    # Initial computation
    # labels is 0/1, so same label means labels[v] == labels[u]
    # same_count[v] = sum over u where labels[u] == labels[v] of adj[v,u]
    
    # Efficient: for each v, same_count[v] = sum(adj[v] * (labels == labels[v]))
    # We can compute this in a loop or vectorized
    
    # Vectorized initial:
    # Create a matrix where M[v,u] = adj[v,u] if labels[v]==labels[u] else 0
    # same_count = M.sum(axis=1)
    
    # But that's O(n^2) memory. For n=100, it's fine.
    
    # Actually, let's use a more memory-efficient approach:
    # same_count[v] = (adj[v] * (labels == labels[v])).sum()
    
    same_count = np.zeros(n, dtype=np.int32)
    for v in range(n):
        same_count[v] = (adj[v] * (labels == labels[v])).sum()
    
    degrees = adj.sum(axis=1).astype(np.int32)
    
    def _update_same_count(v, old_label, new_label):
        """Update same_count for all neighbors of v when v flips from old_label to new_label."""
        # v's neighbors u: 
        # If u had same label as old_label (i.e., labels[u] == old_label), then before flip, 
        # adj[v,u] contributed to same_count[u] (since labels[u]==labels[v]==old_label).
        # After flip, labels[v]=new_label != old_label, so adj[v,u] no longer contributes to same_count[u].
        # If u had different label from old_label (i.e., labels[u] != old_label), then before flip,
        # adj[v,u] did NOT contribute to same_count[u]. After flip, labels[v]=new_label, 
        # and if labels[u]==new_label, then it now contributes.
        
        # More simply: for each neighbor u of v:
        #   if labels[u] == old_label: same_count[u] -= 1 (no longer same as v)
        #   if labels[u] == new_label: same_count[u] += 1 (now same as v)
        
        neighbors = np.where(adj[v] > 0)[0]
        for u in neighbors:
            if labels[u] == old_label:
                same_count[u] -= 1
            elif labels[u] == new_label:
                same_count[u] += 1
        # Also update v's own same_count: it becomes the number of neighbors with new_label
        # which is (degrees[v] - same_count[v]) before flip... wait, let's just recompute
        # Actually, after flipping v, same_count[v] = number of neighbors u with labels[u] == new_label
        # Before flip, same_count[v] = number of neighbors u with labels[u] == old_label
        # So same_count[v] after = degrees[v] - same_count[v] before
        # But we need to be careful: we're updating same_count for neighbors first, then v.
        # Let's just recompute v's same_count at the end.
        
        # Actually, the gain for flipping v is:
        #   gain = other_count - same_count = (degrees[v] - same_count[v]) - same_count[v] = degrees[v] - 2*same_count[v]
        # If gain > 0, flip v.
        
        # After flipping v, same_count[v] becomes degrees[v] - old_same_count[v]
        # And for neighbors, we update as above.
        
    # We'll do iterative passes until no improvement or time limit
    best_labels = labels.copy()
    best_cut = _cut_size(adjacency, labels)
    
    start_time = time.time()
    
    improved = True
    while improved:
        improved = False
        for v in range(n):
            if time_limit is not None and time.time() - start_time > time_limit:
                break
            old_label = labels[v]
            sc = same_count[v]
            deg = degrees[v]
            gain = deg - 2 * sc
            if gain > 0:
                new_label = 1 - old_label
                # Update same_count for neighbors
                neighbors = np.where(adj[v] > 0)[0]
                for u in neighbors:
                    if labels[u] == old_label:
                        same_count[u] -= 1
                    elif labels[u] == new_label:
                        same_count[u] += 1
                # Update v's same_count
                same_count[v] = deg - sc
                labels[v] = new_label
                improved = True
        if improved and time_limit is not None and time.time() - start_time > time_limit:
            break
    
    # Compute final cut size
    final_cut = _cut_size(adjacency, labels)
    if final_cut > best_cut:
        best_labels = labels.copy()
        best_cut = final_cut
    
    return best_labels, best_cut


def partition(adjacency):
    n = len(adjacency)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    # 1. Alternating
    # 2. Random
    # 3. Random with different seeds
    
    rng = np.random.RandomState(42)
    
    initializations = []
    # Alternating
    initializations.append(np.arange(n) % 2)
    # A few random ones
    for i in range(5):
        initializations.append(rng.randint(0, 2, size=n))
    
    # Try each initialization
    for init_labels in initializations:
        if time.time() > deadline:
            break
        # Local search from this init
        labels, cut = _local_search(adjacency, init_labels, time_limit=deadline - time.time())
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    # If we still have time, do more random restarts
    while time.time() < deadline - 1.0:
        init_labels = rng.randint(0, 2, size=n)
        labels, cut = _local_search(adjacency, init_labels, time_limit=deadline - time.time())
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
    
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
