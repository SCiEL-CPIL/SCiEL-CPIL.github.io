"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _cut_size(adjacency, labels):
    """Compute the number of edges crossing the partition."""
    # labels is a 0/1 array of length n
    # For each edge (i,j) with i<j, it crosses if labels[i] != labels[j]
    # Using matrix form: cut = sum over i<j of A[i,j] * (1 if labels[i]!=labels[j] else 0)
    # = 0.5 * sum_{i,j} A[i,j] * (labels[i] != labels[j])
    # Since A is symmetric and 0-diagonal:
    # labels_bool = labels.astype(bool)
    # diff = labels_bool[:, None] ^ labels_bool[None, :]
    # cut = (adjacency * diff).sum() / 2
    # More efficiently:
    # Let s = labels (0/1). For each vertex i, edges to same side = sum_j A[i,j] where s[j]==s[i]
    # edges to other side = sum_j A[i,j] where s[j]!=s[i]
    # Total cut = sum_i (edges to other side) / 2 ... no, each crossing edge counted once from each side? 
    # Actually each crossing edge (i,j) with s[i]!=s[j] is counted in i's "other" and j's "other", so total = 2*cut
    # So cut = 0.5 * sum_i (sum_j A[i,j] * (s[i]!=s[j]))
    # Alternative: cut = 0.5 * (sum_i deg[i] - sum_i (sum_j A[i,j]*(s[j]==s[i])))
    # Or directly:
    s = labels.astype(np.int8)
    # diff matrix: 1 where labels differ
    diff = (s[:, None] != s[None, :]).astype(np.int8)
    cut = int((adjacency * diff).sum() / 2)
    return cut


def _local_search(adjacency, labels, time_limit):
    """Greedy local search: flip vertex if it improves cut size."""
    n = len(labels)
    labels = labels.copy()
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Precompute degrees
    deg = adj.sum(axis=1)
    
    # For each vertex, track how many neighbors are in the same group
    # same_count[v] = number of neighbors of v in the same group as v
    # When we flip v, the change in cut is: 
    #   new_cut = old_cut - (deg[v] - same_count[v]) + same_count[v]
    #   = old_cut + 2*same_count[v] - deg[v]
    # So flipping improves cut if same_count[v] > deg[v] - same_count[v], i.e., 2*same_count[v] > deg[v]
    
    # Initialize same_count
    s = labels.astype(np.int8)
    # same_count[v] = sum_j adj[v,j] * (s[j] == s[v])
    # = sum_j adj[v,j] for j where s[j] == s[v]
    # We can compute: for each v, same_count[v] = (adj[v] * (s == s[v])).sum()
    # But let's use a vectorized approach
    # same_count = (adj * (s[:, None] == s[None, :])).sum(axis=1)
    # This is O(n^2) memory but n=100 so it's fine
    
    start_time = time.time()
    improved = True
    while improved:
        improved = False
        # Compute same_count for all vertices at once
        same_mask = (s[:, None] == s[None, :]).astype(np.int8)
        same_count = (adj * same_mask).sum(axis=1)
        
        # Find vertices where flipping would improve
        # Improvement condition: 2*same_count[v] > deg[v]
        # i.e., same_count[v] > deg[v] - same_count[v]
        # i.e., 2*same_count[v] - deg[v] > 0
        delta = 2 * same_count - deg
        candidates = np.where(delta > 0)[0]
        
        if len(candidates) == 0:
            break
        
        # Flip all candidates? Or one at a time? 
        # Flipping multiple at once can be problematic since they interact.
        # Let's flip the one with the largest improvement first, then recompute.
        # For efficiency with n=100, let's do sequential but vectorized where possible.
        
        # Actually, let's flip all that improve simultaneously, then recompute.
        # This is a parallel hill-climbing step. It may not converge to the same local optimum
        # as sequential, but it's faster.
        for v in candidates:
            s[v] = 1 - s[v]
            improved = True
        
        # Check time
        if time.time() - start_time > time_limit:
            break
    
    labels = s.astype(np.int8)
    return labels


def _simulated_annealing(adjacency, labels, time_limit, T0=10.0, cooling=0.99):
    """Simulated annealing for Max-Cut."""
    n = len(labels)
    labels = labels.copy().astype(np.int8)
    adj = np.asarray(adjacency, dtype=np.int8)
    deg = adj.sum(axis=1)
    
    start_time = time.time()
    T = T0
    
    # Compute initial cut
    s = labels
    same_mask = (s[:, None] == s[None, :]).astype(np.int8)
    same_count = (adj * same_mask).sum(axis=1)
    cut = int((deg - same_count).sum() / 2)
    
    best_cut = cut
    best_labels = s.copy()
    
    while time.time() - start_time < time_limit:
        # Pick a random vertex to flip
        v = np.random.randint(n)
        old_s = s[v]
        s[v] = 1 - old_s
        
        # Recompute same_count for v
        # same_count[v] changes: neighbors that were same are now different and vice versa
        # old_same = same_count[v]
        # new_same = deg[v] - old_same  (since all neighbors flip their relationship)
        # Wait, that's not right. same_count[v] = number of neighbors of v in same group.
        # When we flip v, the neighbors that were in same group as v are now in different group,
        # and those in different group are now in same group.
        # So new_same_count[v] = deg[v] - old_same_count[v]
        
        old_same = same_count[v]
        new_same = deg[v] - old_same
        same_count[v] = new_same
        
        # But we also need to update same_count for all neighbors of v
        # For each neighbor u of v:
        #   if u was in same group as v before flip: now u is in different group from v
        #   so same_count[u] decreases by 1
        #   if u was in different group from v before flip: now u is in same group as v
        #   so same_count[u] increases by 1
        neighbors = np.where(adj[v] > 0)[0]
        for u in neighbors:
            if s[u] == s[v]:  # now in same group
                same_count[u] += 1
            else:
                same_count[u] -= 1
        
        # Compute change in cut
        # cut = 0.5 * sum_i (deg[i] - same_count[i])
        # Change: for v, deg[v] - same_count[v] changes from deg[v] - old_same to deg[v] - new_same = deg[v] - (deg[v] - old_same) = old_same
        # So delta for v: old_same - (deg[v] - old_same) = 2*old_same - deg[v]
        # For each neighbor u: deg[u] - same_count[u] changes by +/-1
        # If u now in same group as v: same_count[u] increased by 1, so deg[u]-same_count[u] decreased by 1
        # If u now in different group from v: same_count[u] decreased by 1, so deg[u]-same_count[u] increased by 1
        
        delta_v = 2 * old_same - deg[v]
        delta_neighbors = 0
        for u in neighbors:
            if s[u] == s[v]:
                delta_neighbors -= 1
            else:
                delta_neighbors += 1
        
        delta_cut = (delta_v + delta_neighbors) / 2
        
        if delta_cut > 0:
            cut += delta_cut
            if cut > best_cut:
                best_cut = cut
                best_labels = s.copy()
        else:
            # Accept with probability exp(delta_cut / T)
            if T > 0 and np.random.rand() < np.exp(delta_cut / T):
                cut += delta_cut
            else:
                # Reject: undo the flip
                s[v] = old_s
                same_count[v] = old_same
                for u in neighbors:
                    if s[u] == s[v]:
                        same_count[u] -= 1
                    else:
                        same_count[u] += 1
        
        T *= cooling
    
    return best_labels.astype(np.int8), best_cut


def partition(adjacency):
    """
    Partition vertices into two groups to maximize the number of crossing edges.
    Uses simulated annealing with local search polish.
    """
    n = len(adjacency)
    adj = np.asarray(adjacency, dtype=np.int8)
    
    # Deadline guard: 120s limit, use 110s to be safe
    total_time_limit = 110.0
    start_total = time.time()
    
    best_labels = None
    best_cut = -1
    
    # Multiple restarts with different initializations
    n_restarts = 5
    time_per_restart = total_time_limit / n_restarts
    
    for restart in range(n_restarts):
        elapsed = time.time() - start_total
        remaining = total_time_limit - elapsed
        if remaining < 5.0:
            break
        
        # Initialize
        if restart == 0:
            labels = np.arange(n) % 2
        elif restart == 1:
            labels = np.random.randint(0, 2, n)
        else:
            # Random with bias
            labels = (np.random.rand(n) < 0.5).astype(np.int8)
        
        # Phase 1: Local search to get a good starting point
        ls_time = min(5.0, remaining * 0.3)
        labels = _local_search(adj, labels, ls_time)
        
        # Compute cut
        s = labels.astype(np.int8)
        same_mask = (s[:, None] == s[None, :]).astype(np.int8)
        same_count = (adj * same_mask).sum(axis=1)
        deg = adj.sum(axis=1)
        cut = int((deg - same_count).sum() / 2)
        
        if cut > best_cut:
            best_cut = cut
            best_labels = labels.copy()
        
        # Phase 2: Simulated annealing
        sa_time = min(remaining * 0.6, total_time_limit - (time.time() - start_total) - 5.0)
        if sa_time > 1.0:
            sa_labels, sa_cut = _simulated_annealing(adj, labels, sa_time)
            if sa_cut > best_cut:
                best_cut = sa_cut
                best_labels = sa_labels.copy()
        
        # Phase 3: Polish with local search
        polish_time = min(3.0, total_time_limit - (time.time() - start_total) - 2.0)
        if polish_time > 0.5 and best_labels is not None:
            polished = _local_search(adj, best_labels, polish_time)
            s = polished.astype(np.int8)
            same_mask = (s[:, None] == s[None, :]).astype(np.int8)
            same_count = (adj * same_mask).sum(axis=1)
            deg = adj.sum(axis=1)
            cut = int((deg - same_count).sum() / 2)
            if cut > best_cut:
                best_cut = cut
                best_labels = polished.copy()
    
    # Final check
    if best_labels is None:
        best_labels = np.arange(n) % 2
    
    return best_labels
# EVOLVE-BLOCK-END
