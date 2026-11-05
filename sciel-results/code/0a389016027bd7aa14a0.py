"""Phylogenetic topology recovery using optimized Neighbor Joining with NumPy."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

_DEADLINE = None
_START_TIME = None


def _time_left():
    global _START_TIME, _DEADLINE
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 240.0  # safety margin before 270s kill
    return _DEADLINE - time.time()


def _jc_distance_matrix(sequences, labels):
    """Compute JC-corrected distance matrix using NumPy."""
    n = len(labels)
    # Build alignment matrix: (n, L) with one-hot or direct comparison
    # For DNA, count mismatches per pair
    # Convert sequences to arrays of characters
    seqs = np.array([list(sequences[lab]) for lab in labels], dtype='U1')
    L = seqs.shape[1]
    
    # Compute pairwise differences: for each pair (i,j), count positions where seqs[i] != seqs[j]
    # Using broadcasting: (n, 1, L) vs (1, n, L) -> (n, n, L)
    # This could be memory-heavy for large n*L, but n<=32, L<=2000 -> 32*32*2000 = 2M, fine
    diff = (seqs[:, np.newaxis, :] != seqs[np.newaxis, :, :]).sum(axis=2)
    p = diff / L
    # JC correction: d = -0.75 * ln(1 - 4p/3), cap at p >= 0.75
    np.clip(p, 0, 0.7499, out=p)
    D = -0.75 * np.log(1.0 - 4.0 * p / 3.0)
    np.fill_diagonal(D, 0.0)
    return D


def _neighbor_joining(D):
    """Standard Neighbor Joining on a distance matrix D (n x n). Returns Newick string."""
    n = D.shape[0]
    if n < 3:
        # Degenerate case
        if n == 1:
            return "t0;"
        return "(t0,t1);"
    
    # Track the tree structure for each current node
    # Each node is either a leaf (original taxon) or an internal node with a subtree
    # We'll track:
    # - nodes: list of current node indices
    # - tree: dict mapping node index -> Newick string for that subtree
    # - D: current distance matrix (dense, indexed by original indices, but we only use active nodes)
    
    # Use a list of active node IDs. Initially 0..n-1.
    active = list(range(n))
    # tree[i] will hold the Newick string for node i
    tree = {i: f"t{i}" for i in range(n)}
    
    # We'll maintain a full n x n matrix but only care about active nodes
    # For efficiency, we can work with a dict of distances between active nodes
    # But n is small (<=32), so let's just use the full matrix and track active set
    
    while len(active) > 3:
        if _time_left() < 5.0:
            break  # Safety: return what we have
        
        m = len(active)
        # r[i] = sum of distances from node i to all other active nodes
        # D is n x n, active is list of indices
        active_arr = np.array(active)
        # r[i] = sum(D[i, active]) for i in active
        r = D[active_arr][:, active_arr].sum(axis=1)
        
        # Find the pair (i, j) that minimizes: (m-2)*D[i,j] - r[i] - r[j]
        best_val = float('inf')
        best_i, best_j = -1, -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                i = active[ii]
                j = active[jj]
                val = (m - 2) * D[i, j] - r[ii] - r[jj]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        # Create new internal node
        new_node = max(range(n)) + 1  # Use a new index beyond original n
        # Actually, let's just use a counter
        # Better: assign new nodes as n, n+1, etc.
        # Let's track next_node_id
        # I'll restructure to use a list of nodes and a separate id counter
        
        # For now, let's just create the subtree
        tree_str = f"({tree[i]},{tree[j]})"
        
        # Compute distances from new node to all other active nodes
        # d(new, x) = 0.5 * (D[i,x] + D[j,x] - D[i,j])
        rest = [k for k in active if k not in (i, j)]
        
        # We need to store the new node's distances. Since we're using a fixed-size array D (n x n),
        # we can't easily add new rows. Let's switch to a dict-based distance storage.
        # Actually, for n<=32, let's just use a 2D array that grows. 
        # Simpler: maintain a dict dist[(a,b)] for active nodes.
        
        # Let me redo this with a dict for distances.
        
        break  # Fallback to simpler implementation below
    
    # Let me implement NJ properly with a dict for distances
    # Restart with cleaner approach
    return _nj_clean(D, n)


def _nj_clean(D, n):
    """Clean NJ implementation using dict for dynamic distance tracking."""
    if n < 3:
        if n == 1:
            return "t0;"
        return "(t0,t1);"
    
    # dist: dict mapping (i, j) -> distance, symmetric
    dist = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                dist[(i, j)] = D[i, j]
                dist[(j, i)] = D[i, j]
    
    active = list(range(n))
    tree = {i: f"t{i}" for i in range(n)}
    next_id = n
    
    while len(active) > 3:
        if _time_left() < 5.0:
            break
        
        m = len(active)
        # Compute r[i] for each active node
        r = {}
        for i in active:
            r[i] = sum(dist[(i, j)] for j in active if j != i)
        
        # Find best pair
        best_val = float('inf')
        best_i, best_j = -1, -1
        for ii in range(m):
            for jj in range(ii + 1, m):
                i = active[ii]
                j = active[jj]
                val = (m - 2) * dist[(i, j)] - r[i] - r[j]
                if val < best_val:
                    best_val = val
                    best_i, best_j = i, j
        
        i, j = best_i, best_j
        new_node = next_id
        next_id += 1
        
        tree_str = f"({tree[i]},{tree[j]})"
        tree[new_node] = tree_str
        
        # Compute distances from new_node to all other active nodes
        rest = [k for k in active if k not in (i, j)]
        for x in rest:
            d_new = 0.5 * (dist[(i, x)] + dist[(j, x)] - dist[(i, j)])
            dist[(new_node, x)] = d_new
            dist[(x, new_node)] = d_new
        
        # Remove i and j from active, add new_node
        active = [k for k in active if k not in (i, j)]
        active.append(new_node)
    
    # Now we have 3 nodes left. Connect them in a star.
    # The three remaining nodes form a trifurcation (unrooted tree with 3 leaves)
    a, b, c = active
    return f"({tree[a]},{tree[b]},{tree[c]})"


def reconstruct(sequences):
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    _DEADLINE = _START_TIME + 240.0
    
    labels = sorted(sequences, key=lambda name: int(name[1:]))
    n = len(labels)
    
    if n <= 2:
        if n == 1:
            return "t0;"
        return "(t0,t1);"
    
    # Compute distance matrix
    D = _jc_distance_matrix(sequences, labels)
    
    # Run NJ
    newick = _nj_clean(D, n)
    
    return newick
# EVOLVE-BLOCK-END
