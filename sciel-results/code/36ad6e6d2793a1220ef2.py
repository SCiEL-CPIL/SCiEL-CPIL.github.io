import time
import random
import numpy as np

_START = time.time()
_DEADLINE = _START + 110.0  # safety margin before 120s limit

def _check_c4(adj, c):
    """Return True if colour class c contains a C4 (two vertices with >=2 common neighbours)."""
    A = adj[c]
    # For each pair of rows, check if dot product >= 2
    # Vectorized: compute all pairwise dot products via matrix multiply
    # A is n x n boolean/integer matrix. A @ A.T gives counts of common neighbours.
    # Diagonal entries are degrees, off-diagonal are common neighbour counts.
    common = A @ A.T
    # Mask out diagonal
    n = common.shape[0]
    np.fill_diagonal(common, 0)
    return bool(np.any(common >= 2))

def _is_valid(adj, n):
    for c in range(5):
        if _check_c4(adj, c):
            return False
    return True

def _construct_greedy(n, seed_val, max_iter=20000):
    """
    Greedy randomized construction: assign colours to edges in lexicographic order.
    For each edge, try random colour order; accept first colour that doesn't create C4.
    If stuck, restart with new seed.
    """
    random.seed(seed_val)
    adj = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]
    
    # Precompute list of edges in lexicographic order
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    for edge_idx, (i, j) in enumerate(edges):
        # Check time
        if time.time() > _DEADLINE - 5.0:
            return None
        
        # Try colours in random order
        colours = list(range(5))
        random.shuffle(colours)
        
        placed = False
        for c in colours:
            # Tentatively assign
            adj[c][i, j] = 1
            adj[c][j, i] = 1
            
            # Check if this creates a C4 in colour c
            # A C4 is created if there exist two vertices u, v (u != v, u != i, v != j)
            # such that both are neighbours of i and j in colour c, OR
            # more generally, the new edge (i,j) combined with existing edges forms a C4.
            # Efficient check: for colour c, check if any pair of vertices has >=2 common neighbours.
            # But doing full check every edge is expensive.
            # Instead: the new edge (i,j) can only create a C4 if there exist vertices u, v
            # such that {u,v} are both adjacent to i and both adjacent to j in colour c,
            # AND u != j, v != i, u != v.
            # Actually, a C4 containing edge (i,j) requires two other vertices u, v
            # such that (i,u), (j,u), (i,v), (j,v) are all in colour c.
            # So: common neighbours of i and j in colour c (excluding each other) must be >= 2.
            
            # Compute common neighbours of i and j in colour c
            # adj[c][i] is the row for i, adj[c][j] is the row for j
            common_mask = adj[c][i] & adj[c][j]
            # Exclude i and j themselves (they shouldn't be neighbours of themselves)
            common_mask[i] = 0
            common_mask[j] = 0
            cnt = np.count_nonzero(common_mask)
            
            if cnt >= 2:
                # This colour creates a C4; revert
                adj[c][i, j] = 0
                adj[c][j, i] = 0
            else:
                # Accept this colour
                placed = True
                break
        
        if not placed:
            # Stuck: no colour works for this edge
            return None
    
    return [int(adj[c][i, j]) for c in range(5) for i in range(n) for j in range(i + 1, n)]

def construct():
    best_n = 11
    best_colors = [min(j - i, 11 - j + i) - 1 for i in range(11) for j in range(i + 1, 11)]
    
    # Try to build valid colorings for increasing n
    # Start from n=12 and go up
    for n in range(12, 29):
        if time.time() > _DEADLINE - 10.0:
            break
        
        # Multiple random restarts for each n
        attempts = 0
        max_attempts = 50
        while time.time() < _DEADLINE - 10.0 and attempts < max_attempts:
            seed_val = attempts * 7919 + n * 31
            result = _construct_greedy(n, seed_val)
            if result is not None:
                # Verify validity
                adj = [np.zeros((n, n), dtype=np.int32) for _ in range(5)]
                idx = 0
                for c in range(5):
                    for i in range(n):
                        for j in range(i + 1, n):
                            adj[c][i, j] = result[idx]
                            adj[c][j, i] = result[idx]
                            idx += 1
                if _is_valid(adj, n):
                    best_n = n
                    best_colors = result
                    break
            attempts += 1
    
    return {'n': best_n, 'colors': best_colors}
