"""Construct a 4-regular graph of girth 10 (the (4,10)-cage)."""
import numpy as np
from collections import deque
import time
import itertools

# EVOLVE-BLOCK-START
def construct(k, g):
    """Construct a k-regular graph of girth at least g.
    
    For k=4, g=10, attempts to find the 320-vertex cage via search,
    falling back to a general construction if not found in time.
    """
    if k == 4 and g == 10:
        return _construct_4_10_cage()
    else:
        return _construct_general(k, g)

def _construct_4_10_cage():
    """Construct the 320-vertex (4,10)-cage."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 40s margin for safety
    
    # Structure: 16 x 20 grid
    # Neighbors:
    # (i, j) -> (i+1, j + f1(i)), (i-1, j + f2(i)), (i, j+1), (i, j-1)
    # f1, f2 are linear: a*i+b, c*i+d
    
    n = 320
    rows, cols = 16, 20
    
    # Precompute neighbor arrays for fast checking
    # For each (a, c, b, d), we can build the full adjacency quickly
    
    units = [1, 3, 7, 9, 11, 13, 17, 19]
    offsets = range(20)
    
    trials = 0
    max_trials = 10000
    
    # Precompute i, j arrays
    i_arr = np.repeat(np.arange(rows), cols)
    j_arr = np.tile(np.arange(cols), rows)
    
    for a, c in itertools.product(units, repeat=2):
        for b, d in itertools.product(offsets, repeat=2):
            if time.time() > deadline:
                break
            trials += 1
            
            # Build neighbor arrays
            # n1: (i+1 mod 16, j + (a*i+b) mod 20)
            n1_i = (i_arr + 1) % rows
            n1_j = (j_arr + (a * i_arr + b) % cols) % cols
            n1 = n1_i * cols + n1_j
            
            # n2: (i-1 mod 16, j + (c*i+d) mod 20)
            n2_i = (i_arr - 1) % rows
            n2_j = (j_arr + (c * i_arr + d) % cols) % cols
            n2 = n2_i * cols + n2_j
            
            # n3: (i, j+1)
            n3 = i_arr * cols + (j_arr + 1) % cols
            
            # n4: (i, j-1)
            n4 = i_arr * cols + (j_arr - 1) % cols
            
            # Quick check: no self-loops, no multi-edges
            # Self-loop: any neighbor == vertex
            verts = np.arange(n)
            if np.any(n1 == verts) or np.any(n2 == verts) or np.any(n3 == verts) or np.any(n4 == verts):
                continue
            
            # Multi-edge: any pair of neighbors are the same
            if np.any(n1 == n2) or np.any(n1 == n3) or np.any(n1 == n4) or \
               np.any(n2 == n3) or np.any(n2 == n4) or np.any(n3 == n4):
                continue
            
            # Build adjacency list (4 neighbors per vertex)
            adj = np.stack([n1, n2, n3, n4], axis=1)  # shape (n, 4)
            
            # Check girth using non-backtracking BFS (vectorized)
            if _check_girth_fast(n, adj, 10):
                # Construct edge list
                edges = []
                edge_set = set()
                for v in range(n):
                    for nb in adj[v]:
                        if v < nb:
                            edge_set.add((v, int(nb)))
                edges = sorted(edge_set)
                return n, edges
                
        if time.time() > deadline:
            break
            
    # Fallback: LW construction
    return _construct_4_10_cage_lw()

def _check_girth_fast(n, adj, min_girth):
    """Check if graph has girth at least min_girth using non-backtracking walks.
    
    Uses a vectorized BFS approach: maintain a set of (current, prev) states
    and expand level by level. If any walk returns to its start vertex
    at distance < min_girth, return False.
    """
    # We need to check from every vertex. For efficiency, we'll do BFS from each
    # but use numpy for the frontier expansion.
    
    # State: (node, prev) -> distance
    # We'll use a dict for the visited states, but process frontiers in batches.
    
    for start in range(n):
        # Initial frontier: (start, nb) at distance 1
        # We track (node, prev) pairs
        visited = {}  # (node, prev) -> dist
        frontier = {}  # (node, prev) -> dist
        
        for nb in adj[start]:
            state = (start, int(nb))
            if state not in visited:
                visited[state] = 1
                frontier[state] = 1
        
        dist = 1
        while frontier and dist < min_girth - 1:
            next_frontier = {}
            for (node, prev), d in frontier.items():
                for nb in adj[node]:
                    nb = int(nb)
                    if nb == prev:
                        continue
                    new_state = (nb, node)
                    if new_state in visited:
                        continue
                    if nb == start:
                        # Found a cycle of length d + 1
                        if d + 1 < min_girth:
                            return False
                        # If d+1 >= min_girth, it's okay, but we don't need to continue
                        # Actually, we should stop tracking this path
                    else:
                        visited[new_state] = d + 1
                        next_frontier[new_state] = d + 1
            frontier = next_frontier
            dist += 1
        
        # If we exhausted the frontier without finding a short cycle, continue
        # to next start vertex
    
    return True

def _construct_4_10_cage_lw():
    """Fallback: Lazebnik-Ustimenko-Woldar construction for (4,10)."""
    q = 4
    d = 5
    
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add = lambda x, y: x ^ y
    sub = lambda x, y: x ^ y
    mul = lambda x, y: table[x][y]
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges

def _construct_general(k, g):
    """General construction for other parameters."""
    n = 2 * k
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j - i <= k // 2 or n - (j - i) <= k // 2:
                edges.append((i, j))
    return n, edges
# EVOLVE-BLOCK-END
