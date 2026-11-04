import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0

    if k == 4 and g == 10:
        # Try to find a circulant graph C_n({s1, s2}) with girth >= 10
        n_target = 320
        result = _search_circulant_optimized(n_target, k, g, deadline)
        if result is not None:
            return result
        
        # Fallback to LUW construction (n=8192)
        return _luw_construct(k, g)

    # General fallback
    return _luw_construct(k, g)

def _search_circulant_optimized(n, k, g, deadline):
    """
    Systematically search for a circulant graph C_n({s1, s2}) with girth >= g.
    Uses a fast vectorized girth check.
    """
    # For k=4, steps are s1, s2.
    # We need s1 != s2, s1 != n-s2, 2*s1 != 0, 2*s2 != 0.
    # We can assume 1 <= s1 < s2 <= n//2.
    # Also s1 != n//2 and s2 != n//2.
    
    max_s = n // 2
    found = False
    best_edges = None
    
    # Precompute direction offsets
    # dirs: 0:+s1, 1:-s1, 2:+s2, 3:-s2
    
    # Iterate over possible s1
    for s1 in range(1, max_s + 1):
        if 2 * s1 % n == 0:
            continue
        for s2 in range(s1 + 1, max_s + 1):
            if 2 * s2 % n == 0:
                continue
            if s1 + s2 == n: # s1 = -s2 mod n
                continue
            
            # Check girth
            if _check_girth_circulant_fast(n, s1, s2, g):
                # Found a valid circulant
                edges = _get_circulant_edges(n, [s1, s2])
                return n, edges
                
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
            
    return None

def _check_girth_circulant_fast(n, s1, s2, g):
    """
    Fast check if C_n({s1, s2}) has girth >= g.
    Uses non-backtracking walks.
    """
    # Directions: 0:+s1, 1:-s1, 2:+s2, 3:-s2
    # Backtracking pairs: (0,1), (1,0), (2,3), (3,2)
    
    # visited[pos][dir] = min depth
    # We only need to track up to depth g-1
    # If we reach pos 0 at depth d < g, we have a cycle of length d.
    
    # Use a flat array for speed
    # visited is n x 4
    visited = np.full((n, 4), -1, dtype=np.int32)
    
    # Queue: (pos, dir, depth)
    # Start from 0, no previous dir
    # Initial moves from 0
    
    # Precompute offsets
    offsets = np.array([s1, -s1, s2, -s2], dtype=np.int32)
    
    queue = []
    
    # Initial moves
    for d in range(4):
        np_ = (0 + offsets[d]) % n
        visited[np_, d] = 1
        queue.append((np_, d, 1))
        
    qi = 0
    max_depth = g - 1
    
    # Backtracking check: opposite directions
    # 0<->1, 2<->3
    # opp = [1, 0, 3, 2]
    opp = np.array([1, 0, 3, 2], dtype=np.int32)
    
    while qi < len(queue):
        pos, prev_dir, depth = queue[qi]
        qi += 1
        
        if depth >= max_depth:
            continue
            
        # Try all 4 directions
        for next_dir in range(4):
            # Check backtracking
            if next_dir == opp[prev_dir]:
                continue
                
            # Compute new pos
            np_ = (pos + offsets[next_dir]) % n
            
            new_depth = depth + 1
            
            # Check if we closed a cycle
            if np_ == 0:
                if new_depth < g:
                    return False
                else:
                    continue # No need to enqueue if depth >= g
                    
            # Check visited
            if visited[np_, next_dir] != -1:
                if visited[np_, next_dir] <= new_depth:
                    continue
            else:
                visited[np_, next_dir] = new_depth
                queue.append((np_, next_dir, new_depth))
                
    return True

def _get_circulant_edges(n, steps):
    edges = []
    for i in range(n):
        for s in steps:
            j = (i + s) % n
            if i < j:
                edges.append((i, j))
    return edges

def _luw_construct(k, g):
    """
    Lazebnik-Ustimenko-Woldar construction for k-regular girth-g graphs.
    """
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
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
            yield (1 - side, tuple(y))
    
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
    
    return len(order), edges
