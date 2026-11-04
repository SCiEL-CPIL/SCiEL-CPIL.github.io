import time
import random
import numpy as np

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin before 120s kill

    if k == 4 and g == 10:
        # Try known circulant candidates first (fast check)
        candidates = [
            (320, [1, 79]),
            (320, [1, 81]),
            (320, [1, 77]),
            (320, [1, 83]),
            (320, [1, 101]),
            (320, [1, 119]),
            (320, [1, 39]),
            (320, [1, 41]),
            (320, [1, 59]),
            (320, [1, 61]),
            (320, [1, 99]),
            (320, [1, 121]),
            (320, [1, 139]),
            (320, [1, 141]),
            (320, [1, 159]),
            (320, [1, 161]),
        ]
        
        for n, steps in candidates:
            if time.time() > deadline:
                break
            if _check_circulant_girth_fast(n, steps, k, g):
                return n, _get_circulant_edges(n, steps)
        
        # Systematic search over more circulant candidates
        if time.time() < deadline - 5.0:
            result = _search_circulants(4, 10, 320, deadline)
            if result is not None:
                return result
        
        # Fallback: corrected LUW construction with d=6 for girth >= 10
        return _luw_construct(k, g)

    # General fallback: LUW construction
    return _luw_construct(k, g)

def _check_circulant_girth_fast(n, steps, k, g):
    """
    Fast girth check for circulant graph C_n(steps) using non-backtracking BFS.
    Uses NumPy for speed where possible, but the core is a BFS with state (pos, prev_step).
    """
    if len(steps) * 2 != k:
        return False
    
    # For a circulant with steps [s1, s2], we have 4 directions:
    # +s1, -s1, +s2, -s2
    # State: (position, last_direction) where last_direction in {0,1,2,3}
    # Direction 0: +s1, 1: -s1, 2: +s2, 3: -s2
    # Backtracking: dir 0 <-> 1, dir 2 <-> 3
    
    s1, s2 = steps[0], steps[1]
    
    # Precompute transitions
    # For each direction d, next position from pos
    # And which directions are forbidden (backtracking)
    
    # We'll do BFS from (0, None) up to depth g-1
    # Track visited: (pos, dir) -> earliest depth reached
    
    # Use a 2D array: visited[pos][dir] = min depth reached, or -1 if unvisited
    # But we need to handle the "None" start separately
    
    max_depth = g - 1  # We check cycles of length < g, so up to depth g-1
    
    # visited[pos][dir] stores the depth at which we first reached this state
    # Initialize to -1
    visited = np.full((n, 4), -1, dtype=np.int32)
    
    # BFS queue: list of (pos, dir, depth)
    # Start: from pos 0, no previous direction, depth 0
    # We can go in any of the 4 directions
    
    queue = []
    for d in range(4):
        if d == 0:
            new_pos = s1 % n
        elif d == 1:
            new_pos = (-s1) % n
        elif d == 2:
            new_pos = s2 % n
        elif d == 3:
            new_pos = (-s2) % n
        queue.append((new_pos, d, 1))
        visited[new_pos][d] = 1
    
    # Process BFS
    qi = 0
    while qi < len(queue):
        pos, prev_dir, depth = queue[qi]
        qi += 1
        
        if depth >= max_depth:
            continue
        
        # For each possible next direction, check backtracking
        for next_dir in range(4):
            # Check backtracking: next_dir is opposite of prev_dir
            if (prev_dir == 0 and next_dir == 1) or (prev_dir == 1 and next_dir == 0) or \
               (prev_dir == 2 and next_dir == 3) or (prev_dir == 3 and next_dir == 2):
                continue
            
            # Compute new position
            if next_dir == 0:
                new_pos = (pos + s1) % n
            elif next_dir == 1:
                new_pos = (pos - s1) % n
            elif next_dir == 2:
                new_pos = (pos + s2) % n
            elif next_dir == 3:
                new_pos = (pos - s2) % n
            
            new_depth = depth + 1
            
            # Check if we've reached position 0 -> cycle found
            if new_pos == 0:
                if new_depth < g:
                    return False
                # Otherwise, no need to enqueue
                continue
            
            # Check if we've seen this state at a shorter or equal depth
            if visited[new_pos][next_dir] != -1:
                if visited[new_pos][next_dir] <= new_depth:
                    continue
            else:
                visited[new_pos][next_dir] = new_depth
            
            queue.append((new_pos, next_dir, new_depth))
    
    return True

def _search_circulants(k, g, n_target, deadline):
    """Search for valid circulant graphs with n_target vertices."""
    # For 4-regular: steps = [s1, s2] with s1 != s2, s1, s2 in 1..n/2-1
    # Also need s1 != n - s2 (to avoid duplicate edges) and s2 != n - s1
    # And s1, s2 should be coprime to n for the graph to be connected? Not necessarily.
    
    n = n_target
    # Try s1 = 1 (common in circulants)
    # s2 ranges over 2..n//2-1, with s2 != n-1 (which is 1 mod n), s2 != n-2 (which is 2)
    # Actually s2 should be in 2..n//2, and s2 != n - s2 (i.e., s2 != n/2)
    
    # Let's be systematic but bounded
    s1 = 1
    candidates = []
    for s2 in range(2, n // 2 + 1):
        if s2 == n - s2:  # s2 == n/2, skip
            continue
        if s2 == n - 1:  # same as s1=1
            continue
        candidates.append((s1, s2))
    
    # Shuffle to try diverse candidates
    random.shuffle(candidates)
    
    # Try a limited number
    max_tries = 200
    for i, (a, b) in enumerate(candidates[:max_tries]):
        if time.time() > deadline:
            break
        if _check_circulant_girth_fast(n, [a, b], k, g):
            return n, _get_circulant_edges(n, [a, b])
    
    return None

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
    For k=4, g=10: use d=6 (girth >= d+4 = 10)
    """
    q = k
    # For girth g, we need d such that d + 4 >= g (for even d) or d + 5 >= g (for odd d)
    # Actually the LUW graph CD(d, q) has girth at least d + 5 for odd d, d + 4 for even d
    # For g=10: d=6 gives girth >= 10, d=5 gives girth >= 10 (5+5=10)
    # Let's use d = g - 5 if (g-5) is odd, else g - 4
    # g=10: g-5=5 (odd), so d=5 -> girth >= 5+5=10
    # Wait, let me re-check: the standard result is girth >= d+5 for odd d, d+4 for even d
    # So for g=10: d=5 (odd) -> girth >= 10. Good.
    # But the original code had d = max(3, g-5 if (g-5)%2 else g-4)
    # For g=10: g-5=5, 5%2=1 (odd), so d=5. That gives girth >= 5+5=10.
    
    # Actually, I need to double-check the LUW girth formula.
    # The LUW graph has girth at least d + 5 for odd d.
    # So d=5 -> girth >= 10. 
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
    
    # Build the rule
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
