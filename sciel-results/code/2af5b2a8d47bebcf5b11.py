import time
import sys
import random
from collections import deque
from math import gcd

_DEADLINE = 115.0

def construct(k, g):
    t_start = time.time()
    
    # Fallback for general k, g
    if k != 4 or g != 10:
        return _luw(k, g)
    
    # Specific search for (4, 10) cage
    # Moore bound for (4, 10) is 246.
    # Best known is 320 (LUW).
    # We search for a circulant graph C(n, {s1, s2})
    
    # Quick circulant search
    # We iterate n from Moore bound up to 320
    # If we find one < 320, it's an improvement.
    
    for n in range(246, 321):
        if time.time() - t_start > _DEADLINE - 5:
            break
        
        max_step = n // 2
        for s1 in range(1, max_step + 1):
            if time.time() - t_start > _DEADLINE - 5:
                break
            
            # Optimization: s1 and s2 must generate the group Z_n with n
            # i.e., gcd(n, s1, s2) == 1
            # Also s1 != s2, s1 != -s2 (mod n)
            # Since s2 > s1, s2 != s1.
            # s2 != n - s1 => s1 + s2 != n
            
            for s2 in range(s1 + 1, max_step + 1):
                # Check simplicity:
                # No loops: 2*s1 != 0 mod n, 2*s2 != 0 mod n
                # No multi-edges: s1 != s2, s1 != -s2 (i.e., s1+s2 != n)
                if (2 * s1) % n == 0 or (2 * s2) % n == 0:
                    continue
                if s1 + s2 == n:
                    continue
                
                # Check connectivity: gcd(n, s1, s2) == 1
                if gcd(n, gcd(s1, s2)) != 1:
                    continue
                
                # Check girth
                if _check_girth_circulant_fast(n, s1, s2, g):
                    edges = []
                    for u in range(n):
                        v1 = (u + s1) % n
                        v2 = (u + s2) % n
                        if u < v1:
                            edges.append((u, v1))
                        if u < v2:
                            edges.append((u, v2))
                    return n, edges
                    
    # If no circulant found < 320, return LUW (320)
    return _luw(k, g)

def _check_girth_circulant_fast(n, s1, s2, g):
    """
    Check if circulant graph C(n, {s1, s2}) has girth at least g.
    We check for non-backtracking closed walks of length L < g starting at 0.
    """
    # We use BFS on states (current_vertex, previous_vertex)
    # Depth 0: (0, None) - but we can't start with None in the step logic easily.
    # Let's initialize depth 1 states: neighbors of 0.
    # Neighbors of 0: s1, -s1, s2, -s2 (mod n)
    
    # If any neighbor is 0, girth is 2 (loop). But we checked no loops.
    # If any neighbor is repeated, multi-edge. Checked.
    
    # Target: find if any walk of length L in [3, g-1] returns to 0.
    
    # State: (v, prev)
    # visited set of states to avoid re-visiting same state at same or shorter depth?
    # Actually, if we visit (v, prev) at depth d, any future visit at depth d' >= d
    # will lead to cycles of length >= d' + dist(prev->0)...
    # Simpler: Just BFS up to depth g-1.
    
    # Since g is small (10), we can just track depth.
    # Max depth 9.
    # States are (v, prev). v in 0..n-1, prev in 0..n-1.
    # Total states n*n. For n=320, 102400 states.
    # We only traverse a subset.
    
    # Use a dictionary or set for visited states to avoid exponential blowup if cycles exist?
    # No, we want to detect cycles.
    # If we reach (0, prev) at depth L, we have a cycle of length L.
    
    # Let's use a queue of (v, prev, depth)
    # To be efficient, we can stop expanding if depth >= g.
    
    q = deque()
    
    # Initialize depth 1
    neighbors = [(s1 % n), ((-s1) % n), (s2 % n), ((-s2) % n)]
    # Remove duplicates if any (shouldn't be any if simple)
    # But if s1 == -s1 mod n, it's a loop, handled.
    
    for v in neighbors:
        # If v == 0, loop of length 1? No, v is neighbor.
        # If v == 0, then edge (0,0) exists. Handled by simplicity check.
        # We start from 0 to v.
        # State: (v, 0)
        q.append((v, 0, 1))
        
    # We need to check if we hit 0 at depth L >= 3.
    # Note: We don't check depth 2 because that would be backtracking 0->v->0, which is not a simple cycle in a simple graph?
    # Actually, 0->v->0 is a walk of length 2. A cycle of length 2 requires a multi-edge.
    # Since graph is simple, no cycle of length 2.
    # So we only care about L >= 3.
    
    # To avoid checking the same state multiple times at different depths (which is fine for BFS, but we can prune):
    # If we visit (v, prev) at depth d, we don't need to visit it again at depth d' > d 
    # IF we are only looking for the SHORTEST cycle.
    # But we just need to know if ANY cycle < g exists.
    # If we find one, return False.
    # If we exhaust BFS up to depth g-1 without hitting 0, return True.
    
    # We can use a set of visited states (v, prev) to prune.
    # If we reach (v, prev) at depth d, and later at depth d' > d, the cycle formed would be longer.
    # So pruning is safe for finding the *shortest* cycle.
    
    visited = set()
    
    while q:
        v, prev, d = q.popleft()
        
        # If we are at 0, and d >= 3, we found a cycle of length d.
        if v == 0 and d >= 3:
            return False
            
        if d >= g - 1:
            continue
            
        # Generate neighbors of v, excluding prev
        # Neighbors of v: (v+s1)%n, (v-s1)%n, (v+s2)%n, (v-s2)%n
        # Exclude prev
        
        for step in (s1, -s1, s2, -s2):
            w = (v + step) % n
            if w == prev:
                continue
                
            # Check if we've seen this state (w, v) at a smaller or equal depth
            # We only need to check if (w, v) is in visited.
            # If it is, we don't need to add it again.
            if (w, v) in visited:
                continue
                
            visited.add((w, v))
            q.append((w, v, d + 1))
            
    return True

def _luw(k, g):
    """Lazebnik-Ustimenko-Woldar construction CD(d, q)."""
    q = k
    if (g - 5) % 2 == 0:
        d = g - 5
    else:
        d = g - 4
    d = max(3, d)

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
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

    start = (0, tuple([None] + [0] * d))

    def neighbours(vertex):
        side, x = vertex
        x = list(x)
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)

    index = {start: 0}
    order = [start]
    edges = []

    for vi, vertex in enumerate(order):
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))

    return len(order), edges
