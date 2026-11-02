"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import sys
from collections import defaultdict, deque

def construct(k, g):
    # Moore bound for odd girth g=2r+1
    r = (g - 1) // 2
    moore = 1 + k * sum((k - 1) ** i for i in range(r))
    
    # We search for a circulant graph
    # k=5 implies n must be even, steps {n/2, s1, -s1, s2, -s2}
    rng = random.Random(42)
    
    def check_girth_circulant(n, steps, g):
        # steps is a list of 5 unique steps (including negatives)
        # Actually, for circulant, adjacency is defined by steps S.
        # We check if there is a cycle of length < g.
        # A cycle corresponds to a sequence of steps $a_1, ..., a_l$ such that
        # sum $a_i = 0$ and no prefix sum is 0 (for simple cycle)
        # and non-backtracking: $a_i \neq -a_{i-1}$.
        # For small g, we can just do BFS on the actual graph.
        
        # Build adjacency list for speed? 
        # n is small (< 2000), so we can just simulate.
        
        # BFS from 0
        # State: (current_node, prev_step)
        # We want to reach 0 with length < g.
        # Start: 0, prev_step = None
        # But we can start from any node. Due to symmetry, checking from 0 is sufficient for existence of cycle?
        # No, a cycle might not pass through 0.
        # However, for a Cayley graph (circulant is a Cayley graph), if there is a cycle,
        # there is a cycle based at the identity?
        # Yes, any cycle can be shifted to start at 0.
        # So we only need to check for a non-backtracking walk from 0 to 0 of length < g.
        
        # BFS on (node, last_step)
        # visited: set of (node, last_step)
        # Queue: (node, last_step, depth)
        
        visited = set()
        q = deque()
        q.append((0, None, 0))
        visited.add((0, None))
        
        while q:
            node, last_step, depth = q.popleft()
            if depth >= g - 1:
                continue
            
            for s in steps:
                if last_step is not None and s == -last_step:
                    continue
                new_node = (node + s) % n
                if new_node == 0 and depth + 1 >= 3 and depth + 1 < g:
                    return False # Found short cycle
                if (new_node, s) not in visited:
                    visited.add((new_node, s))
                    q.append((new_node, s, depth + 1))
                    
        return True # No short cycle found

    # Search range
    # Start from moore, go up to moore + 1000
    # Step 2 for n (must be even)
    max_n = moore + 1000
    for n in range(moore, max_n + 1, 2):
        if n < 2 * k:
            continue
        half = n // 2
        # Steps must be distinct
        # s1, s2 in {1, ..., n-1}
        # s1 != 0, s1 != half, s2 != 0, s2 != half
        # s1 != s2, s1 != -s2 (mod n)
        
        # Try random pairs
        for _ in range(200):
            s1 = rng.randint(1, n - 1)
            if s1 == half:
                continue
            s2 = rng.randint(1, n - 1)
            if s2 == half or s2 == s1 or s2 == (n - s1) % n:
                continue
            
            steps = [half, s1, -s1 % n, s2, -s2 % n]
            # Ensure steps are unique
            if len(set(steps)) != 5:
                continue
            
            if check_girth_circulant(n, steps, g):
                # Construct edges
                edges = []
                for u in range(n):
                    for v in steps:
                        v_idx = (u + v) % n
                        if u < v_idx:
                            edges.append((u, v_idx))
                return n, edges
                
    # Fallback: Use the seed's algebraic construction if circulant fails
    # Or just return a large valid graph?
    # Let's try the seed's fallback logic here to be safe
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2)}
    for c in range(5, d + 1):
        r_val, c0 = (c - 5) % 4, c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r_val]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
            yield 1 - side, tuple(y)
    
    start = (0, (None,) + (0,) * d)
    index, order, edges = {start: 0}, [start], []
    for vertex in order:
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    return len(order), edges
# EVOLVE-BLOCK-END
