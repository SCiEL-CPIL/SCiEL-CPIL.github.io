"""Construct 5-regular graph with girth 9 using random regular graph generation."""
import random
import time
import numpy as np
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # Moore bound for odd girth
    moore = 1
    for i in range(1, (g - 1) // 2 + 1):
        moore += k * (k - 1) ** i
    # For g=9, k=5: 1 + 5 + 20 + 80 + 320 = 426
    
    # Try increasing n until we find a valid graph
    # Start near the known smallest (1116) and go down
    # But we need to be fast, so start with a reasonable n
    
    # For k=5, g=9, try n values
    # Random regular graphs with n ~ 1000+ can have girth >= 9
    
    def has_short_cycle(n, adj, max_len):
        """Check if any cycle of length < max_len exists.
        First check for cycles of length 3 and 4 using adjacency matrix.
        Then use BFS for longer cycles.
        """
        # Build adjacency matrix for fast cycle detection
        A = np.zeros((n, n), dtype=np.int8)
        for u in range(n):
            for v in adj[u]:
                A[u, v] = 1
        
        # Check for triangles (3-cycles): trace(A^3) / 3
        A2 = A @ A
        # Diagonal of A2 gives degree (should be k)
        # For 3-cycles: (A^3)[i,i] = number of 2-walks from i back to i via 2 steps
        # A 3-cycle through vertex i means: i->j->k->i, which contributes to (A^3)[i,i]
        A3 = A2 @ A
        if np.any(np.diag(A3) > 0):
            return True
        
        # Check for 4-cycles: (A^2)[i,j] > 1 means multiple common neighbors -> 4-cycle
        # Actually, a 4-cycle i-j-k-l-i exists iff there are vertices i,k with >= 2 common neighbors
        # (A^2)[i,k] = number of common neighbors of i and k
        # If (A^2)[i,k] >= 2 and i != k and A[i,k] == 0, we have a 4-cycle
        # More precisely: count 4-cycles via (A^2)
        A2_int = A2.astype(np.int32)
        # For each pair (i,k) with i<k, if A2_int[i,k] >= 2, there's a 4-cycle
        # This is O(n^2) which for n=1000 is 1M operations, fast with numpy
        np.fill_diagonal(A2_int, 0)
        if np.any(A2_int >= 2):
            return True
        
        # For cycles of length 5 through 8, use BFS from each vertex
        # But we can early-exit: if no 3 or 4 cycles, check 5-8 cycles
        # BFS with non-backtracking
        for start in range(n):
            # State: (node, prev, depth)
            # We track visited states to avoid reprocessing
            # Use a set of (node, prev) pairs visited at depth <= max_len-2
            visited_states = set()
            queue = deque([(start, -1, 0)])
            while queue:
                node, prev, depth = queue.popleft()
                if depth >= max_len - 1:
                    continue
                for nb in adj[node]:
                    if nb == prev:
                        continue
                    if nb == start and depth + 1 >= 3:
                        return True
                    state = (nb, node)
                    if state in visited_states:
                        continue
                    visited_states.add(state)
                    queue.append((nb, node, depth + 1))
        return False
    
    def generate_random_regular(n, degree):
        """Generate a random degree-regular graph using configuration model with rejection."""
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            attempts += 1
            # Create stubs
            stubs = []
            for v in range(n):
                for _ in range(degree):
                    stubs.append(v)
            random.shuffle(stubs)
            # Pair up stubs
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i + 1]
                if u == v:
                    valid = False
                    break
                if v in adj[u]:
                    valid = False
                    break
                adj[u].append(v)
                adj[v].append(u)
            if valid:
                return adj
        return None
    
    # Try different n values
    # For k=5, g=9, the graph needs to be large enough
    # Start with n around 1000 and increase if needed
    # Moore bound is 426, known best is 1116
    # Try from smaller to larger, with more attempts at smaller n
    
    # First, try a range of n values starting from a reasonable minimum
    # For 5-regular, n must be even (5*n even -> n even)
    # Try n from 1000 to 1500 in steps of 10, with more attempts at smaller n
    best_n = None
    best_edges = None
    
    # Try smaller n first to minimize
    for n in range(1000, 1500, 10):
        if time.time() > deadline - 15:
            break
        
        # Ensure n is compatible with k-regular (n*k must be even)
        if (n * k) % 2 != 0:
            continue
        
        # More attempts for smaller n
        max_attempts_for_n = max(5, 30 - (n - 1000) // 10)
        
        for attempt in range(max_attempts_for_n):
            if time.time() > deadline - 15:
                break
            
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
            
            # Verify girth >= g
            if not has_short_cycle(n, adj, g):
                # Build edge list
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # If we didn't find one in the range, try larger n
    for n in range(1500, 2000, 20):
        if time.time() > deadline - 15:
            break
        
        if (n * k) % 2 != 0:
            continue
        
        for attempt in range(10):
            if time.time() > deadline - 15:
                break
            
            adj = generate_random_regular(n, k)
            if adj is None:
                continue
            
            if not has_short_cycle(n, adj, g):
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
    
    # Fallback: use a known construction or return a simple valid graph
    # Use the LUW construction as fallback
    q = k
    d = max(3, g - 5 if (g - 5) % 2 else g - 4)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    else:
        add, sub, mul = (lambda x, y: (x + y) % q), (lambda x, y: (x - y) % q), (lambda x, y: x * y % q)
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
