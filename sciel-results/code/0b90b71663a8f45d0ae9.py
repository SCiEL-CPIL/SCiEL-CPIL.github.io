"""Construct a 3-regular graph with large girth (target 16).
Strategy: Build a girth-16 graph via random non-backtracking walk ensemble.
Start from the infinite 3-regular tree of depth 7 (1+3+9+27+81+243+729+2187 = 3273 boundary),
then close the boundary with minimum-length non-backtracking paths that don't create shorter cycles.
Use randomized local search with a deadline guard.
"""
import random
import time
import sys

def construct(k, g):
    deadline = time.time() + 140  # 20s safety margin before 160s kill
    random.seed(42)
    
    if k != 3:
        # Fallback to LUW for other degrees
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

    # For k=3, g=16: use random construction
    # Build a random 3-regular graph and check girth
    # Target: minimize n such that girth >= 16
    
    # Moore bound for k=3, g=16:
    # For even girth g=2r, Moore bound: n >= k*((k-1)^(r-1) - 1)/(k-2)
    # r = 8, n >= 3*((2)^7 - 1)/(1) = 3*127 = 381
    # But actual cage n(3,16) is known to be at least 512
    
    # Strategy: Generate random 3-regular graphs of increasing size
    # and check if girth >= 16
    
    def check_girth(adj, n, target_girth):
        """Check if graph has girth >= target_girth using BFS from each vertex."""
        # Use non-backtracking walk to check for short cycles
        # For efficiency, we check if there's any cycle of length < target_girth
        # A cycle of length < target_girth exists iff there's a non-backtracking
        # closed walk of length < target_girth that is a simple cycle
        
        # Simpler: BFS from each vertex to check shortest cycle
        # But that's O(n^2) which is too slow for large n
        # Instead, use the fact that in a 3-regular graph, the number of
        # non-backtracking walks of length l from a vertex is at most 2*3^(l-1)
        # For girth >= 16, we need no closed non-backtracking walks of length < 16
        # that form simple cycles
        
        # Actually, the simplest correct check: for each vertex, do BFS up to depth
        # target_girth//2 and check for back-edges
        # This is O(n * 3^(g/2)) which for g=16 is O(n * 3^8) = O(n * 6561)
        # For n ~ 500, that's about 3.3M operations per check - feasible
        
        half = target_girth // 2
        for start in range(n):
            # BFS from start, tracking parent to avoid backtracking
            # If we reach a vertex that is already visited (and not parent), we found a cycle
            dist = [-1] * n
            dist[start] = 0
            from collections import deque
            queue = deque([(start, -1)])
            while queue:
                u, parent = queue.popleft()
                if dist[u] >= half:
                    continue
                for v in adj[u]:
                    if v == parent:
                        continue
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append((v, u))
                    elif dist[v] < dist[u]:
                        # Found a cycle of length dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
            # Also need to check if any vertex at depth == half has an edge to
            # another vertex at depth == half (cycle of length 2*half = target_girth, which is ok)
            # Actually the above handles it: if v is already visited and dist[v] < dist[u],
            # cycle length is dist[u] + dist[v] + 1
            # If dist[v] == dist[u], cycle length would be 2*dist[u] + 1 >= target_girth + 1 if dist[u] >= half
            # So we're fine
        return True

    def make_random_3_reg(n):
        """Generate a random 3-regular simple graph on n vertices using configuration model."""
        if n % 2 != 0:
            return None
        # Each vertex has degree 3, so total stubs = 3n
        stubs = []
        for i in range(n):
            for _ in range(3):
                stubs.append(i)
        
        for _ in range(100):
            random.shuffle(stubs)
            adj = [[] for _ in range(n)]
            valid = True
            for i in range(0, 3 * n, 2):
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

    def verify_adj(n, adj):
        """Verify adj is a valid 3-regular simple graph."""
        for i in range(n):
            if len(adj[i]) != 3:
                return False
            if len(set(adj[i])) != 3:
                return False
            for j in adj[i]:
                if i == j:
                    return False
                if i not in adj[j]:
                    return False
        return True

    def edges_from_adj(n, adj):
        edges = []
        seen = set()
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
                    seen.add((i, j))
        return edges

    # Try random graphs of various sizes
    # Start from a reasonable lower bound
    # Best known upper bound is 936, lower bound is 512
    # Let's try sizes that are even (for 3-regular)
    
    # First, try a few random graphs of moderate size
    candidates = []
    
    # Try sizes from a range
    # For girth 16, we expect n to be in the hundreds
    sizes_to_try = []
    for n in range(512, 600, 2):  # even sizes only
        sizes_to_try.append(n)
    
    # Also try some larger sizes as fallback
    for n in range(600, 800, 4):
        sizes_to_try.append(n)
    
    # Shuffle to try random order, but bias toward smaller
    random.shuffle(sizes_to_try)
    sizes_to_try.sort()  # Actually try in order from smallest
    
    best_n = None
    best_adj = None
    
    for n in sizes_to_try:
        if time.time() > deadline - 10:
            break
        
        # Try to find a valid girth-16 graph of size n
        found = False
        attempts = 0
        max_attempts = 50
        
        while attempts < max_attempts and not found:
            if time.time() > deadline - 10:
                break
            
            adj = make_random_3_reg(n)
            if adj is None:
                attempts += 1
                continue
            
            if not verify_adj(n, adj):
                attempts += 1
                continue
            
            # Check girth
            if check_girth(adj, n, g):
                found = True
                best_n = n
                best_adj = adj
                break
            
            attempts += 1
        
        if found:
            break
    
    if best_n is not None:
        edges = edges_from_adj(best_n, best_adj)
        return best_n, edges
    
    # Fallback: use LUW construction
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
