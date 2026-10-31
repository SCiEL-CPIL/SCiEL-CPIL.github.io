import time
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 115
    
    if k == 3 and g == 16:
        # Search for circulant graphs
        # n must be even for bipartite? No, but girth 16 is even.
        # Circulants C_n(1, a, b)
        
        # Lower bound from prompt: 512
        # Upper bound: 936
        for n in range(512, 937, 2):
            if time.time() > deadline - 10:
                break
            
            # Check candidates for this n
            # a, b such that 1 < a < b < n/2
            # Heuristic: a, b should be "large" to avoid small cycles
            # But we must check all.
            
            # To speed up, we can check n in increasing order.
            # If we find one, we return it.
            
            found = _search_circulant(n, g, deadline)
            if found:
                return found
        
        # If no circulant found, try LUW with small d
        # But LUW is huge.
        # Maybe try a random search?
        
        # Fallback: Return a known valid graph if we have one, or construct a large one.
        # If we didn't find any, we might have to return a valid but large graph.
        # The prompt says "n must lie between Moore bound and max_vertices".
        # If we don't find a small one, we should return the best we found or a valid one.
        
        # Let's assume we found 936 in the loop.
        # If the loop finishes without finding, we need a fallback.
        
    return _luw_construct(k, g)

def _search_circulant(n, g, deadline):
    # G = {1, -1, a, -a, b, -b}
    # We want no cycle < g.
    
    half = n // 2
    # a from 2 to half-1
    # b from a+1 to half-1
    
    # Optimization: Precompute -1, 1, etc.
    
    for a in range(2, half):
        if time.time() > deadline:
            return None
        
        # Quick check: 2a != 1, n-1, a, n-a
        # 2a % n not in {1, n-1, a, n-a}
        two_a = 2 * a % n
        if two_a in (1, n-1, a, n-a):
            continue
            
        for b in range(a+1, half):
            if time.time() > deadline:
                return None
            
            # Quick 4-cycle check
            if _has_4cycle_fast(n, a, b):
                continue
            
            # Quick 6-cycle check
            if _has_6cycle_fast(n, a, b):
                continue
            
            # Full girth check
            if _check_girth_fast(n, a, b, g):
                edges = []
                for i in range(n):
                    for shift in (1, a, b):
                        j = (i + shift) % n
                        if i < j:
                            edges.append((i, j))
                return n, edges
                
    return None

def _has_4cycle_fast(n, a, b):
    # Check sums of 4 elements from {1, -1, a, -a, b, -b}
    # that sum to 0 mod n, with no backtracking.
    # Backtracking: step i and i+1 are not inverses.
    
    G = [1, n-1, a, n-a, b, n-b]
    
    # Instead of 4 nested loops, we can check pairs.
    # A 4-cycle is x1 + x2 + x3 + x4 = 0.
    # x1 + x2 = -(x3 + x4).
    # Let S2 be the set of sums of 2 non-backtracking steps.
    # If any sum in S2 appears twice with compatible paths, we have a 4-cycle.
    
    sums = {}
    for i in range(6):
        for j in range(6):
            if G[i] == (n - G[j]) % n: # backtrack
                continue
            s = (G[i] + G[j]) % n
            # Store the pair (i, j) or just the sum?
            # If s appears again, we need to check if the path is valid.
            # Path: 0 -> G[i] -> G[i]+G[j] -> G[i]+G[j]+G[k] -> 0
            # We need G[i]+G[j] != 0, G[i] != 0, etc.
            
            if s not in sums:
                sums[s] = []
            sums[s].append((i, j))
            
    # Now check for collisions
    for s, pairs in sums.items():
        if len(pairs) < 2:
            continue
        # Check all pairs of pairs
        for idx1 in range(len(pairs)):
            for idx2 in range(idx1 + 1, len(pairs)):
                i1, j1 = pairs[idx1]
                i2, j2 = pairs[idx2]
                # Path 1: 0 -> G[i1] -> s
                # Path 2: 0 -> G[i2] -> s
                # To form a cycle, we need to go 0 -> G[i1] -> s -> G[i2] -> 0?
                # No, the cycle is 0 -> G[i1] -> s -> (s + G[i2])? No.
                # The cycle is 0 -> v1 -> v2 -> v3 -> 0.
                # v1 = G[i1], v2 = s, v3 = s + G[k]?
                # Actually, if x1+x2 = x3+x4 = s, then x1+x2-x3-x4 = 0.
                # The cycle is 0 -> G[i1] -> s -> s+G[i2] -> 0?
                # Wait, x3+x4 = s implies -x3 - x4 = -s.
                # So G[i1]+G[j1] + G[i2]+G[j2] = 2s? No.
                # We need x1+x2+x3+x4 = 0.
                # If x1+x2 = s and x3+x4 = -s, then sum is 0.
                # So we need to check if s and -s are both in sums.
                
    # Better: Check if any sum s has a pair, and -s has a pair.
    # And the paths don't collide.
    
    # This is getting complicated. The 4 nested loop in the original code was simple.
    # 6^4 = 1296 iterations. Very fast.
    # Let's stick to the simple loop but optimize the inner check.
    
    for i in range(6):
        v1 = G[i]
        if v1 == 0: continue
        for j in range(6):
            if G[j] == (n - v1) % n: continue
            v2 = (v1 + G[j]) % n
            if v2 == 0: continue
            for k in range(6):
                if G[k] == (n - G[j]) % n: continue
                v3 = (v2 + G[k]) % n
                if v3 == 0: continue
                for l in range(6):
                    if G[l] == (n - G[k]) % n: continue
                    if (v3 + G[l]) % n == 0:
                        # Check distinct vertices
                        if v1 != v2 and v2 != v3 and v1 != v3 and v1 != 0 and v2 != 0 and v3 != 0:
                            return True
    return False

def _has_6cycle_fast(n, a, b):
    # 6^6 = 46656. A bit slow for 100k pairs.
    # Use BFS depth 3.
    G = [1, n-1, a, n-a, b, n-b]
    
    dist = [-1] * n
    parent = [-1] * n
    dist[0] = 0
    queue = deque([0])
    
    # BFS to depth 3
    # Actually, we just need to see if any vertex at depth 3 has a neighbor at depth 3
    # that is not its parent.
    
    # This is a standard girth check.
    # We can just run the full check for g=6?
    # No, we want to reject early.
    
    # Let's use the same BFS logic as _check_girth_fast but stop at depth 3.
    
    return _check_girth_fast(n, a, b, 6)

def _check_girth_fast(n, a, b, g):
    # Non-backtracking BFS
    # State: (u, prev)
    # Use a 2D array for visited?
    # visited[u][prev] = depth
    
    # n is up to 936. n*n = 876k.
    # A list of lists of ints is memory heavy.
    # Use a dict? Or a single array of size n*n?
    # Index: u * n + prev
    
    visited = [-1] * (n * n)
    
    G = [1, n-1, a, n-a, b, n-b]
    
    queue = deque()
    # Start from 0, prev = -1 (represented as n)
    start_state = 0 * n + n
    visited[start_state] = 0
    queue.append((0, n))
    
    max_depth = g - 1
    
    while queue:
        u, prev = queue.popleft()
        d = visited[u * n + prev]
        
        if d >= max_depth:
            continue
            
        for step in G:
            v = (u + step) % n
            if v == prev:
                continue
                
            # Check if v == 0 and u != 0 (cycle closed)
            # Actually, if v == 0, we found a cycle of length d+1.
            # If d+1 < g, return False.
            
            if v == 0:
                if d + 1 < g:
                    return False
                continue
            
            state_idx = v * n + u
            if visited[state_idx] == -1:
                visited[state_idx] = d + 1
                queue.append((v, u))
                
    return True

def _luw_construct(k, g):
    # Fallback
    q = k
    d = g - 5
    if d < 3: d = 3
    if d % 2 == 0: d += 1
    
    return _luw_construct_with_d(k, g, d)

def _luw_construct_with_d(k, g, d):
    q = k
    # ... (copy from original) ...
    # This is large, but valid.
    
    # For k=3, d=11, n = 2 * 3^11 = 354294.
    # This will score very low (2808/354294 ~ 0.008).
    # But it's valid.
    
    # We should avoid this if possible.
    
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]] if q == 4 else None
    if q == 4:
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

    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                y[c] = add(x[c], mul(y[a], x[b])) if side == 0 else sub(x[c], mul(x[a], y[b]))
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

    return len(order), edges
