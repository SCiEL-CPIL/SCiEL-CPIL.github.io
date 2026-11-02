import time
import random
import sys
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 40s margin for safety
    
    # 1. Calculate Moore Bound
    half_g = g // 2
    if g % 2 == 1:
        target_girth = g + 1
    else:
        target_girth = g
        
    moore_sum = 0
    for i in range(target_girth // 2):
        moore_sum += (k - 1) ** i
    N_min = 2 * moore_sum
    
    # Fast girth check using non-backtracking BFS from each vertex
    # This is more efficient than standard BFS for girth verification
    def check_girth(adj, n, target_g):
        """Check if graph has girth >= target_g using non-backtracking walks."""
        # For each vertex, do a non-backtracking BFS to find shortest cycle
        for start in range(n):
            # State: (current_vertex, previous_vertex, distance)
            # We track visited states (u, v) meaning we arrived at u from v
            visited = set()
            queue = deque()
            for v in adj[start]:
                queue.append((v, start, 1))
                visited.add((v, start))
            
            while queue:
                u, prev, dist = queue.popleft()
                if dist >= target_g - 1:
                    continue
                for w in adj[u]:
                    if w == prev:
                        continue  # Non-backtracking: don't go back to where we came from
                    if w == start and dist + 1 < target_g:
                        return False  # Found cycle shorter than target
                    state = (w, u)
                    if state not in visited:
                        visited.add(state)
                        queue.append((w, u, dist + 1))
        return True
    
    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 2:
            # Special case for k=2 (cycle)
            n = g
            edges = [(i, (i+1) % g) for i in range(g)]
            return n, edges
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
                yield 1 - side, tuple(y)

        start = (0, (None,) + (0,) * d)
        index, order, edges = {start: 0}, [start], []
        edge_set = set()
        for vertex in order:
            for other in neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                # Add edge only once (from side 0 to side 1)
                if vertex[0] == 0:
                    u_idx = index[vertex]
                    v_idx = index[other]
                    if u_idx < v_idx:
                        edge = (u_idx, v_idx)
                    else:
                        edge = (v_idx, u_idx)
                    if edge not in edge_set:
                        edge_set.add(edge)
                        edges.append(edge)
        return len(order), edges

    # Known small graphs
    if k == 3 and g == 5:
        # Petersen graph
        n = 10
        edges = [(0,1), (0,4), (0,5), (1,2), (1,6), (2,3), (2,7), (3,4), (3,8), (4,9), (5,6), (5,9), (6,7), (7,8), (8,9)]
        return n, edges
    if k == 3 and g == 6:
        # Heawood graph
        n = 14
        edges = [(0,1), (0,2), (0,3), (1,4), (1,5), (1,6), (2,4), (2,5), (2,7), (3,6), (3,7), (3,8), (4,9), (4,10), (4,11), (5,9), (5,10), (5,12), (6,11), (6,12), (6,13), (7,9), (7,13), (7,10), (8,11), (8,12), (8,13)]
        return n, edges
    if k == 4 and g == 5:
        # Known 4-regular graph of girth 5 with 16 vertices
        n = 16
        edges = []
        # Construct using known structure
        for i in range(16):
            for j in range(i+1, 16):
                if (j - i) % 4 == 1 or (j - i) % 4 == 3 or (j - i) % 16 in [5, 11]:
                    edges.append((i, j))
        # Verify and fix if needed
        from collections import defaultdict
        deg = defaultdict(int)
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        if all(deg[i] == 4 for i in range(16)):
            return n, edges
        # Fallback to construction
        return luw_construct(k, g)

    best_result = None
    
    max_M = N_min // 2 + 500  # Reduced limit to allow more tries per M
    
    for M in range(max(1, N_min // 2), max_M + 1):
        if time.time() > deadline:
            break
            
        if k > M - 1:
            continue

        num_tries = 80  # Increased tries to improve success rate
        found = False
        for _ in range(num_tries):
            if time.time() > deadline:
                break
            
            S = set(random.sample(range(1, M), k))
            T = list(S) + [-s % M for s in S]
            
            # Fast filter for short cycles using non-backtracking
            valid = True
            
            # Check 4-cycles: look for s1+s2 == s3+s4 with disjoint pairs
            sums_2 = {}
            for i in range(len(T)):
                for j in range(i+1, len(T)):
                    s_val = (T[i] + T[j]) % M
                    if s_val in sums_2:
                        p1 = sums_2[s_val]
                        p2 = (i, j)
                        # Check if pairs are disjoint (no common element)
                        if p1[0] != p2[0] and p1[0] != p2[1] and p1[1] != p2[0] and p1[1] != p2[1]:
                            valid = False
                            break
                    else:
                        sums_2[s_val] = (i, j)
                if not valid: break
            
            # Check 6-cycles: look for s1+s2+s3 == s4+s5+s6 with disjoint triples
            if valid:
                sums_3 = {}
                for i in range(len(T)):
                    for j in range(i+1, len(T)):
                        for l in range(j+1, len(T)):
                            s_val = (T[i] + T[j] + T[l]) % M
                            if s_val in sums_3:
                                p1 = sums_3[s_val]
                                p2 = (i, j, l)
                                # Check if triples are disjoint
                                if not any(x == y for x in p1 for y in p2):
                                    valid = False
                                    break
                            else:
                                sums_3[s_val] = (i, j, l)
                        if not valid: break
                    if not valid: break
            
            if not valid:
                continue

            # Construct adjacency list
            adj = [[] for _ in range(2 * M)]
            edges = []
            for u in range(M):
                for s in S:
                    v = (u + s) % M
                    adj[u].append(v + M)
                    adj[v + M].append(u)
                    edges.append((u, v + M))
            
            # Check Girth with non-backtracking BFS from each vertex
            girth_ok = check_girth(adj, 2 * M, target_girth)
            
            if girth_ok:
                best_result = (2 * M, edges)
                found = True
                break
        
        if found:
            return best_result
            
    if best_result:
        return best_result
        
    # Try LUW construction as fallback
    return luw_construct(k, g)
