import random
import time
import sys
from collections import deque

def construct(k, g):
    """
    Constructs a k-regular graph with girth at least g.
    Strategy: Search for cyclic (Z_n-regular) graphs with small n.
    Fallback: Lazebnik-Ustimenko-Woldar (LW) graph.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill

    def check_girth_nb(adj, n, g):
        """
        Check girth using non-backtracking walks from every vertex.
        Returns True if no cycle shorter than g exists.
        """
        max_depth = (g - 1) // 2
        
        for start in range(n):
            for first_neighbor in adj[start]:
                visited = set()
                queue = deque()
                queue.append((first_neighbor, start, 0))
                visited.add((first_neighbor, start))
                
                found_short_cycle = False
                while queue:
                    curr, prev, depth = queue.popleft()
                    if depth >= g - 1:
                        continue
                    for nxt in adj[curr]:
                        if nxt == prev:
                            continue
                        if nxt == start and depth + 1 < g:
                            found_short_cycle = True
                            break
                        if (nxt, curr) not in visited:
                            visited.add((nxt, curr))
                            queue.append((nxt, curr, depth + 1))
                    if found_short_cycle:
                        break
                if found_short_cycle:
                    return False
        
        return True

    def make_cyclic_graph(n, offsets):
        edges = []
        edge_set = set()
        for i in range(n):
            for off in offsets:
                j = (i + off) % n
                if i == j:
                    return None
                key = (min(i, j), max(i, j))
                if key in edge_set:
                    return None
                edge_set.add(key)
                edges.append((i, j))
        
        degree = [0] * n
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
        if any(d != k for d in degree):
            return None
        
        return edges

    def check_girth_from_edges(edges, n, g):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return check_girth_nb(adj, n, g)

    def check_girth_cyclic(n, offsets):
        """
        Check if the cyclic graph with given offsets has girth >= g.
        Uses BFS on (current_sum, last_offset_index) states.
        """
        if len(offsets) != k:
            return False
        
        if any(o % n == 0 for o in offsets):
            return False
        for i in range(k):
            for j in range(i+1, k):
                if offsets[i] % n == offsets[j] % n:
                    return False
                if (offsets[i] + offsets[j]) % n == 0:
                    return False
        
        visited = set()
        queue = deque()
        
        for i in range(k):
            s = offsets[i] % n
            state = (s, i)
            visited.add(state)
            queue.append((s, i, 1))
        
        while queue:
            s, last_idx, depth = queue.popleft()
            if depth >= g:
                continue
            for j in range(k):
                if j == last_idx:
                    continue
                if (offsets[j] + offsets[last_idx]) % n == 0:
                    continue
                new_s = (s + offsets[j]) % n
                new_state = (new_s, j)
                if new_state in visited:
                    continue
                if new_s == 0:
                    if depth + 1 < g:
                        return False
                    visited.add(new_state)
                    continue
                visited.add(new_state)
                queue.append((new_s, j, depth + 1))
        
        return True

    best_n = float('inf')
    best_edges = None

    # Search range: The Moore bound for (4,10) is 240 (or 243 as stated).
    # Known smallest is 320. We search 243..320.
    min_n = 243
    max_n_to_try = 320

    for n in range(min_n, max_n_to_try + 1):
        if time.time() > deadline - 10:
            break
        
        # Heuristic: Try offsets that are multiples of small integers or random
        # We want to avoid short cycles. Large offsets are often better for girth.
        # Try a mix of random and structured.
        
        found = False
        # Increase attempts as n gets closer to known bounds? No, keep fixed to ensure speed.
        # 500 attempts per n * 80 n-values = 40k checks. Each check is BFS on n*k states.
        # n=320, k=4 -> 1280 states. Very fast.
        
        max_attempts = 1000
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Generate random offsets
            offsets = []
            used = set()
            valid = True
            for _ in range(k):
                o = random.randint(1, n - 1)
                if o in used:
                    valid = False
                    break
                for existing in offsets:
                    if (o + existing) % n == 0:
                        valid = False
                        break
                if not valid:
                    break
                offsets.append(o)
                used.add(o)
            
            if not valid or len(offsets) != k:
                continue
            
            if check_girth_cyclic(n, offsets):
                edges = make_cyclic_graph(n, offsets)
                if edges is not None:
                    if check_girth_from_edges(edges, n, g):
                        if n < best_n:
                            best_n = n
                            best_edges = edges
                        found = True
                        break
        
        if found:
            # Since we iterate n in increasing order, the first found n is minimal
            # for this random search. However, we might have missed a valid set for
            # a smaller n due to randomness. But since we go up, if we find n=300,
            # we assume 243..299 didn't have one (or we missed them).
            # To be robust, we could continue searching a bit, but time is limited.
            # Given the "smallest known is 320", finding anything < 320 is a win.
            # Finding 320 is a tie.
            # We break early to save time for the final verification or just return.
            # Actually, if we find n=320, we should probably stop.
            # If we find n=300, we stop.
            # Let's just return the first one found in the increasing n loop.
            # This ensures we return the smallest n found by the search.
            break

    if best_edges is not None:
        return best_n, best_edges

    # Fallback: LW Graph
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
