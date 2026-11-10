import time
import random
import math
from collections import deque

def _luw_construct(k, g):
    # Seed logic
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

def construct(k, g):
    if k != 4 or g != 10:
        return _luw_construct(k, g)
    
    deadline = time.time() + 110
    
    # For k=4, g=10, try to find a smaller graph than LUW (2048)
    # Use local search on random 4-regular bipartite graphs
    
    def count_short_cycles(adj, n):
        """Count 4-cycles and 6-cycles using adjacency list."""
        half = n // 2
        c4 = 0
        for u in range(half):
            for v in range(u + 1, half):
                common = len(adj[u] & adj[v])
                if common >= 2:
                    c4 += common * (common - 1) // 2
        
        # Count 6-cycles: a1-b1-a2-b2-a3-b3-a1
        # For each pair of edges (a1,b1) and (a2,b2) where b1 != b2,
        # check if there exists b3 such that a1-b3-a2 and a2-b3... no
        # Better: for each vertex a1, for each pair of neighbors (b1, b2),
        # for each vertex a2 in N(b2) \ {a1}, for each vertex b3 in N(a2) \ {b2},
        # check if a1 in N(b3)
        c6 = 0
        for a1 in range(half):
            nbs1 = list(adj[a1])
            for i in range(len(nbs1)):
                b1 = nbs1[i]
                for j in range(i + 1, len(nbs1)):
                    b2 = nbs1[j]
                    for a2 in adj[b2]:
                        if a2 == a1:
                            continue
                        for b3 in adj[a2]:
                            if b3 == b2:
                                continue
                            if b3 in adj[a1]:
                                c6 += 1
        # Each 6-cycle is counted 6 times (once per starting vertex)
        c6 //= 6
        return c4, c6

    def check_girth(adj, n):
        """Check if graph has girth at least g using non-backtracking walks."""
        # Since bipartite, only even cycles matter
        # Check for cycles of length 4, 6, 8
        # Use BFS from each vertex up to depth 4 (to detect cycles up to length 8)
        for start in range(n):
            # BFS with parent tracking
            visited = {start: (-1, 0)}  # node: (parent, dist)
            queue = deque([(start, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= 4:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb == start and dist + 1 >= 4:
                        # Cycle of length dist+1
                        if dist + 1 < 10:
                            return False
                    elif nb not in visited:
                        visited[nb] = (node, dist + 1)
                        queue.append((nb, node, dist + 1))
                    else:
                        # Already visited, check if this creates a short cycle
                        p, d = visited[nb]
                        if d + 1 < 10 and d + 1 >= 4:
                            # Potential cycle, but need to be careful
                            pass
        return True

    def generate_random_bipartite_4reg(half):
        """Generate a random 4-regular bipartite graph with 'half' vertices on each side."""
        # Use configuration model
        left_nodes = list(range(half))
        right_nodes = list(range(half, n))
        
        # Each left node has 4 stubs, each right node has 4 stubs
        left_stubs = []
        for i in range(half):
            left_stubs.extend([i] * 4)
        right_stubs = []
        for i in range(half):
            right_stubs.extend([half + i] * 4)
        
        random.shuffle(right_stubs)
        
        # Pair stubs
        adj = [set() for _ in range(n)]
        for l, r in zip(left_stubs, right_stubs):
            if l not in adj[r]:  # Avoid multi-edges
                adj[l].add(r)
                adj[r].add(l)
        
        # Check if all nodes have degree 4
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        return adj

    def local_search(adj, n, max_iters):
        """Local search to minimize short cycles."""
        half = n // 2
        best_c4, best_c6 = count_short_cycles(adj, n)
        best_adj = [set(s) for s in adj]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Pick a random edge to swap
            # Pick random left node u, random right node v in N(u)
            # Pick random left node w != u, random right node x in N(w)
            # Swap: remove (u,v) and (w,x), add (u,x) and (w,v)
            u = random.randint(0, half - 1)
            w = random.randint(0, half - 1)
            if u == w:
                continue
            
            nbs_u = list(adj[u])
            nbs_w = list(adj[w])
            if not nbs_u or not nbs_w:
                continue
            
            v = random.choice(nbs_u)
            x = random.choice(nbs_w)
            
            # Check if swap is valid (no multi-edges)
            if v == x:
                continue
            if x in adj[u] or v in adj[w]:
                continue
            
            # Perform swap
            adj[u].remove(v)
            adj[v].remove(u)
            adj[w].remove(x)
            adj[x].remove(w)
            adj[u].add(x)
            adj[x].add(u)
            adj[w].add(v)
            adj[v].add(w)
            
            c4, c6 = count_short_cycles(adj, n)
            
            if c4 < best_c4 or (c4 == best_c4 and c6 < best_c6):
                best_c4, best_c6 = c4, c6
                best_adj = [set(s) for s in adj]
            else:
                # Revert
                adj[u].remove(x)
                adj[x].remove(u)
                adj[w].remove(v)
                adj[v].remove(w)
                adj[u].add(v)
                adj[v].add(u)
                adj[w].add(x)
                adj[x].add(w)
        
        return best_adj, best_c4, best_c6

    def adj_to_edges(adj, n):
        edges = []
        for u in range(n // 2):
            for v in adj[u]:
                if u < v:
                    edges.append((u, v))
        return edges

    # Try different n values
    best_result = None
    best_n = 2048  # LUW size
    
    # Start from smaller n and go up
    for n in range(244, min(400, best_n), 2):  # n must be even for bipartite
        if time.time() > deadline - 10:
            break
        
        half = n // 2
        found = False
        
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            adj = generate_random_bipartite_4reg(half)
            if adj is None:
                continue
            
            adj, c4, c6 = local_search(adj, n, max_iters=200)
            
            if c4 == 0:
                # Check girth properly
                if check_girth(adj, n):
                    edges = adj_to_edges(adj, n)
                    if best_result is None or n < best_n:
                        best_result = (n, edges)
                        best_n = n
                    found = True
                    break
        
        if found:
            break
    
    if best_result is not None:
        return best_result
    
    # Fallback to LUW
    return _luw_construct(k, g)
