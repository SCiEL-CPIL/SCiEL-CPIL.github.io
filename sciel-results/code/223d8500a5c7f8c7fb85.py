"""Construct a small 4-regular graph of girth 10 (cage problem n(4,10))."""
import time
import random

# EVOLVE-BLOCK-START

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110

    # Moore bound for (4,10)-cage: 1 + 4 + 4*3 + 4*3^2 + 4*3^3 = 1+4+12+36+108 = 161 (even)
    # Actually: 1 + 4*(1 + 3 + 9 + 27) = 1 + 4*40 = 161. But since g is even, 
    # Moore bound is 1 + k + k(k-1) + ... + k(k-1)^(g/2-1) = 1 + 4 + 12 + 36 = 53? 
    # No, for even girth g, Moore bound is 1 + k * (1 + (k-1) + ... + (k-1)^(g/2-1)).
    # For g=10, g/2-1 = 4. So 1 + 4*(1+3+9+27) = 1 + 4*40 = 161.
    # Best known lower bound is 243, best known construction is 320.
    
    # LUW graph construction
    def luw_construct(k, g):
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

    # Try to construct a 320-vertex (4,10)-cage using a voltage graph lift
    # The (4,10)-cage with 320 vertices can be constructed as a covering of a smaller graph.
    # One known construction: take a 4-regular graph on 80 vertices with girth >= 10
    # and lift it with voltage group Z_4. But this requires careful voltage assignment.
    
    # Alternative: use the known fact that the 320-vertex (4,10)-cage exists and
    # try to find it via randomized search with girth verification.
    
    def verify_girth(n, edges, min_girth):
        """Verify that the graph has girth at least min_girth using BFS."""
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for i in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[i] = 0
            queue = [i]
            while queue:
                u = queue.pop(0)
                if dist[u] >= min_girth // 2:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u]:
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < min_girth:
                            return False
        return True

    def generate_random_4reg(n):
        """Generate a random 4-regular graph on n vertices."""
        # Start with a random permutation and create two disjoint cycles,
        # then add two more perfect matchings.
        if n < 4:
            return None
        
        # Create a random 2-regular graph (disjoint cycles)
        perm = list(range(n))
        random.shuffle(perm)
        
        # Build adjacency for two cycles
        edges = set()
        # First cycle: 0-1-2-...-(n-1)-0
        for i in range(n):
            j = (i + 1) % n
            edges.add((min(i, j), max(i, j)))
        
        # Second cycle: use a different permutation
        perm2 = list(range(n))
        random.shuffle(perm2)
        # perm2[i] -> perm2[i+1]
        for i in range(n):
            a = perm2[i]
            b = perm2[(i + 1) % n]
            if a == b:
                continue
            edges.add((min(a, b), max(a, b)))
        
        # Now add two more perfect matchings
        # Perfect matching 1
        unmatched = list(range(n))
        random.shuffle(unmatched)
        for i in range(0, n, 2):
            a, b = unmatched[i], unmatched[i + 1]
            if (min(a, b), max(a, b)) not in edges:
                edges.add((min(a, b), max(a, b)))
            else:
                # Find another vertex to connect
                for j in range(i + 1, n):
                    a2, b2 = unmatched[i], unmatched[j]
                    if (min(a2, b2), max(a2, b2)) not in edges:
                        edges.add((min(a2, b2), max(a2, b2)))
                        # Swap to keep unmatched list consistent
                        unmatched[j] = unmatched[i + 1]
                        unmatched[i + 1] = b2
                        break
                else:
                    return None
        
        # Perfect matching 2
        unmatched = list(range(n))
        random.shuffle(unmatched)
        for i in range(0, n, 2):
            a, b = unmatched[i], unmatched[i + 1]
            if (min(a, b), max(a, b)) not in edges:
                edges.add((min(a, b), max(a, b)))
            else:
                for j in range(i + 1, n):
                    a2, b2 = unmatched[i], unmatched[j]
                    if (min(a2, b2), max(a2, b2)) not in edges:
                        edges.add((min(a2, b2), max(a2, b2)))
                        unmatched[j] = unmatched[i + 1]
                        unmatched[i + 1] = b2
                        break
                else:
                    return None
        
        # Verify all vertices have degree 4
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        for i in range(n):
            if len(adj[i]) != 4:
                return None
        
        return list(edges)

    # Try to find a 320-vertex graph with girth >= 10
    if k == 4 and g == 10:
        best_n = float('inf')
        best_edges = None
        
        # Try n = 320 with multiple seeds
        for seed in range(50):
            if time.time() > deadline:
                break
            random.seed(seed)
            edges = generate_random_4reg(320)
            if edges is not None:
                if verify_girth(320, edges, 10):
                    best_n = 320
                    best_edges = edges
                    break
        
        if best_edges is not None:
            return best_n, best_edges

    # Fallback: LUW graph
    return luw_construct(k, g)

# EVOLVE-BLOCK-END
