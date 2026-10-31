"""Construct a 3-regular graph with girth >= g on as few vertices as possible."""
import time
from collections import deque

# EVOLVE-BLOCK-START
def construct(k, g):
    deadline = time.time() + 140  # 10s safety margin

    def check_girth_fast(n, adj, k, g):
        """
        Check if graph has girth >= g.
        For a graph to have girth >= g (even), the BFS tree from any vertex
        up to depth r = g//2 - 1 must be a tree (no cycles, correct vertex counts).
        """
        r = g // 2 - 1
        expected_counts = [1] + [k * (k - 1) ** i for i in range(r)]
        
        for start in range(n):
            dist = [-1] * n
            dist[start] = 0
            queue = deque([start])
            counts = [0] * (r + 1)
            counts[0] = 1
            
            while queue:
                u = queue.popleft()
                if dist[u] >= r:
                    continue
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        if dist[v] <= r:
                            counts[dist[v]] += 1
                            queue.append(v)
                    # If dist[v] != -1, we have a cycle. 
                    # But we rely on the count check. If counts match, 
                    # it implies no cycles in that radius.
            
            if counts != expected_counts:
                return False
        return True

    def margulis_graph(p):
        """
        Construct the Margulis 3-regular graph on SL(2, Z_p).
        Note: This implementation uses a symmetric generating set to ensure 
        the graph is undirected and 3-regular.
        Generators: 
        A = [[1, 1], [0, 1]]
        B = [[1, 0], [1, 1]]
        C = [[0, 1], [-1, 0]]
        To make it undirected, we need S = S^-1.
        We use S = {C, A, A^-1}? No, that's 3 generators.
        C is an involution? C^2 = -I. Not I.
        Let's use the standard 3-regular Ramanujan construction:
        Generators: 
        G1 = [[1, 1], [0, 1]]
        G2 = [[1, 0], [1, 1]]
        G3 = [[0, 1], [-1, 0]]
        This set is not symmetric.
        However, we can define the graph on SL(2, p) with edges (M, M*G) for G in {G1, G2, G3}.
        This is a directed graph.
        To make it undirected, we add edges (M, M*G^-1) as well?
        That would make it 6-regular.
        
        Alternative: Use the bipartite double cover of the Margulis graph?
        Or use a different construction.
        
        Given the complexity, let's just use the LUW construction but try to optimize it 
        or use a smaller d if possible.
        """
        return None

    # Fallback: LUW
    q = k
    d = g - 5 if (g - 5) % 2 else g - 4
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
# EVOLVE-BLOCK-END
