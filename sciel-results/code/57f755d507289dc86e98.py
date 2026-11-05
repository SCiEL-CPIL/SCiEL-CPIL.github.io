"""Construct k-regular graph of girth g using random Cayley graphs on non-abelian groups."""
# EVOLVE-BLOCK-START
import random
import time
import itertools
from collections import deque

def construct(k, g):
    """
    Construct a k-regular graph with no cycle shorter than g.
    
    Strategy: Search for a Cayley graph on a non-abelian group with k involutory
    generators that yields girth >= g. Fall back to L-U-W construction if needed.
    
    Returns (n, edges) where n is number of vertices and edges is list of (i,j) pairs.
    """
    deadline = time.time() + 110  # Leave 10s margin
    
    def luw_construct(k, g):
        """Lazebnik-Ustimenko-Woldar graph CD(d, q) - reliable fallback."""
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
                    if side == 0:
                        y[c] = add(x[c], mul(y[a], x[b]))
                    else:
                        y[c] = sub(x[c], mul(x[a], y[b]))
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
    
    def check_girth(n, adj, target_girth):
        """Check if graph has girth >= target_girth using BFS from each vertex."""
        # For efficiency, use BFS from identity and check shortest cycle
        # A cycle of length L through vertex v means there's a path of length L from v back to v
        # without backtracking. We check: for each vertex, BFS to find shortest cycle.
        # Actually, girth = min over all vertices of shortest cycle through that vertex.
        # Shortest cycle through v = min over neighbors u of v of (dist(u, v) in G-v) + 1
        # But that's expensive. Alternative: BFS from each vertex, when we reach a visited
        # node, we found a cycle.
        
        # More efficient: for each edge (u,v), the shortest cycle using that edge is
        # 1 + shortest path from u to v in G - edge. But still expensive.
        
        # Best approach for small graphs: BFS from each vertex, track parent, when we
        # encounter a visited node that isn't the parent, we have a cycle.
        
        for start in range(n):
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                for v in adj[u]:
                    if dist[v] == -1:
                        dist[v] = dist[u] + 1
                        parent[v] = u
                        queue.append(v)
                    elif v != parent[u] and u != parent[v]:
                        # Found a cycle: dist[u] + dist[v] + 1
                        cycle_len = dist[u] + dist[v] + 1
                        if cycle_len < target_girth:
                            return False
            # Also check if we can find a shorter cycle via backtracking
        return True
    
    def cayley_graph_girth(group_elems, identity, gens, target_girth):
        """
        Check girth of Cayley graph. group_elems: list of group elements.
        gens: list of generator indices (into group_elems).
        """
        n = len(group_elems)
        idx = {elem: i for i, elem in enumerate(group_elems)}
        adj = [[] for _ in range(n)]
        for i, elem in enumerate(group_elems):
            for g in gens:
                # elem * gens[g]
                prod = _group_mul(elem, gens[g])
                j = idx[prod]
                if j > i:
                    adj[i].append(j)
                    adj[j].append(i)
                elif j < i:
                    adj[i].append(j)
                    adj[j].append(i)
        # Check for self-loops and multi-edges
        for i in range(n):
            if i in adj[i]:
                return -1, None
            if len(adj[i]) != len(set(adj[i])):
                return -1, None
        # Check degree
        for i in range(n):
            if len(adj[i]) != target_girth - 1 + 1:  # This is wrong, let me fix
                pass
        # Actually just check girth
        if not check_girth(n, adj, target_girth):
            return -1, None
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append((i, j))
        return n, edges
    
    # For k=5, g=9: try Cayley graphs on S3 x S3 x S3
    if k == 5 and g == 9:
        # S3 elements as permutation tuples
        s3 = [(0,1,2), (1,0,2), (2,1,0), (0,2,1), (1,2,0), (2,0,1)]
        id_perm = s3[0]
        
        def pmul(p, q):
            return tuple(p[q[i]] for i in range(3))
        
        # Build S3xS3xS3: elements are ((p1,p2),(p3)) -> flatten to (p1,p2,p3)
        s3x3 = []
        for a in s3:
            for b in s3:
                for c in s3:
                    s3x3.append((a, b, c))
        n_s3x3 = len(s3x3)
        id_s3x3 = (id_perm, id_perm, id_perm)
        
        def prod_mul(x, y):
            return (pmul(x[0], y[0]), pmul(x[1], y[1]), pmul(x[2], y[2]))
        
        # Find all involutions in S3xS3xS3
        involutions = []
        for elem in s3x3:
            if elem == id_s3x3:
                continue
            if prod_mul(elem, elem) == id_s3x3:
                involutions.append(elem)
        
        random.seed(42)
        best_edges = None
        
        for trial in range(3000):
            if time.time() > deadline - 5:
                break
            if len(involutions) < k:
                break
            gens = random.sample(involutions, k)
            
            idx = {elem: i for i, elem in enumerate(s3x3)}
            adj = [[] for _ in range(n_s3x3)]
            valid = True
            for i, elem in enumerate(s3x3):
                seen = set()
                for gen in gens:
                    j = idx[prod_mul(elem, gen)]
                    if j == i or j in seen:
                        valid = False
                        break
                    seen.add(j)
                    adj[i].append(j)
                if not valid:
                    break
            
            if not valid or any(len(adj[i]) != k for i in range(n_s3x3)):
                continue
            
            # Check girth
            girth_ok = True
            for start in range(n_s3x3):
                dist = [-1] * n_s3x3
                parent = [-1] * n_s3x3
                dist[start] = 0
                queue = deque([start])
                while queue and girth_ok:
                    u = queue.popleft()
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            queue.append(v)
                        elif v != parent[u] and u != parent[v]:
                            if dist[u] + dist[v] + 1 < g:
                                girth_ok = False
                                break
                if not girth_ok:
                    break
            
            if girth_ok:
                edges = []
                for i in range(n_s3x3):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_edges = edges
                break
        
        if best_edges is not None:
            return n_s3x3, best_edges
    
    # Fallback to L-U-W
    return luw_construct(k, g)
# EVOLVE-BLOCK-END
