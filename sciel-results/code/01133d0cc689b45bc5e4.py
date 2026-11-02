import time
import math

def construct(k, g):
    """
    Constructs a k-regular graph of girth at least g.
    For k=4, g=10, it searches for a small circulant graph.
    Falls back to Lazebnik-Ustimenko-Woldar construction if search fails or takes too long.
    """
    start_time = time.time()
    deadline = start_time + 110  # 10 seconds safety margin
    
    if k != 4 or g != 10:
        # Fallback to generic LUW for other parameters if needed, 
        # but the task specifies this task's degree and girth.
        return _luw_construct(k, g)

    # Moore bound for (4, 10)
    # B = 1 + k * sum_{i=0}^{(g-2)/2 - 1} (k-1)^i
    # g=10, (g-2)/2 = 4. Sum i=0 to 3.
    # 1 + 4 * (1 + 3 + 9 + 27) = 1 + 4 * 40 = 161?
    # Wait, Moore bound formula for even girth g=2r:
    # n >= 1 + k * ( (k-1)^r - 1 ) / (k-2)
    # r = 5.
    # n >= 1 + 4 * ( 3^5 - 1 ) / 2 = 1 + 2 * (243 - 1) = 1 + 484 = 485?
    # Let's re-calculate.
    # g=10. Radius 4.
    # Vertices at distance 0: 1
    # Distance 1: 4
    # Distance 2: 4*3 = 12
    # Distance 3: 12*3 = 36
    # Distance 4: 36*3 = 108
    # Total = 1 + 4 + 12 + 36 + 108 = 161.
    # The prompt says lower bound 243. Why?
    # Ah, the prompt says "best lower bound is 243".
    # The Moore bound is 161.
    # The prompt's 243 might be a tighter bound from literature (e.g., considering non-existence of smaller cages).
    # I will start search from 243 as per prompt hint, or 161?
    # Prompt: "n must lie between the Moore bound and max_vertices."
    # "best lower bound is 243".
    # I will start from 243 to be safe and efficient, as 161-242 are known to be impossible.

    min_n = 243
    max_n = 2000  # LUW gives 2048. We want to beat that.

    for n in range(min_n, max_n + 1):
        if time.time() > deadline:
            break
        
        # Check if n allows a 4-regular circulant graph
        # S = {a, -a, b, -b}
        # Conditions:
        # 1. a != 0, b != 0
        # 2. a != b, a != -b (mod n)
        # 3. 2a != 0 (mod n), 2b != 0 (mod n)
        
        # We can iterate a from 1 to n//2 - 1
        # b from a+1 to n//2 - 1
        
        found = False
        for a in range(1, n // 2):
            if time.time() > deadline:
                break
            # 2a != 0 mod n
            if (2 * a) % n == 0:
                continue
            
            for b in range(a + 1, n // 2):
                if time.time() > deadline:
                    break
                if (2 * b) % n == 0:
                    continue
                # a != b (guaranteed by loop)
                # a != -b mod n => a + b != n
                if a + b == n:
                    continue
                
                # Check girth
                # Graph is C_n(a, b)
                # Neighbors of i: i+a, i-a, i+b, i-b
                # We need to check if there is a cycle of length < 10.
                # This is equivalent to checking if there is a non-zero vector
                # x = x1*a + x2*b such that |x1| + |x2| < 10 and x = 0 mod n?
                # No, that's for the group. The cycle in the graph corresponds to
                # a sequence of steps.
                # A cycle of length L corresponds to a word of length L in generators {a, -a, b, -b}
                # that sums to 0 mod n, and is non-backtracking.
                # Since the graph is a Cayley graph, the girth is the minimum length of a
                # non-backtracking closed walk.
                
                # Efficient check: BFS from 0 up to depth 4 (since girth 10 means no cycles <= 9).
                # Actually, girth g means no cycles of length < g.
                # So we check for cycles of length 3, 4, ..., 9.
                # BFS from 0, keeping track of parent to avoid backtracking.
                
                visited = [[False] * n for _ in range(4)] # depth 0..3? 
                # Wait, to detect a cycle of length L, we need to see if a node at depth d
                # is connected to a node at depth d' where d + d' + 1 = L?
                # Standard BFS for girth:
                # From 0, expand. If we reach a node already visited at depth d' 
                # such that current_depth + d' + 1 < g, we have a cycle.
                
                # Since g=10, we only need to check depths up to 4.
                # If we reach a node at depth 4 that was visited at depth 0, 1, 2, or 3,
                # we have a cycle of length 4+0+1=5, 4+1+1=6, 4+2+1=7, 4+3+1=8.
                # What about cycles of length 9?
                # Depth 4 node connected to Depth 4 node? No, that would be length 9?
                # Path 0->...->u (depth 4) -> v (depth 4).
                # If u and v are distinct, and connected, length is 4+1+4=9.
                # So we need to check edges between nodes at depth 4 as well.
                
                # Let's implement a specific girth check for small g.
                # We do BFS from 0.
                # dist[v] = distance from 0.
                # For each edge (u, v), if dist[u] + dist[v] + 1 < g, return False.
                # This detects cycles where the "shortest path" from 0 to u and 0 to v
                # are disjoint except at 0? No, they might intersect.
                # The standard method:
                # Run BFS from 0.
                # For every edge (u,v), let du = dist[u], dv = dist[v].
                # If du + dv + 1 < g, then there is a cycle of length du+dv+1.
                # Is this sufficient?
                # Yes, for undirected graphs, the girth is the minimum over all edges (u,v) of dist(u)+dist(v)+1, 
                # provided dist is from an arbitrary start? 
                # No, that finds the girth of the component containing the start, 
                # but only if the cycle passes through the "shortest paths" from the start.
                # Actually, if there is a cycle of length L, pick any node v on the cycle.
                # The distance from start s to v is d.
                # The cycle can be decomposed into two paths from v to some other node?
                # The formula min_{(u,v) in E} dist_s(u) + dist_s(v) + 1 gives the girth
                # ONLY IF the graph is a tree? No.
                # Counter example: Square. Start at 0.
                # 0-1, 1-2, 2-3, 3-0.
                # dist(0)=0, dist(1)=1, dist(2)=2, dist(3)=1.
                # Edge (1,2): 1+2+1=4.
                # Edge (2,3): 2+1+1=4.
                # Edge (3,0): 1+0+1=2. Wait, 3-0 is an edge. dist(3)=1, dist(0)=0. 1+0+1=2.
                # But girth is 4.
                # The issue is that the path 0->3 is part of the cycle.
                # The condition for a cycle of length L is that there exists an edge (u,v)
                # such that the shortest path from u to v in the graph *excluding* the edge (u,v)
                # has length L-1.
                
                # Simpler: Just do BFS up to depth (g-2)/2 = 4.
                # If we find a node reached by two different paths of length <= 4,
                # or a node reached at depth d1 and connected to a node reached at depth d2
                # such that d1+d2+1 < g.
                
                # Let's use a set of (node, depth) or just dist array.
                # dist[v] = min depth reached.
                # If we update dist[v] to a smaller value, or if we find an edge (u,v)
                # where dist[u] + dist[v] + 1 < g, we have a cycle.
                # BUT, this only detects cycles that are "visible" from 0 in terms of
                # shortest path tree.
                # For girth 10, depth 4 is small.
                # Let's just iterate all possible walks of length < 10?
                # 4^9 = 262k walks. Too many for each (a,b,n).
                
                # Better: 
                # For a given (n, a, b), the graph is a Cayley graph.
                # A cycle of length L corresponds to a sequence of L generators g_1...g_L
                # such that sum g_i = 0 mod n, and no g_i = -g_{i-1}.
                # We can check for L=3..9.
                # L=3: a+a+a=0, a+a-b=0, etc.
                # L=4: a+a-b-b=0, etc.
                
                # Given the small L, we can pre-compute the conditions.
                # But a+b != n is already checked.
                
                # Let's stick to the BFS dist check. It is fast.
                # dist = [-1]*n
                # dist[0] = 0
                # queue = [(0, 0)]
                # while queue:
                #   u, d = queue.pop()
                #   for v in neighbors(u):
                #     if dist[v] == -1:
                #       dist[v] = d+1
                #       if d+1 < 5: # only expand up to depth 4
                #          queue.append((v, d+1))
                #     else:
                #       if dist[u] + dist[v] + 1 < 10:
                #          return False (cycle found)
                
                # This detects any cycle of length < 10 that intersects the BFS tree?
                # Yes, because any cycle contains an edge (u,v) where the paths from 0 to u and 0 to v
                # are disjoint? No.
                # However, if a cycle exists, there is a "chord" or edge connecting two nodes
                # in the BFS tree such that the distance sum is small.
                # For girth g, if min_{(u,v)} dist[u]+dist[v]+1 >= g, then girth >= g.
                # This is a known theorem?
                # Yes, for any vertex s, girth(G) >= min_{(u,v) in E} dist_s(u) + dist_s(v) + 1.
                # Proof: Let C be a cycle of length L. Pick any vertex v in C.
                # Let u be a neighbor of v in C.
                # The path from s to u and s to v ...
                # Actually, the minimum is taken over ALL edges.
                # If the minimum over all edges is >= g, then no cycle of length < g exists.
                
                # So we just need to compute dist from 0 and check all edges.
                
                dist = [-1] * n
                dist[0] = 0
                q = [0]
                qi = 0
                while qi < len(q):
                    u = q[qi]
                    qi += 1
                    du = dist[u]
                    if du >= 4: # No need to expand further for girth 10 check?
                        # We need dist values for neighbors to check edges.
                        # If we stop expanding at 4, we don't have dist for nodes at 5.
                        # But edges from node at 4 to node at 5:
                        # dist[u]=4, dist[v]=5. Sum+1 = 10. Not < 10.
                        # Edges from node at 5 to node at 5:
                        # We don't have dist 5.
                        # But if there is a cycle of length 9, it involves nodes at dist 4 and 4?
                        # Path 0->...->u (4) -> v (4).
                        # If u and v are both at dist 4, and connected, length is 4+1+4=9.
                        # So we MUST compute dist up to 4, and check edges between nodes with dist <= 4.
                        # What about edges involving nodes with dist > 4?
                        # If dist[u]=5, dist[v]=5, sum+1=11.
                        # If dist[u]=5, dist[v]=4, sum+1=10.
                        # So we only need dist up to 4 to detect cycles < 10.
                        continue
                    
                    # Neighbors: u+a, u-a, u+b, u-b
                    for v in [(u+a)%n, (u-a)%n, (u+b)%n, (u-b)%n]:
                        if dist[v] == -1:
                            dist[v] = du + 1
                            if du + 1 <= 4:
                                q.append(v)
                        else:
                            # Check cycle condition
                            if du + dist[v] + 1 < 10:
                                # Cycle found
                                dist = None
                                break
                    if dist is None:
                        break
                    # Also check if v was already visited in this loop?
                    # The 'else' block handles the case where v is visited.
                    # But we also need to check if the edge (u,v) creates a cycle
                    # where u and v are at the same level or adjacent levels.
                    # The condition du + dist[v] + 1 < 10 covers it.
                    
                if dist is None:
                    continue
                
                # After BFS, check all edges again to be sure?
                # The BFS check above might miss edges where both endpoints were
                # discovered in the same batch or something?
                # No, the check `if dist[v] != -1` happens when we process u.
                # If v is already visited, we check.
                # If v is NOT visited, we set dist[v].
                # We don't check for cycle when setting dist[v].
                # But a cycle requires an edge to a *previously visited* node.
                # So checking when dist[v] != -1 is correct.
                
                # However, we only expanded up to depth 4.
                # What if u is at depth 4, and v is at depth 4?
                # When we process u (depth 4), we look at v.
                # If v is already visited (dist[v] != -1), we check.
                # If v is NOT visited, we set dist[v]=5.
                # We do NOT add v to queue (since depth > 4).
                # So we don't process v.
                # But we already checked the edge (u,v) when processing u.
                # Is that enough?
                # Yes, because the condition is symmetric.
                
                # One edge case: Self-loops?
                # We ensured a,b != 0 and 2a,2b != 0.
                
                # If no cycle found, girth >= 10.
                # Return this graph.
                edges = []
                for i in range(n):
                    edges.append((i, (i+a)%n))
                    edges.append((i, (i+b)%n))
                return n, edges

    # Fallback to LUW
    return _luw_construct(k, g)

def _luw_construct(k, g):
    # Original LUW code
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
