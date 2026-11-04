import time
import random
from collections import deque

def construct(k, g):
    # Deadline: must return well before 150s
    deadline = time.time() + 140.0

    # Moore bound for (k, g)
    # For even girth g = 2r: n >= k + k*(k-1)*( (k-1)^r - 1 ) / (k-2)
    # For odd girth g = 2r+1: n >= 1 + k*sum_{i=0}^{r-1} (k-1)^i
    if g % 2 == 0:
        r = g // 2
        if k == 2:
            moore = g
        else:
            moore = k + k * (k - 1) * (((k - 1) ** r - 1) // (k - 2)) if k > 2 else 0
            # Actually for k=2, girth g means a cycle of length g, so n >= g
            if k == 2:
                moore = g
    else:
        r = (g - 1) // 2
        if k == 2:
            moore = g
        else:
            s = 0
            for i in range(r):
                s += (k - 1) ** i
            moore = 1 + k * s

    # For k=4, g=10: g=2*5, r=5
    # moore = 4 + 4*3*((3^5 - 1)/2) = 4 + 12*(242/2) = 4 + 12*121 = 4 + 1452 = 1456? 
    # Wait, that's wrong. Let me recompute.
    # For girth 10 (even), the Moore bound: n >= k + k(k-1)*sum_{i=0}^{g/2-2} (k-1)^i
    # Actually the standard Moore bound for even girth 2r:
    # n >= k + k(k-1) * ( (k-1)^{r-1} - 1 ) / (k-2)
    # For k=4, r=5: n >= 4 + 4*3*(3^4 - 1)/2 = 4 + 12*(80/2) = 4 + 12*40 = 4 + 480 = 484
    # Hmm, let me just use a simpler approach and not worry about exact Moore bound.

    # Strategy: use the Lazebnik-Ustimenko-Woldar construction, which is proven to give
    # girth >= d+5 for the CD(d,q) construction. For k=4, g=10, we need girth >= 10.
    # The LUW construction with d = g-5 = 5 (odd) gives girth >= d+5 = 10. 
    # The number of vertices in the connected component is 2*q^(d+1) / something... 
    # Actually for CD(d,q), the bipartite graph has 2*q^{d+1} vertices total but the 
    # connected component of (0, 0) has size 2*q^d. For q=4, d=5: 2*4^5 = 2*1024 = 2048.
    # That's way too many. 
    # 
    # Actually, I recall the LUW construction gives a graph with girth at least d+5 
    # on 2q^{d+1} vertices. For q=4, d=5: 2*4^6 = 8192. Even worse.
    # 
    # The original seed program uses this and gets some number. The score is 1.0 meaning
    # it's the baseline. Let me think of a better approach.
    #
    # Alternative: Use a known explicit construction. The smallest known 4-regular 
    # girth-10 graph has 320 vertices. Can we construct it explicitly?
    #
    # The (4,10)-cage is known but its explicit edge list is complex. 
    # 
    # Better approach: Use a circulant or semi-regular graph construction.
    # 
    # Actually, let me try a different explicit construction: the "M11" or 
    # voltage graph approach, or simply try to use a known parametrized family.
    #
    # Key insight: The LUW construction CD(d,q) for q=4, d=5 gives a bipartite 
    # 4-regular graph. Let me just use it properly and see what n we get.
    # The seed program already does this. The issue is it might be giving a 
    # larger graph than necessary.
    #
    # Let me try a completely different approach: use a random regular graph 
    # with girth verification, searching for small n.
    # 
    # For n=320, k=4: we need a 4-regular graph on 320 vertices with girth >= 10.
    # Random 4-regular graphs on 320 vertices typically have girth around 8-12.
    # 
    # Strategy: Generate random 4-regular graphs on increasing n (starting from Moore bound)
    # and check girth via BFS from each vertex up to depth (g-1)//2.
    # 
    # Actually, the most practical approach within time limits:
    # 1. Try the LUW construction (deterministic, guaranteed girth)
    # 2. Try to find smaller via local improvement or different parameters
    #
    # Let me first just implement the LUW construction correctly and efficiently.

    # LUW construction CD(d, q):
    # Vertices: points (0, p) and lines (1, l) where p, l are (d+1)-tuples over GF(q)
    # with first coordinate ignored (set to None/0).
    # Actually, the standard LUW: vertices are pairs (side, v) where v is in GF(q)^d
    # Points: side=0, v in GF(q)^d
    # Lines: side=1, v in GF(q)^d
    # Adjacency: point (0, p) ~ line (1, l) iff for all i in 2..d: l_i - p_i = l_{a_i} * p_{b_i}
    # where (a_i, b_i) is the rule for coordinate i.
    #
    # For d=5, q=4: we have 4^5 = 1024 points and 1024 lines, total 2048 vertices.
    # But the connected component might be smaller.
    #
    # Actually wait - in the seed, d is computed as:
    # d = max(3, g-5 if (g-5)%2 else g-4)
    # For g=10: g-5=5 (odd), so d=5.
    # The girth is at least d+5 = 10. Good.
    # 
    # The connected component size: for the LUW graph, if the group generated is the 
    # whole space, the component has 2*q^d vertices. For q=4, d=5: 2048.
    # 
    # That's a lot. The known optimum is 320. So LUW is far from optimal here.
    #
    # New plan: Use a known explicit (4,10)-cage construction if I can recall one,
    # or use a search-based approach.
    #
    # Practical search approach:
    # 1. Start with n = some lower bound (say 320, the known optimum)
    # 2. Generate random 4-regular graphs
    # 3. Verify girth >= 10 using BFS
    # 4. If found, return it
    # 5. If time runs out, return the LUW construction as fallback
    #
    # For random 4-regular graph generation: use configuration model (perfect matching of 4n stubs)
    #
    # Girth verification: BFS from each vertex up to depth 4 (since girth 10 means 
    # no cycle of length < 10, so BFS to depth 4 from each vertex should not find
    # any back edge to an ancestor at depth <= 3... actually need to be more careful).
    # 
    # To verify girth >= 10: from each vertex, do BFS up to depth 4. If any vertex 
    # at depth 4 has an edge to a vertex at depth <= 1 (other than its parent at depth 3),
    # that creates a cycle of length <= 5. Actually, I need to think more carefully.
    #
    # A cycle of length L means there's a closed walk of length L. For girth >= 10,
    # we need no closed walk of length 2,3,4,5,6,7,8,9.
    # 
    # Simplest: BFS from each vertex to depth 4. Track the tree. If any edge connects
    # two vertices at depths d1, d2 with d1 + d2 + 1 <= 9 (and neither is the parent-child
    # relation in the BFS tree), we have a cycle of length d1 + d2 + 1.
    # Actually, if from vertex v, we BFS and find that two vertices at depths a and b
    # (both <= 4) are adjacent, and the path from v to one is a, to the other is b,
    # then cycle length is a + b + 1. We need a + b + 1 >= 10, so a + b >= 9.
    # Since a, b <= 4, max a+b = 8, so a+b+1 <= 9 < 10. 
    # Wait, that means ANY edge between two vertices in the BFS ball of radius 4 
    # (that's not a tree edge) creates a cycle of length <= 9.
    # 
    # So the check is: from each vertex, BFS to depth 4. For every non-tree edge 
    # (u,w) where both u,w are in the BFS tree of v, compute depth(u) + depth(w) + 1.
    # If this is < 10, we have a violation.
    # 
    # Actually more precisely: the cycle goes v -> ... -> u (depth a), edge (u,w), w -> ... -> v (depth b).
    # Cycle length = a + 1 + b. We need this >= 10 for all such edges.
    # Since a, b <= 4, a + b + 1 <= 9, so ANY non-tree edge within the ball violates girth.
    # 
    # Therefore: girth >= 10 iff for every vertex v, the subgraph induced by the BFS 
    # ball of radius 4 around v is a tree (i.e., no cycles, no extra edges).
    # 
    # This is efficient to check!

    def generate_random_regular(n, degree, rng):
        """Generate a random degree-regular graph using configuration model."""
        if n * degree % 2 != 0:
            return None
        # Create stubs
        stubs = list(range(n * degree))
        rng.shuffle(stubs)
        # Pair them up
        adj = [set() for _ in range(n)]
        valid = True
        for i in range(0, n * degree, 2):
            u = stubs[i] // degree
            v = stubs[i + 1] // degree
            if u == v:
                valid = False
                break
            if v in adj[u]:
                valid = False
                break
            adj[u].add(v)
            adj[v].add(u)
        if not valid:
            return None
        # Verify regularity
        for i in range(n):
            if len(adj[i]) != degree:
                return None
        return adj

    def check_girth(adj, n, g):
        """Check if graph has girth >= g. Returns True if valid."""
        # BFS from each vertex to depth (g-1)//2
        # For g=10, depth = 4
        max_depth = (g - 1) // 2  # 4 for g=10
        
        for start in range(n):
            # BFS
            dist = [-1] * n
            parent = [-1] * n
            dist[start] = 0
            queue = deque([start])
            while queue:
                u = queue.popleft()
                if dist[u] >= max_depth:
                    continue
                for w in adj[u]:
                    if dist[w] == -1:
                        dist[w] = dist[u] + 1
                        parent[w] = u
                        if dist[w] < max_depth:
                            queue.append(w)
                    elif dist[w] <= max_depth and w != parent[u] and u != parent[w]:
                        # Found a non-tree edge: cycle of length dist[u] + dist[w] + 1
                        # But we need to be more careful - this edge might have been seen before
                        # Actually, we just need: is dist[u] + dist[w] + 1 < g?
                        if dist[u] + dist[w] + 1 < g:
                            return False
                    elif dist[w] == -1:
                        # Shouldn't happen
                        pass
        return True

    # Fallback: LUW construction
    def luw_construct(k, g):
        q = k
        d = max(3, g - 5 if (g - 5) % 2 else g - 4)
        if q == 4:
            table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
            add_fn = lambda x, y: x ^ y
            sub_fn = lambda x, y: x ^ y
            mul_fn = lambda x, y: table[x][y]
        else:
            add_fn = lambda x, y: (x + y) % q
            sub_fn = lambda x, y: (x - y) % q
            mul_fn = lambda x, y: x * y % q

        # Rule for LUW
        rule = {}
        rule[2] = (1, 1)
        rule[3] = (2, 1)
        rule[4] = (1, 2)
        for c in range(5, d + 1):
            r = (c - 5) % 4
            c0 = c - (c - 5) % 4
            rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]

        # Build adjacency via BFS from origin
        # Vertex: (side, tuple of d elements)
        # side 0 = point, side 1 = line
        start = (0, tuple([0] * d))
        
        index = {start: 0}
        order = [start]
        edges_set = set()
        
        def get_neighbours(vertex):
            side, x = vertex
            result = []
            for first in range(q):
                y = [0] * (d + 1)  # y[1]..y[d], y[0] unused
                y[1] = first
                for c in range(2, d + 1):
                    a, b = rule[c]
                    if side == 0:
                        # point to line: y[c] = x[c] + y[a] * x[b]
                        y[c] = add_fn(x[c - 1], mul_fn(y[a - 1], x[b - 1]))
                    else:
                        # line to point: y[c] = x[c] - x[a] * y[b]
                        y[c] = sub_fn(x[c - 1], mul_fn(x[a - 1], y[b - 1]))
                result.append((1 - side, tuple(y[1:])))
            return result

        for vertex in order:
            for other in get_neighbours(vertex):
                if other not in index:
                    index[other] = len(order)
                    order.append(other)
                if vertex[0] == 0:
                    e = (index[vertex], index[other])
                    if e not in edges_set:
                        edges_set.add(e)
        
        edges = list(edges_set)
        return len(order), edges

    # Try random search first if we have time
    rng = random.Random(42)
    found = False
    
    # Try n values starting from a reasonable lower bound
    # For k=4, g=10, known optimum is 320
    # Moore bound for (4,10): 4 + 4*3*(3^4 - 1)/2 = 4 + 12*40 = 484? 
    # Let me just try from 320 upward
    
    best_result = None
    
    # First, get the LUW result as baseline
    luw_n, luw_edges = luw_construct(k, g)
    best_result = (luw_n, luw_edges)
    
    # Now try random graphs for smaller n
    # Start from known optimum and go down a bit, then up
    # Actually, let's try a range
    start_n = 320
    end_n = min(600, luw_n)
    
    # Try in order of preference: smaller n first
    for trial_n in range(start_n, end_n + 1, 16):
        if time.time() > deadline - 30:
            break
        if trial_n % 2 != 0:
            continue
        # Need n * k to be even, which it is for k=4
        # Try multiple random graphs
        max_tries = 50
        for attempt in range(max_tries):
            if time.time() > deadline - 30:
                break
            adj = generate_random_regular(trial_n, k, rng)
            if adj is None:
                continue
            if check_girth(adj, trial_n, g):
                # Found it!
                edges = []
                for i in range(trial_n):
                    for j in adj[i]:
                        if i < j:
                            edges.append((i, j))
                best_result = (trial_n, edges)
                found = True
                break
        if found:
            break
    
    return best_result
