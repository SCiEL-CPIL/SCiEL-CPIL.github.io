import math
from collections import deque

def _check_circulant_girth(n, a, b, min_girth):
    """
    Check if the circulant graph on Z_n with steps {a, -a, b, -b} 
    has girth at least min_girth.
    """
    # Verify simplicity
    if a == b or a == -b % n or a == 0 or b == 0:
        return False
    if 2 * a % n == 0 or 2 * b % n == 0:
        return False
        
    # BFS from vertex 0
    # State: (current_node, previous_node, dist)
    # We track visited nodes with their distance from 0 and the path parent?
    # Actually, to find the shortest cycle, we can do BFS from 0.
    # A cycle exists if we find a neighbor that is already visited.
    # The cycle length is dist(current) + dist(neighbor) + 1.
    # This works for finding the girth if we start from one vertex?
    # No, the shortest cycle might not pass through 0.
    # However, for circulant graphs, the graph is vertex-transitive.
    # So the shortest cycle length is the same from any vertex.
    # Thus, checking from vertex 0 is sufficient.
    
    visited = {0: (0, -1)} # node -> (dist, parent)
    queue = deque([(0, -1, 0)])
    
    while queue:
        node, parent, dist = queue.popleft()
        
        if dist >= min_girth // 2:
            continue
            
        neighbors = [
            (node + a) % n,
            (node - a) % n,
            (node + b) % n,
            (node - b) % n
        ]
        
        for nb in neighbors:
            if nb == parent:
                continue
            if nb in visited:
                # Cycle found
                cycle_len = dist + visited[nb][0] + 1
                if cycle_len < min_girth:
                    return False
            else:
                visited[nb] = (dist + 1, node)
                queue.append((nb, node, dist + 1))
                
    return True

def construct(k, g):
    if k != 4 or g != 10:
        # Fallback to original logic or simple error
        # The task specifies k=4, g=10.
        # We should handle other cases gracefully, perhaps returning a large valid graph if possible, 
        # but the contract says "evaluator passes this task's degree k and girth g".
        # Assuming k=4, g=10 is the main target.
        pass

    # Search for circulant graphs
    # Moore bound for (4,10) is 243.
    # Try n from 243 to 320 (known best).
    
    for n in range(243, 321):
        # a, b must be such that the graph is simple.
        # a, b in [1, n//2 - 1]
        # We can restrict search space.
        # Usually small a, b are bad (short cycles).
        # We want a, b to be "large" relative to n? Or specific algebraic properties.
        
        # To speed up, we can skip many pairs.
        # Let's iterate a from 1 to n//2 - 1
        # and b from a+1 to n//2 - 1
        
        # Optimization: 
        # If n is small, the number of pairs is O(n^2).
        # For n=320, n^2/4 ~ 25600 pairs.
        # Girth check is O(n * degree^depth). Depth 5.
        # 320 * 4^5 ~ 320 * 1024 ~ 300k ops per check.
        # 25600 * 300k is too slow.
        
        # We need to prune.
        # 4-cycles exist if there is a relation a +/- a = b +/- b (0), a +/- a = b (impossible), etc.
        # Specifically, a 4-cycle 0 -> a -> ... -> 0.
        # 0, a, a+b, b. Cycle if a+b = -a => 2a = -b. Or a+b = -b => a = -2b.
        # So avoid 2a = -b mod n, 2b = -a mod n, a = b, a = -b.
        
        found = False
        for a in range(1, n // 2):
            for b in range(a + 1, n // 2):
                # Check simple conditions
                if (2 * a + b) % n == 0: continue
                if (a + 2 * b) % n == 0: continue
                if (a + b) % n == 0: continue # a = -b
                
                # Check girth
                if _check_circulant_girth(n, a, b, g):
                    # Construct edges
                    edges = []
                    for i in range(n):
                        edges.append((i, (i + a) % n))
                        edges.append((i, (i - a) % n))
                        edges.append((i, (i + b) % n))
                        edges.append((i, (i - b) % n))
                    # Remove duplicates if any (shouldn't be any if simple)
                    # The list might have duplicates if a == -a etc, but we checked that.
                    # However, (i, i+a) and (i+a, i) are the same edge.
                    # My loop adds both directions.
                    # The task expects a list of edges (i, j). Usually unordered pairs.
                    # Let's deduplicate and sort.
                    edge_set = set()
                    for u, v in edges:
                        if u > v: u, v = v, u
                        edge_set.add((u, v))
                    return n, list(edge_set)
        
    # If no circulant graph found, fall back to the original random search logic
    # (Paste original logic here as fallback)
    
    import random
    from collections import deque as dq
    
    def _check_girth(n, edges, min_girth):
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        
        for start in range(n):
            visited = {start: (-1, 0)}
            queue = dq([(start, -1, 0)])
            while queue:
                node, parent, dist = queue.popleft()
                if dist >= min_girth // 2:
                    continue
                for nb in adj[node]:
                    if nb == parent:
                        continue
                    if nb in visited:
                        cycle_len = dist + visited[nb][1] + 1
                        if cycle_len < min_girth:
                            return False
                    else:
                        visited[nb] = (node, dist + 1)
                        queue.append((nb, node, dist + 1))
        return True

    def _generate_random_4reg(n, max_tries=500):
        for _ in range(max_tries):
            stubs = []
            for v in range(n):
                stubs.extend([v] * 4)
            random.shuffle(stubs)
            edges = set()
            valid = True
            for i in range(0, len(stubs), 2):
                u, v = stubs[i], stubs[i+1]
                if u == v:
                    valid = False
                    break
                key = (min(u, v), max(u, v))
                if key in edges:
                    valid = False
                    break
                edges.add(key)
            if not valid:
                continue
            edge_list = list(edges)
            deg = [0] * n
            for u, v in edge_list:
                deg[u] += 1
                deg[v] += 1
            if any(d != 4 for d in deg):
                continue
            return edge_list
        return None

    def _try_random(n, g, max_graphs=200):
        for _ in range(max_graphs):
            edges = _generate_random_4reg(n, max_tries=100)
            if edges is None:
                continue
            if _check_girth(n, edges, g):
                return n, edges
        return None

    random.seed(42)
    for n in range(243, 321):
        result = _try_random(n, g, max_graphs=50)
        if result is not None:
            return result
    
    # Last resort: return a known valid construction (L-U-W style from original)
    # ... (Keep the original L-U-W fallback)
    
    q = 4
    d = 5
    table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
    add, sub, mul = (lambda x, y: x ^ y), (lambda x, y: x ^ y), (lambda x, y: table[x][y])
    rule = {2: (1, 1), 3: (2, 1), 4: (1, 2), 5: (1, 3)}
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
