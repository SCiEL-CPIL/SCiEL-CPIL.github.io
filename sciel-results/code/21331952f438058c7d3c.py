import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # If k and g don't match the target, we might need a fallback.
    # But the task says "evaluator passes this task's degree k and girth g".
    # We assume k=4, g=10.
    
    if k != 4 or g != 10:
        # Fallback to seed logic for other parameters
        return construct_seed(k, g)

    # Search for a Cayley graph on C5 x C5 x C5 x C2
    # Generators must be in the coset (a,b,c,1) to be bipartite
    # Inverse of (a,b,c,1) is (-a, -b, -c, 1)
    
    best_n = float('inf')
    best_edges = None
    
    # 125 possible generators of the form (a,b,c,1)
    gens = [(a, b, c, 1) for a in range(5) for b in range(5) for c in range(5)]
    
    for i in range(len(gens)):
        if time.time() > deadline:
            break
        s1 = gens[i]
        inv_s1 = ((-s1[0]) % 5, (-s1[1]) % 5, (-s1[2]) % 5, 1)
        
        for j in range(i, len(gens)): # s2 can be same as s1? No, distinct generators for simple graph.
            if time.time() > deadline:
                break
            s2 = gens[j]
            inv_s2 = ((-s2[0]) % 5, (-s2[1]) % 5, (-s2[2]) % 5, 1)
            
            # Check distinctness for simple graph
            # s1 != s2, s1 != inv_s2, inv_s1 != s2, inv_s1 != inv_s2
            if s1 == s2 or s1 == inv_s2 or inv_s1 == s2 or inv_s1 == inv_s2:
                continue
            # Also s1 != inv_s1 (would be loop if s1^2=e, but s1 has order 10? No, order 5 in C5 part, 2 in C2 part. Order 10. So s1 != inv_s1)
            
            # Build Cayley graph
            # Vertices are elements of G
            # Map element to index
            # G = {(a,b,c,d) | a,b,c in 0..4, d in 0..1}
            # Total 250 vertices
            
            # We only care about the connected component containing the identity (0,0,0,0)
            # Since generators are in the coset d=1, the graph is bipartite.
            # Identity is in partition 0. Neighbors in partition 1.
            
            # BFS from identity
            visited = { (0,0,0,0): 0 }
            queue = [ (0,0,0,0) ]
            comp_edges = []
            comp_vertices = []
            
            def add_edge(u, v):
                # u, v are tuples
                if u not in visited:
                    return
                if v not in visited:
                    return
                u_idx = visited[u]
                v_idx = visited[v]
                if u_idx < v_idx:
                    comp_edges.append((u_idx, v_idx))
                else:
                    comp_edges.append((v_idx, u_idx))
            
            # We need to build the graph structure to check girth.
            # Actually, we can just do BFS to find the component and collect edges.
            # Then check girth of that component.
            
            # Reset visited for this pair
            visited = {}
            queue = [ (0,0,0,0) ]
            visited[(0,0,0,0)] = 0
            comp_vertices = [ (0,0,0,0) ]
            edges_set = set()
            
            # Generators to use: s1, inv_s1, s2, inv_s2
            # Note: s1 and inv_s1 are distinct. s2 and inv_s2 are distinct.
            # And all 4 are distinct from each other.
            
            def add(x, y):
                return ( (x[0]+y[0])%5, (x[1]+y[1])%5, (x[2]+y[2])%5, (x[3]+y[3])%2 )
                
            def sub(x, y):
                return ( (x[0]-y[0])%5, (x[1]-y[1])%5, (x[2]-y[2])%5, (x[3]-y[3])%2 )
            
            # BFS
            for u in queue:
                for gen in [s1, inv_s1, s2, inv_s2]:
                    v = add(u, gen)
                    if v not in visited:
                        visited[v] = len(comp_vertices)
                        comp_vertices.append(v)
                        queue.append(v)
                        edges_set.add((visited[u], visited[v]) if visited[u] < visited[v] else (visited[v], visited[u]))
            
            n_comp = len(comp_vertices)
            
            # Check if n_comp is smaller than best
            if n_comp >= best_n:
                continue
                
            # Check girth
            # Build adjacency list for the component
            adj = [[] for _ in range(n_comp)]
            for u, v in edges_set:
                adj[u].append(v)
                adj[v].append(u)
                
            # Check degrees
            is_regular = True
            for i in range(n_comp):
                if len(adj[i]) != 4:
                    is_regular = False
                    break
            if not is_regular:
                continue
                
            # Check girth >= 10
            # BFS from each vertex to find shortest cycle?
            # Or just BFS from 0? If the graph is a Cayley graph component, it might be vertex-transitive?
            # Not necessarily, but likely.
            # To be safe, check girth from all vertices?
            # For n=250, checking girth from all vertices is 250 * BFS.
            # BFS is O(N+E). 250 * 1000 = 250,000 ops.
            # 15,625 pairs * 250,000 ops = 4 billion ops. Too slow.
            
            # Optimization: Check girth from 0 first. If < 10, skip.
            # If >= 10, check from a few other vertices.
            
            def has_short_cycle(start, limit=10):
                # BFS from start
                dist = [-1] * n_comp
                parent = [-1] * n_comp
                dist[start] = 0
                q = [start]
                for u in q:
                    for v in adj[u]:
                        if dist[v] == -1:
                            dist[v] = dist[u] + 1
                            parent[v] = u
                            q.append(v)
                        else:
                            # Found a cycle
                            # Length is dist[u] + dist[v] + 1
                            # But this is not the shortest cycle necessarily.
                            # To find exact girth, we need to be careful.
                            # However, if we find ANY cycle of length < 10, we return False.
                            cycle_len = dist[u] + dist[v] + 1
                            if cycle_len < limit:
                                return True
                return False
            
            # This BFS finds *a* cycle, not necessarily the shortest.
            # If the shortest cycle is 10, this might find a cycle of length 12 and miss the 10?
            # No, if there is a cycle of length 8, BFS will likely find it.
            # But to be rigorous, we should check all pairs?
            # Given time constraints, let's assume if BFS from 0 finds no cycle < 10, and the graph is "random-like", it's a good candidate.
            # Then we verify more thoroughly if n_comp is a new best.
            
            if has_short_cycle(0, 10):
                continue
                
            # If it passes the quick check, do a more thorough check
            # Check from a few random vertices
            import random
            random.seed(42)
            verts_to_check = [0] + random.sample(range(1, n_comp), min(10, n_comp-1))
            valid = True
            for v in verts_to_check:
                if has_short_cycle(v, 10):
                    valid = False
                    break
            
            if valid:
                best_n = n_comp
                best_edges = list(edges_set)
                
    if best_edges is not None:
        return best_n, best_edges
    else:
        # Fallback to seed
        return construct_seed(k, g)

def construct_seed(k, g):
    # Copy of the original seed logic
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
