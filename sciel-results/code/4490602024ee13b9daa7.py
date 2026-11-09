import time
import random
import sys

def construct(k, g):
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Moore bound
    # For k=4, g=10: M = 2 * (1 + 3 + 9 + 27 + 81) = 242
    # We know 320 is possible.
    # Let's try from 242 up to 320.
    
    def get_girth(n, adj):
        """Returns the girth of the graph. INF if no cycles (impossible for regular)."""
        min_girth = float('inf')
        for v in range(n):
            # BFS
            visited = {v: 0}
            parent = {v: -1}
            queue = [(v, 0)]
            while queue:
                u, dist = queue.pop(0)
                for w in adj[u]:
                    if w == parent[u]:
                        continue
                    if w in visited:
                        # Found a cycle
                        # Cycle length = dist + visited[w] + 1
                        cycle_len = dist + visited[w] + 1
                        if cycle_len < min_girth:
                            min_girth = cycle_len
                    else:
                        visited[w] = dist
                        parent[w] = u
                        queue.append((w, dist + 1))
            if min_girth < g: # Early exit if we already found a short cycle
                return min_girth
        return min_girth

    def is_valid(n, adj, g):
        return get_girth(n, adj) >= g

    def generate_random_4reg(n):
        # Generate a random 4-regular graph
        # Using the configuration model or simple edge swapping
        # Start with a cycle 0-1-...-n-1, then add edges to make it 4-reg?
        # Easier: Start with empty, add edges randomly ensuring degree <= 4.
        # Or: Start with a random permutation-based construction.
        
        # Simple method:
        # Create 2n stubs for each vertex? No, 4 stubs per vertex.
        # Pair stubs randomly.
        # This creates a multigraph with loops.
        # Then repair loops and multi-edges by swapping.
        
        stubs = []
        for i in range(n):
            for _ in range(k):
                stubs.append(i)
        
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        for i in range(0, 2*n*k, 2):
            u = stubs[i]
            v = stubs[i+1]
            if u == v:
                # Loop, will fix later
                pass
            # Add edge
            adj[u].append(v)
            adj[v].append(u)
            
        # Now we have a multigraph. We need to remove loops and multi-edges.
        # This is complex.
        
        # Alternative: Use a known generator for simple regular graphs.
        # Or just use a greedy edge addition.
        
        # Let's use a different approach:
        # Start with a random 2-regular graph (union of cycles).
        # Then add a random 2-regular graph (union of cycles) disjoint? No.
        # Just add edges one by one.
        
        adj = [[] for _ in range(n)]
        degrees = [0] * n
        edges = set()
        
        # Add a random Hamiltonian cycle to ensure connectivity and base degree 2
        perm = list(range(n))
        random.shuffle(perm)
        for i in range(n):
            u = perm[i]
            v = perm[(i+1)%n]
            if u != v:
                adj[u].append(v)
                adj[v].append(u)
                degrees[u] += 1
                degrees[v] += 1
                edges.add((min(u,v), max(u,v)))
        
        # Now add edges to reach degree 4
        # We need to add 2 edges per vertex.
        # Total edges to add: n.
        # Available pairs: (u, v) where deg[u] < 4, deg[v] < 4, and (u,v) not in edges.
        
        # This is a matching problem on the remaining stubs.
        # Let's just do random swaps from a complete graph?
        
        # Simpler: Use the "Random 4-regular" generator from networkx if available? 
        # No, standard lib only.
        
        # Let's try a simple iterative repair.
        # Start with random pairing of 4-stubs.
        # Fix loops and multi-edges.
        
        def fix_graph(adj, n):
            # Remove loops
            for u in range(n):
                adj[u] = [v for v in adj[u] if v != u]
            
            # Remove multi-edges
            for u in range(n):
                adj[u] = list(set(adj[u]))
            
            # Now degrees are messed up.
            # This is getting complicated.
            
            pass

        # Let's use a robust method:
        # 1. Create list of all possible edges.
        # 2. Shuffle.
        # 3. Greedily add edges if they don't violate degree constraints.
        # This might get stuck.
        
        # Better: Start with a random 4-regular multigraph (config model) and repair.
        
        stubs = []
        for i in range(n):
            stubs.extend([i]*k)
        random.shuffle(stubs)
        
        adj = [[] for _ in range(n)]
        for i in range(0, len(stubs), 2):
            u, v = stubs[i], stubs[i+1]
            adj[u].append(v)
            adj[v].append(u)
            
        # Repair phase
        # 1. Remove loops
        for u in range(n):
            adj[u] = [v for v in adj[u] if v != u]
            
        # 2. Remove multi-edges (keep one)
        for u in range(n):
            adj[u] = list(set(adj[u]))
            
        # 3. Now degrees are < 4. We need to add edges to restore degree 4.
        # And we removed some edges, so we have "holes".
        # This is equivalent to finding a k-regular graph superimposed on the current one?
        
        # Actually, a simpler repair for config model:
        # While there are loops or multi-edges:
        #   Find a loop (u,u). Pick a random edge (v,w). Remove (u,u) and (v,w). Add (u,v) and (u,w).
        #   Find a multi-edge (u,v) appearing twice. Pick random edge (x,y). Remove one (u,v) and (x,y). Add (u,x) and (v,y).
        
        # This preserves regularity!
        
        # Let's implement this repair.
        
        # We need a list of edges to operate on.
        # Reconstruct edge list from adj?
        # adj is a list of lists, might have duplicates.
        
        # Let's maintain a list of edges.
        
        # Restarting with edge list representation for repair.
        
        def generate_repairable(n):
            stubs = []
            for i in range(n):
                stubs.extend([i]*k)
            random.shuffle(stubs)
            
            edges = []
            for i in range(0, len(stubs), 2):
                edges.append((stubs[i], stubs[i+1]))
            
            # Repair
            max_iter = 1000
            for _ in range(max_iter):
                # Find loop
                loop_idx = None
                multi_idx = None
                # We need to check for loops and multi-edges.
                # This is O(E) per check.
                
                edge_set = {}
                for idx, (u, v) in enumerate(edges):
                    if u == v:
                        if loop_idx is None:
                            loop_idx = idx
                    else:
                        key = (min(u,v), max(u,v))
                        if key in edge_set:
                            if multi_idx is None:
                                multi_idx = idx
                                # Store the other index? We just need one pair to fix.
                                # Actually, we need to know which edge to swap with.
                                # We can just pick a random other edge.
                        else:
                            edge_set[key] = idx
                
                if loop_idx is not None:
                    # Fix loop
                    u, v = edges[loop_idx]
                    # Pick random edge j != loop_idx
                    j = random.choice([i for i in range(len(edges)) if i != loop_idx])
                    x, y = edges[j]
                    # Remove (u,u) and (x,y)
                    # Add (u,x) and (u,y)
                    edges[loop_idx] = (u, x)
                    edges[j] = (u, y)
                    continue
                    
                if multi_idx is not None:
                    # Fix multi-edge
                    u, v = edges[multi_idx]
                    # Find the other edge with same endpoints
                    other_idx = None
                    for idx, (a, b) in enumerate(edges):
                        if idx == multi_idx: continue
                        if (min(a,b), max(a,b)) == (min(u,v), max(u,v)):
                            other_idx = idx
                            break
                    if other_idx is None: continue # Should not happen
                    
                    # Pick random edge j != multi_idx, other_idx
                    j = random.choice([i for i in range(len(edges)) if i != multi_idx and i != other_idx])
                    x, y = edges[j]
                    # Remove one (u,v) and (x,y)
                    # Add (u,x) and (v,y)
                    edges[multi_idx] = (u, x)
                    edges[j] = (v, y)
                    continue
                    
                # No loops or multi-edges
                break
            
            # Build adj
            adj = [[] for _ in range(n)]
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            return adj

        adj = generate_repairable(n)
        return adj

    best_n = None
    best_edges = None
    
    # Try n from 242 to 320
    for n in range(242, 321):
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this n
        found = False
        for restart in range(50):
            if time.time() > deadline:
                break
                
            adj = generate_random_4reg(n)
            
            # Check girth
            girth = get_girth(n, adj)
            
            if girth >= g:
                # Valid!
                edges = []
                for u in range(n):
                    for v in adj[u]:
                        if u < v:
                            edges.append((u, v))
                return n, edges
            else:
                # Local search
                # Try to improve
                for step in range(100):
                    if time.time() > deadline:
                        break
                    
                    # Pick a random edge (u,v)
                    # and a random edge (x,y)
                    # Swap
                    # We need to pick edges from adj.
                    # Flatten adj to edge list for easy random pick?
                    # Or just pick u, v in adj[u].
                    
                    u = random.randint(0, n-1)
                    if not adj[u]: continue
                    v = random.choice(adj[u])
                    
                    # Find another edge (x,y) disjoint from (u,v)
                    x = random.randint(0, n-1)
                    if x in (u, v): continue
                    if not adj[x]: continue
                    y = random.choice(adj[x])
                    if y in (u, v): continue
                    if (x, y) == (u, v) or (x, y) == (v, u): continue
                    # Also check if (u,x) or (u,y) or (v,x) or (v,y) already exist?
                    # If they do, we create multi-edges.
                    # We should avoid that.
                    
                    # Check validity of swap
                    if x in adj[u] or y in adj[u] or x in adj[v] or y in adj[v]:
                        continue
                    if u in adj[x] or v in adj[x] or u in adj[y] or v in adj[y]:
                        continue
                        
                    # Perform swap
                    # Remove (u,v) and (x,y)
                    adj[u].remove(v)
                    adj[v].remove(u)
                    adj[x].remove(y)
                    adj[y].remove(x)
                    
                    # Add (u,x) and (v,y)
                    adj[u].append(x)
                    adj[x].append(u)
                    adj[v].append(y)
                    adj[y].append(v)
                    
                    # Check girth
                    new_girth = get_girth(n, adj)
                    if new_girth >= g:
                        found = True
                        break
                    # If girth didn't improve, we might want to revert?
                    # For now, just keep going (greedy/random walk)
                    
        if found:
            edges = []
            for u in range(n):
                for v in adj[u]:
                    if u < v:
                        edges.append((u, v))
            return n, edges

    # If no small graph found, fall back to LUW
    # Re-implementing LUW from the seed to ensure it's there
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
