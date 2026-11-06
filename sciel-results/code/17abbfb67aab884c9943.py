import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s margin before 120s kill
    
    random.seed(42)
    
    def is_k4_free(adj, n):
        """Check if graph has no K4 using bitmasks."""
        # Build adjacency as bitmasks
        adj_bits = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_bits[i] |= (1 << j)
        
        for i in range(n):
            for j in range(i + 1, n):
                if j in adj[i]:
                    # Common neighbors of i and j
                    cn = adj_bits[i] & adj_bits[j]
                    # Remove i and j
                    cn &= ~(1 << i)
                    cn &= ~(1 << j)
                    # Check if any edge exists in cn
                    while cn:
                        w = (cn & -cn).bit_length() - 1
                        cn &= cn - 1
                        # Check if w has a neighbor in cn
                        if adj_bits[w] & cn:
                            return False
        return True
    
    def has_independent_set(adj, n, size=6):
        """Check if graph has independent set of given size using backtracking."""
        # Convert to adjacency bitmasks for speed
        adj_bits = [0] * n
        for i in range(n):
            for j in adj[i]:
                adj_bits[i] |= (1 << j)
        
        # Use Bron-Kerbosch style for maximal independent sets
        # Or simpler: backtracking with pruning
        def backtrack(candidates, chosen):
            if len(chosen) == size:
                return True
            if not candidates:
                return False
            if len(candidates) + len(chosen) < size:
                return False
            
            # Pick a vertex from candidates
            # Heuristic: pick vertex with fewest neighbors in candidates
            best_v = -1
            best_deg = n + 1
            for v in candidates:
                deg = bin(adj_bits[v] & candidates).count('1')
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
            
            v = best_v
            candidates.remove(v)
            
            # Option 1: Include v
            # New candidates: vertices not adjacent to v
            new_cands = [u for u in candidates if u not in adj[v]]
            if backtrack(new_cands, chosen + [v]):
                return True
            
            # Option 2: Exclude v
            if backtrack(candidates, chosen):
                return True
            
            return False
        
        return backtrack(list(range(n)), [])
    
    def is_valid(adj, n):
        """Check both K4-free and no independent set of 6."""
        if not is_k4_free(adj, n):
            return False
        if has_independent_set(adj, n, 6):
            return False
        return True
    
    def try_add_vertex(adj, n):
        """Try to add a new vertex (vertex n) to the graph.
        Returns (new_adj, success) where new_adj is adj with vertex n added."""
        new_n = n + 1
        new_adj = [set() for _ in range(new_n)]
        for i in range(n):
            new_adj[i] = set(adj[i])
        
        # Try different connection patterns for the new vertex
        # The new vertex connects to a subset of existing vertices
        # To avoid K4: new vertex's neighbors must form an independent set in the old graph
        # To avoid I6: new vertex + 5 non-neighbors must not form I6
        
        # Get all subsets of existing vertices that form independent sets
        # (so that adding edges to them doesn't create K4 with the new vertex)
        
        # Simpler approach: try random subsets of neighbors
        # For the new vertex to not create K4, its neighbor set must be an independent set
        # in the original graph (no edges among neighbors)
        
        # Find all independent sets in the current graph (as candidate neighbor sets)
        # This is expensive, so let's use a heuristic:
        
        # Try: new vertex connects to a random independent set of existing vertices
        # The size of this set determines how many non-neighbors we have
        
        # For no I6: we need that no 5 vertices among the non-neighbors form an independent set
        # with the new vertex. So the non-neighbor set must have alpha <= 4.
        
        # Let's try different sizes of neighbor sets
        for attempt in range(20):
            # Pick a random subset of vertices to connect to
            # To keep it K4-free, the connected vertices must have no edges among them
            # (since new vertex connects to all of them, any edge among them + new vertex = K3, 
            #  but we need K4, so we need two adjacent vertices both connected to new vertex
            #  AND both connected to a common neighbor... wait)
            
            # Actually: new vertex v connects to set S. K4 containing v requires:
            # {v, a, b, c} where a,b,c in S and a~b, b~c, c~a (triangle in S)
            # So S must be triangle-free to avoid K4 with v.
            # Wait no: K4 requires all 6 edges. v-a, v-b, v-c, a-b, b-c, c-a.
            # So yes, S must contain no triangle.
            
            # But also: v connects to S. If there's an edge a-b in S, and a and b 
            # have a common neighbor c in S, then {v,a,b,c} is K4.
            # So S must be triangle-free.
            
            # Let's just try random subsets and check
            subset_size = random.randint(0, min(n, 15))
            S = random.sample(range(n), subset_size)
            
            # Check if S is triangle-free (no K3 in induced subgraph on S)
            s_set = set(S)
            triangle_free = True
            for a, b, c in combinations(S, 3):
                if b in adj[a] and c in adj[a] and c in adj[b]:
                    triangle_free = False
                    break
            
            if not triangle_free:
                continue
            
            # Build new adjacency
            new_adj = [set() for _ in range(new_n)]
            for i in range(n):
                new_adj[i] = set(adj[i])
            
            for v in S:
                new_adj[v].add(new_n - 1)
                new_adj[new_n - 1].add(v)
            
            # Check validity
            if is_valid(new_adj, new_n):
                return new_adj, True
        
        return None, False
    
    def local_search(adj, n):
        """Try to improve the graph by flipping edges to allow adding more vertices."""
        # This is a placeholder for now
        pass
    
    # Start with a small valid graph
    # Use a known small valid graph: C5 (5-cycle) is K4-free and has alpha=2
    # Or start with empty graph and build up
    
    # Let's start with a few known valid graphs and try to grow
    
    best_n = 0
    best_adj = None
    
    # Try starting from different small graphs
    start_graphs = []
    
    # Empty graph on 5 vertices (trivially valid)
    adj5 = [set() for _ in range(5)]
    start_graphs.append(adj5)
    
    # C5
    adj_c5 = [set() for _ in range(5)]
    for i in range(5):
        adj_c5[i].add((i + 1) % 5)
        adj_c5[(i + 1) % 5].add(i)
    start_graphs.append(adj_c5)
    
    # C7 (7-cycle)
    adj_c7 = [set() for _ in range(7)]
    for i in range(7):
        adj_c7[i].add((i + 1) % 7)
        adj_c7[(i + 1) % 7].add(i)
    start_graphs.append(adj_c7)
    
    # Paley 7 (actually Paley graph exists for p = 1 mod 4, so p=5, 13, 17, ...)
    # For p=5: squares mod 5 = {1, 4}
    adj_p5 = [set() for _ in range(5)]
    squares5 = {1, 4}
    for i in range(5):
        for j in range(i + 1, 5):
            if (j - i) % 5 in squares5:
                adj_p5[i].add(j)
                adj_p5[j].add(i)
    start_graphs.append(adj_p5)
    
    for start_adj in start_graphs:
        if time.time() > deadline - 20:
            break
        
        n = len(start_adj)
        adj = [set(s) for s in start_adj]
        
        # Verify starting graph is valid
        if not is_valid(adj, n):
            continue
        
        if n > best_n:
            best_n = n
            best_adj = [set(s) for s in adj]
        
        # Try to grow
        for _ in range(50):  # Try up to 50 growth steps
            if time.time() > deadline - 10:
                break
            
            new_adj, success = try_add_vertex(adj, n)
            if success:
                adj = new_adj
                n += 1
                if n > best_n:
                    best_n = n
                    best_adj = [set(s) for s in adj]
            else:
                # Try local search to enable growth
                # For now, just break and try a different starting graph
                break
    
    # If we haven't done well, try a more aggressive approach
    if best_n < 20:
        # Try random initialization and local search
        for trial in range(10):
            if time.time() > deadline - 5:
                break
            
            n_try = random.randint(15, 25)
            adj = [set() for _ in range(n_try)]
            
            # Add random edges while maintaining K4-free
            all_edges = [(i, j) for i in range(n_try) for j in range(i + 1, n_try)]
            random.shuffle(all_edges)
            
            for u, v in all_edges:
                if time.time() > deadline - 5:
                    break
                # Check if adding edge creates K4
                cn = adj[u].intersection(adj[v])
                k4 = False
                for w in cn:
                    if w in adj[v] and any(x in adj[w] for x in cn if x != w):
                        k4 = True
                        break
                if not k4:
                    adj[u].add(v)
                    adj[v].add(u)
            
            # Check validity
            if is_valid(adj, n_try):
                if n_try > best_n:
                    best_n = n_try
                    best_adj = [set(s) for s in adj]
    
    # Fallback to Paley 17 if nothing better
    if best_n < 17:
        squares = {x * x % 17 for x in range(1, 17)}
        edges = [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]
        return {'n': 17, 'edges': edges}
    
    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}

