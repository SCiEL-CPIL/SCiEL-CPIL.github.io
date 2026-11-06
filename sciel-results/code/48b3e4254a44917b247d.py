import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def check_k4_free(n, adj):
        """Check if graph is K4-free using adjacency sets."""
        for v in range(n):
            nbrs_v = adj[v]
            # Get neighbors of v
            for u in nbrs_v:
                if u <= v:
                    continue
                # For each w in nbrs_v ∩ nbrs_u with w > u
                common = nbrs_v & nbrs_u
                for w in common:
                    if w <= u:
                        continue
                    # Check if any x in common ∩ nbrs_w
                    if common & adj[w]:
                        return False
        return True

    def check_no_independent_set(n, adj, k=6):
        """Check if graph has no independent set of size k using branch and bound."""
        # Independent set in G = clique in complement
        # Complement adjacency
        comp_adj = [set() for _ in range(n)]
        for v in range(n):
            for u in range(n):
                if u != v and u not in adj[v]:
                    comp_adj[v].add(u)
        
        # Find clique of size k in complement
        # Use greedy coloring upper bound for pruning
        def greedy_color(candidates):
            """Greedy coloring to get upper bound on clique size."""
            if not candidates:
                return 0
            colors = {}
            uncolored = list(candidates)
            color_idx = 0
            while uncolored:
                # Pick a vertex with max degree in remaining
                v = max(uncolored, key=lambda x: len(comp_adj[x] & set(uncolored)))
                colors[v] = color_idx
                uncolored.remove(v)
                color_idx += 1
            return color_idx
        
        def find_clique(candidates, k_remaining):
            if k_remaining == 0:
                return True
            if not candidates:
                return False
            if len(candidates) < k_remaining:
                return False
            # Upper bound: greedy coloring
            if greedy_color(candidates) < k_remaining:
                return False
            for v in list(candidates):
                candidates.remove(v)
                new_candidates = candidates & comp_adj[v]
                if find_clique(new_candidates, k_remaining - 1):
                    return True
            return False
        
        return not find_clique(set(range(n)), k)

    def is_valid(n, adj):
        """Check K4-free and no independent set of 6."""
        if not check_k4_free(n, adj):
            return False
        if not check_no_independent_set(n, adj, 6):
            return False
        return True

    def edges_from_adj(n, adj):
        edges = []
        for i in range(n):
            for j in adj[i]:
                if i < j:
                    edges.append([i, j])
        return edges

    # Strategy: Start with known valid graphs and try to extend them
    # Known: Paley(17) is valid. Try to build larger graphs.
    
    # First, try the 35-vertex construction from the seed (verify it)
    n35 = 35
    adj35 = [set() for _ in range(n35)]
    D5 = {1, 4}
    D7 = {1, 2, 4}
    for i in range(n35):
        a1, b1 = i // 7, i % 7
        for j in range(i + 1, n35):
            a2, b2 = j // 7, j % 7
            da = (a2 - a1) % 5
            db = (b2 - b1) % 7
            if (da in D5 and db in D7) or (da not in D5 and db not in D7):
                adj35[i].add(j)
                adj35[j].add(i)
    
    if is_valid(n35, adj35):
        return {'n': 35, 'edges': edges_from_adj(n35, adj35)}
    
    # Try Paley graph on 17 vertices (known valid)
    n17 = 17
    adj17 = [set() for _ in range(n17)]
    squares17 = {x * x % 17 for x in range(1, 17)}
    for i in range(n17):
        for j in range(i + 1, n17):
            if (j - i) % 17 in squares17:
                adj17[i].add(j)
                adj17[j].add(i)
    
    if is_valid(n17, adj17):
        # Try to extend this graph by adding vertices
        best_n = n17
        best_adj = adj17[:]
        
        # Try adding one vertex at a time
        for attempt in range(200):
            if time.time() > deadline - 10:
                break
            
            # Try to add a new vertex
            new_n = len(best_adj) + 1
            new_adj = [s.copy() for s in best_adj]
            new_adj.append(set())
            
            # Try all possible neighborhoods for the new vertex
            # The new vertex must not create a K4 or independent 6-set
            # For K4-free: new vertex's neighbors must not contain a triangle
            # For no independent 6-set: non-neighbors of new vertex must not contain independent 5-set
            
            # Sample random neighborhoods
            for trial in range(50):
                if time.time() > deadline - 5:
                    break
                
                # Random subset of existing vertices as neighbors
                candidate_nbrs = set(random.sample(range(len(best_adj)), 
                                                   random.randint(0, len(best_adj))))
                
                new_adj2 = [s.copy() for s in best_adj]
                new_adj2.append(set())
                for v in candidate_nbrs:
                    new_adj2[v].add(new_n - 1)
                    new_adj2[new_n - 1].add(v)
                
                if is_valid(new_n, new_adj2):
                    best_adj = new_adj2
                    best_n = new_n
                    break
        
        if best_n > n17:
            return {'n': best_n, 'edges': edges_from_adj(best_n, best_adj)}
        
        return {'n': n17, 'edges': edges_from_adj(n17, adj17)}
    
    # Fallback: return empty graph
    return {'n': 1, 'edges': []}
