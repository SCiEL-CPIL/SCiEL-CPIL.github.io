import time
import random

def construct():
    """
    Constructs a C5-free graph with no independent set of size 9.
    Uses the known 32-vertex construction (8 disjoint K4s) and attempts
    to extend it using local search with exact validity checks.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Helper: check if graph has a C5 (cycle of length 5)
    def has_c5(n, adj):
        """Check if graph with adjacency sets has a 5-cycle."""
        # For each edge (u, v), check if there's a path of length 4 from u to v
        # not using edge (u,v), which would form a C5.
        # Path of length 4: u-a-b-c-v
        # So we need: a in N(u), c in N(v), a != c, a != v, c != u,
        # and there exists b such that b in N(a) ∩ N(c), b != u, b != v, b != a, b != c.
        
        # For each edge (u, v):
        for u in range(n):
            for v in adj[u]:
                if v <= u:
                    continue  # avoid duplicate edges
                # Find common neighbors of N(u) and N(v) that can form a path
                # We need a in N(u)\{v}, c in N(v)\{u}, a != c, and N(a) ∩ N(c) non-empty
                # excluding u, v, a, c
                
                nu = adj[u] - {v}
                nv = adj[v] - {u}
                
                for a in nu:
                    for c in nv:
                        if a == c:
                            continue
                        # Check if there's a common neighbor b of a and c
                        # b must be in N(a) ∩ N(c), b != u, v, a, c
                        common = adj[a] & adj[c]
                        common -= {u, v, a, c}
                        if common:
                            return True
        return False
    
    # Helper: check if graph has an independent set of size 9
    def has_independent_set_9(n, adj):
        """Check if there's an independent set of size 9."""
        # Use branch and bound with bitsets
        masks = [0] * n
        for i in range(n):
            for j in adj[i]:
                masks[i] |= (1 << j)
        
        def dfs(vertices_mask, size):
            if size >= 9:
                return True
            if vertices_mask == 0:
                return False
            
            # Upper bound: size + popcount(vertices_mask)
            if size + bin(vertices_mask).count('1') < 9:
                return False
            
            # Pick a vertex
            lsb = vertices_mask & (-vertices_mask)
            v = lsb.bit_length() - 1
            
            # Include v
            if dfs(vertices_mask & ~(masks[v] | (1 << v)), size + 1):
                return True
            
            # Exclude v
            if dfs(vertices_mask & ~(1 << v), size):
                return True
            
            return False
        
        all_mask = (1 << n) - 1
        return dfs(all_mask, 0)
    
    # Helper: check if graph is valid
    def is_valid(n, adj):
        if has_c5(n, adj):
            return False
        if has_independent_set_9(n, adj):
            return False
        return True
    
    # Start with 8 disjoint K4s
    n = 32
    adj = [set() for _ in range(n)]
    for b in range(8):
        base = b * 4
        for i in range(4):
            for j in range(i+1, 4):
                adj[base+i].add(base+j)
                adj[base+j].add(base+i)
    
    # Verify base graph is valid
    base_valid = is_valid(n, adj)
    if not base_valid:
        return {'n': 0, 'edges': []}
    
    best_n = n
    best_adj = [set(s) for s in adj]
    
    # Try to extend to 33 vertices
    # Connect vertex 32 to one vertex from each K4
    n_new = 33
    
    # Try systematic patterns
    found_33 = False
    for pattern_seed in range(500):
        if time.time() > deadline:
            break
        
        if pattern_seed == 0:
            # First try: connect to vertex 0 of each K4
            conn = [0, 0, 0, 0, 0, 0, 0, 0]
        else:
            random.seed(pattern_seed)
            conn = [random.randint(0, 3) for _ in range(8)]
        
        adj_new = [set(s) for s in adj]
        adj_new.append(set())
        
        for b in range(8):
            v = b * 4 + conn[b]
            adj_new[32].add(v)
            adj_new[v].add(32)
        
        if is_valid(n_new, adj_new):
            best_n = n_new
            best_adj = [set(s) for s in adj_new]
            found_33 = True
            break
    
    # If we found a 33-vertex graph, try to extend to 34
    if best_n >= 33:
        n_new = 34
        adj_base = [set(s) for s in best_adj]
        
        for seed in range(200):
            if time.time() > deadline:
                break
            random.seed(seed)
            
            adj_new = [set(s) for s in adj_base]
            adj_new.append(set())
            
            # Connect vertex 33 to some vertices
            # Try different numbers of connections
            num_conn = random.randint(1, 6)
            targets = random.sample(range(33), num_conn)
            for v in targets:
                adj_new[33].add(v)
                adj_new[v].add(33)
            
            if is_valid(n_new, adj_new):
                best_n = n_new
                best_adj = [set(s) for s in adj_new]
                break
    
    # Convert best_adj to edge list
    edges = []
    for i in range(best_n):
        for j in best_adj[i]:
            if i < j:
                edges.append([i, j])
    
    return {'n': best_n, 'edges': edges}
