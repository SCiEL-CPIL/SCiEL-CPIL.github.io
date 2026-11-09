import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s kill
    
    n = 18
    best_l = n
    best_graph = None  # adj as list of bitmasks
    
    # --- Helper: check K4-free using bitmasks ---
    def is_k4_free(adj):
        for a in range(n):
            na = adj[a]
            for b in range(a+1, n):
                if not (na & (1 << b)): continue
                nb = adj[b]
                common_ab = na & nb
                # c must be > b and in common_ab
                for c in range(b+1, n):
                    if not (common_ab & (1 << c)): continue
                    nc = adj[c]
                    # triangle a-b-c exists. Check if any d in common_ab & nc
                    if common_ab & nc:
                        return False
        return True
    
    # --- Helper: find largest triangle-free induced subgraph size ---
    # Optimized: iterate k from n down, use early termination
    def get_largest_tf(adj):
        # Precompute for each edge (u,v) the bitmask of common neighbors
        # A subset S is triangle-free iff for every edge (u,v) with u,v in S,
        # (common_neighbors[u][v] & S) == 0
        
        # Actually simpler: for each subset, check if any triangle exists.
        # Use bitmasks for speed.
        
        # Precompute edge bitmasks: for each u, the set of v>u adjacent
        # And for each pair (u,v), common neighbors mask
        
        # For n=18, 2^18 = 262144 subsets. For each, checking triangles
        # can be done by: for each edge in the induced subgraph, check common neighbors.
        
        # Faster approach: iterate k from n down to 1
        # For each k, iterate all C(n,k) subsets, check triangle-free
        # Stop at first k where any subset is triangle-free
        
        # Precompute: for each u, adj[u] as bitmask (already have)
        # For each pair (u,v) with u<v, precompute common[u][v] = adj[u] & adj[v]
        common = [[0]*n for _ in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                if adj[u] & (1 << v):
                    common[u][v] = adj[u] & adj[v]
                else:
                    common[u][v] = 0
        
        for k in range(n, 0, -1):
            for subset in combinations(range(n), k):
                m = 0
                for v in subset:
                    m |= (1 << v)
                # Check triangle-free: for each edge (u,v) in subset, common[u][v] & m == 0
                tf = True
                for i in range(len(subset)):
                    u = subset[i]
                    for j in range(i+1, len(subset)):
                        v = subset[j]
                        if common[u][v] & m:
                            tf = False
                            break
                    if not tf:
                        break
                if tf:
                    return k
        return 0
    
    # --- Helper: generate random K4-free graph ---
    def gen_random_k4_free(p=0.5):
        adj = [0] * n
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < p:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        # Repair: while not K4-free, remove an edge from a K4
        for _ in range(200):
            if is_k4_free(adj):
                return adj
            # Find a K4 and remove an edge
            found = False
            for a in range(n):
                na = adj[a]
                for b in range(a+1, n):
                    if not (na & (1 << b)): continue
                    nb = adj[b]
                    common_ab = na & nb
                    for c in range(b+1, n):
                        if not (common_ab & (1 << c)): continue
                        nc = adj[c]
                        if common_ab & nc:
                            # K4 found: a,b,c,d where d is in common_ab & nc
                            dmask = common_ab & nc
                            d = (dmask & -dmask).bit_length() - 1
                            # Remove edge (a,d)
                            adj[a] &= ~(1 << d)
                            adj[d] &= ~(1 << a)
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                return adj
        return adj
    
    # --- Helper: local search to improve (reduce largest TF) ---
    def local_search(adj, max_iter=500):
        """Try edge additions/removals to reduce the largest triangle-free subgraph."""
        current_l = get_largest_tf(adj)
        
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            improved = False
            # Try adding a random non-edge
            for attempt in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if adj[u] & (1 << v): continue
                # Try adding edge (u,v)
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                if is_k4_free(adj):
                    new_l = get_largest_tf(adj)
                    if new_l < current_l:
                        current_l = new_l
                        improved = True
                        break
                    else:
                        # Revert
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
                else:
                    # Revert
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
            
            if improved:
                continue
            
            # Try removing a random edge
            for attempt in range(20):
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u == v: continue
                if not (adj[u] & (1 << v)): continue
                # Try removing edge (u,v)
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
                new_l = get_largest_tf(adj)
                if new_l < current_l:
                    current_l = new_l
                    improved = True
                    break
                else:
                    # Revert
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            if not improved:
                break
        
        return adj, current_l
    
    # --- Main search loop ---
    # Start with structured initialization: 3-partite with parts of 6
    # Then random restarts with local search
    
    # Initial structured graph: blow-up of C5 or similar
    # Try a few structured starts
    
    # Start 1: Complete 3-partite graph K_{6,6,6} minus some edges to be K4-free
    # K_{6,6,6} has K4s (pick 2 from two parts). Need to reduce.
    # Actually, let's just do random with various p values.
    
    p_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    while time.time() < deadline - 10:
        p = random.choice(p_values)
        adj = gen_random_k4_free(p)
        if adj is None:
            continue
        
        l = get_largest_tf(adj)
        
        if l < best_l:
            best_l = l
            best_graph = adj[:]
        
        # Local search to try to improve
        if time.time() < deadline - 15:
            adj_ls, l_ls = local_search(adj, max_iter=200)
            if l_ls < best_l:
                best_l = l_ls
                best_graph = adj_ls[:]
        
        if best_l <= 6:
            # Very good, can stop early
            break
    
    # Convert best_graph to list of edges
    if best_graph is None:
        # Fallback: empty graph
        return []
    
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if best_graph[i] & (1 << j):
                edges.append([i, j])
    return edges
