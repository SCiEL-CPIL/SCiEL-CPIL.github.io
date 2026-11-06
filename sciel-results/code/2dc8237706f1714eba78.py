import time
import random

def _is_3_colorable(adj, n):
    # adj is list of bitmasks
    # Order by degree
    order = sorted(range(n), key=lambda x: -bin(adj[x]).count('1'))
    colors = [-1] * n
    
    def backtrack(idx):
        if idx == n:
            return True
        v = order[idx]
        forbidden = 0
        for nb in range(n):
            if (adj[v] >> nb) & 1:
                if colors[nb] != -1:
                    forbidden |= (1 << colors[nb])
        
        for c in range(3):
            if not (forbidden & (1 << c)):
                colors[v] = c
                if backtrack(idx + 1):
                    return True
                colors[v] = -1
        return False
        
    return backtrack(0)

def construct():
    n = 16
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    grotzsch_edges = [
        (0,1), (1,2), (2,3), (3,4), (4,0),
        (5,1), (5,4),
        (6,0), (6,2),
        (7,1), (7,3),
        (8,2), (8,4),
        (9,0), (9,3),
        (10,5), (10,6), (10,7), (10,8), (10,9)
    ]
    
    def init_adj(edges, n):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    def count_edges(adj):
        return sum(bin(a).count('1') for a in adj) // 2

    def greedy_random_complete(adj, n, rng):
        adj = adj[:]
        while True:
            candidates = []
            for u in range(n):
                for v in range(u+1, n):
                    if not (adj[u] >> v) & 1:
                        if (adj[u] & adj[v]) == 0:
                            candidates.append((u, v))
            if not candidates:
                break
            u, v = rng.choice(candidates)
            adj[u] |= 1 << v
            adj[v] |= 1 << u
        return adj

    best_adj = None
    best_edges = 0
    
    # Precompute base adj for Grötzsch
    base_adj = init_adj(grotzsch_edges, n)
    base_edges = count_edges(base_adj)
    
    # Verify base is not 3-colorable (should be)
    # if _is_3_colorable(base_adj, n):
    #    print("Error: Grötzsch is 3-colorable?")
    
    num_trials = 20000
    
    for i in range(num_trials):
        if time.time() > deadline - 10:
            break
            
        rng = random.Random(i)
        
        # Generate a core
        # 50% chance use full Grötzsch, 50% remove 1-2 edges
        core_edges = grotzsch_edges[:]
        if i % 2 == 0:
             # Remove 1 or 2 random edges
             num_removals = rng.randint(1, 2)
             for _ in range(num_removals):
                 if core_edges:
                     idx = rng.randrange(len(core_edges))
                     core_edges.pop(idx)
        
        core_adj = init_adj(core_edges, n)
        
        # Check if core is still not 3-colorable
        # Only check if we removed edges
        if i % 2 == 0:
             if _is_3_colorable(core_adj, n):
                 continue # Skip if it became 3-colorable
        
        # Complete
        adj_full = greedy_random_complete(core_adj, n, rng)
        edges = count_edges(adj_full)
        
        if edges > best_edges:
            best_edges = edges
            best_adj = adj_full[:]
            
        # Local search
        if time.time() < deadline - 20:
             for _ in range(20):
                 if time.time() > deadline - 15: break
                 all_edges = []
                 for u in range(n):
                     for v in range(u+1, n):
                         if (adj_full[u] >> v) & 1:
                             all_edges.append((u,v))
                 if not all_edges: break
                 u, v = rng.choice(all_edges)
                 
                 adj_temp = adj_full[:]
                 adj_temp[u] &= ~(1 << v)
                 adj_temp[v] &= ~(1 << u)
                 
                 adj_new = greedy_random_complete(adj_temp, n, rng)
                 new_edges = count_edges(adj_new)
                 
                 if new_edges > best_edges:
                     best_edges = new_edges
                     best_adj = adj_new[:]
                     adj_full = adj_new[:]

    edges_list = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj[u] >> v) & 1:
                edges_list.append([u, v])
                
    return edges_list
