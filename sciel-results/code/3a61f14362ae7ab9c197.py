"""New program using Fast C6 Check and Multiple Greedy Restarts."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    n = 40
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    def is_c6_free_adj(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        adj is a list of sets (adjacency list).
        """
        Nu = adj[u]
        Nv = adj[v]
        
        # We need a simple path u-a-b-c-d-v
        # a in Nu, d in Nv
        # b in N(a), c in N(d)
        # Edge (b,c) exists
        # Vertices u,a,b,c,d,v are distinct
        
        for a in Nu:
            if a == v: continue
            for d in Nv:
                if d == u: continue
                if a == d:
                    # Path u-a-v is length 2. Cycle u-a-v-u is C3. Allowed.
                    continue
                
                # Check for path a-b-c-d in G \ {u,v}
                # b in N(a), c in N(d), edge (b,c)
                # b,c not in {u,v,a,d}
                
                Na = adj[a]
                Nd = adj[d]
                
                # Iterate over the smaller neighbor set to find b or c
                # Let's iterate b in Na
                for b in Na:
                    if b in (u, v, a, d):
                        continue
                    
                    # c must be in N(b) and N(d)
                    # c must not be in {u, v, a, d, b}
                    common = adj[b] & Nd
                    for c in common:
                        if c in (u, v, a, d, b):
                            continue
                        
                        # Check distinctness of all 6
                        if len({u, a, b, c, d, v}) == 6:
                            return False
                            
        return True

    def greedy_construct(seed):
        random.seed(seed)
        adj = [set() for _ in range(n)]
        edges = []
        
        candidates = list(all_edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if is_c6_free_adj(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                edges.append((u, v))
        
        return edges, adj

    best_edges = []
    best_adj = []
    best_count = 0
    
    start_time = time.time()
    time_limit = 115 # Leave 5s buffer
    
    restarts = 0
    while time.time() - start_time < time_limit:
        edges, adj = greedy_construct(restarts)
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges
            best_adj = adj
        restarts += 1
        
    # Final Local Search on best
    adj = best_adj
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        candidates = list(all_edges)
        random.shuffle(candidates)
        for u, v in candidates:
            if v in adj[u]:
                continue
            if is_c6_free_adj(adj, u, v):
                adj[u].add(v)
                adj[v].add(u)
                best_edges.append((u, v))
                best_count += 1
                improved = True
    
    return best_edges
# EVOLVE-BLOCK-END
