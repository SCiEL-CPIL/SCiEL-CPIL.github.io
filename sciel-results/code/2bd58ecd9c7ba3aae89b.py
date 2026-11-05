import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n = 40
    # Initial graph: 8 disjoint K5s
    # Vertices 0-4, 5-9, ..., 35-39
    initial_edges = set()
    for k in range(8):
        base = 5 * k
        for i in range(5):
            for j in range(i + 1, 5):
                initial_edges.add((base + i, base + j))
                initial_edges.add((base + j, base + i))
    
    # Precompute all possible edges
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    
    # Function to check if adding edge (u, v) creates a C6
    # Current graph is represented by adjacency sets
    # We will maintain adj as a list of sets
    def is_valid_edge(adj, u, v):
        # Check if path u-a-b-v-c-d-u exists with distinct vertices
        # u, v are endpoints.
        # a in N(u), b in N(a), (b,v) in E
        # c in N(v), d in N(c), (d,u) in E
        # Distinctness: u, a, b, v, c, d all distinct.
        
        # Optimization:
        # 1. Get neighbors of u: Nu
        # 2. Get neighbors of v: Nv
        # 3. We need a in Nu, c in Nv.
        # 4. b is a common neighbor of a and v? No, b is neighbor of a and v is neighbor of b. So b in N(a) and (b,v) in E.
        #    This means b is in N(a) and b is in N(v). So b is a common neighbor of a and v.
        # 5. d is in N(c) and (d,u) in E. So d is a common neighbor of c and u.
        
        # Let's refine:
        # Path: u - a - b - v - c - d - u
        # a in N(u)
        # b in N(a) AND b in N(v)  => b in N(a) ∩ N(v)
        # c in N(v)
        # d in N(c) AND d in N(u)  => d in N(c) ∩ N(u)
        
        # Also need:
        # a != v, a != c, a != d (a is distinct from u, v, c, d)
        # b != u, b != a, b != c, b != d (b is distinct from u, a, v, c, d)
        # c != u, c != a, c != b, c != d
        # d != u, d != a, d != b, d != c
        
        # Let's iterate a in N(u)
        for a in adj[u]:
            if a == v: continue
            # Find candidates for b: common neighbors of a and v
            # b must be distinct from u, a, v
            common_av = adj[a] & adj[v]
            for b in common_av:
                if b == u or b == a or b == v: continue
                # Now we have u-a-b-v.
                # We need c in N(v) and d in N(c) such that d in N(u).
                # c must be distinct from u, a, b, v.
                # d must be distinct from u, a, b, v, c.
                
                # Iterate c in N(v)
                for c in adj[v]:
                    if c == u or c == a or c == b or c == v: continue
                    # Find candidates for d: common neighbors of c and u
                    common_cu = adj[c] & adj[u]
                    for d in common_cu:
                        if d == u or d == a or d == b or d == v or d == c: continue
                        # If we found such a d, then u-a-b-v-c-d-u is a C6
                        return False
        return True

    best_edge_count = len(initial_edges) // 2
    best_adj = None
    best_edges_list = None
    
    # We will try multiple random orderings of the missing edges
    # to escape local optima.
    
    missing_edges = [e for e in all_edges if e not in initial_edges]
    
    # To ensure we don't spend too much time, we limit the number of restarts
    # or simply run one good greedy pass.
    # Given N=40, the graph is small. A single greedy pass with a good heuristic
    # (e.g. adding edges with lower degree sum first? or random) is fast.
    # Let's try 100 random restarts if time permits.
    
    max_restarts = 1000 # We will break early based on time
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        adj = [set() for _ in range(n)]
        current_edges = set(initial_edges)
        
        # Build adjacency from initial edges
        for u, v in current_edges:
            adj[u].add(v)
            adj[v].add(u)
            
        # Create a list of missing edges
        # We can re-calculate missing edges or just use the global list and filter
        # It's faster to just iterate through all possible edges and check if they are in current_edges
        # But current_edges changes.
        
        # Let's shuffle the missing edges for this restart
        # We need to be careful: if an edge is rejected, it stays rejected.
        # So we can just take the list of all non-initial edges, shuffle, and process.
        
        candidates = list(missing_edges)
        random.shuffle(candidates)
        
        for u, v in candidates:
            if (u, v) in current_edges:
                continue
            
            # Check validity
            if is_valid_edge(adj, u, v):
                # Add edge
                current_edges.add((u, v))
                current_edges.add((v, u))
                adj[u].add(v)
                adj[v].add(u)
                
                if len(current_edges) // 2 > best_edge_count:
                    best_edge_count = len(current_edges) // 2
                    best_adj = [s.copy() for s in adj]
                    best_edges_list = list(current_edges)
        
        # If we reached the theoretical max or a very high number, we might stop early?
        # Best known is 105. Max possible is 780.
        # If we hit 105, it's likely optimal or near optimal.
        if best_edge_count >= 105:
            break

    if best_edges_list is None:
        # Fallback to initial
        return [[u, v] for u, v in initial_edges if u < v]
    
    # Return only u < v to match format if needed, though task says [u, v]
    # "list of edges [u, v] of a simple graph" usually implies unordered, 
    # but let's stick to u < v for cleanliness, or just return as is.
    # The seed returned [[5*k+a, 5*k+b] ...] where a < b.
    # So we should return edges with u < v.
    
    final_edges = [[u, v] for u, v in best_edges_list if u < v]
    return final_edges

# EVOLVE-BLOCK-END
