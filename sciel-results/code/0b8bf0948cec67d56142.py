import time
import sys

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Start with K_7,8
    # A: 0-6, B: 7-14
    edges = set()
    for u in range(7):
        for v in range(7, 15):
            edges.add((u, v))
    
    # Helper to check validity - optimized with early exit
    def is_valid(edge_list):
        if len(edge_list) == 0:
            return True
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all simple paths of length 3: s-u-v-t
        # For each edge (u,v), for each s in N(u)\{v}, for each t in N(v)\{u,s}
        # Check deg[s] != deg[t]
        for u, v in edge_list:
            for s in adj[u]:
                if s == v:
                    continue
                ds = deg[s]
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if ds == deg[t]:
                        return False
        return True

    # Candidate edges to add
    all_edges = set()
    for i in range(n):
        for j in range(i+1, n):
            all_edges.add((i, j))
    
    missing_edges = list(all_edges - edges)
    
    # Prioritize: try edges that are less likely to cause violations first
    # Intra-part edges in the larger part might be slightly less constrained
    def edge_priority(e):
        u, v = e
        if u < 7 and v < 7:
            return 0  # Intra A (size 7)
        elif u >= 7 and v >= 7:
            return 1  # Intra B (size 8)
        else:
            return 2  # Cross edge (already present, shouldn't be here)

    missing_edges.sort(key=edge_priority)
    
    current_edges = list(edges)
    
    # Greedily add edges, with time check
    for e in missing_edges:
        if time.time() > deadline:
            break
        next_edges = current_edges + [e]
        if is_valid(next_edges):
            current_edges = next_edges
    
    # Convert to required format with exact type checking
    final_edges = []
    for u, v in current_edges:
        final_edges.append([int(u), int(v)])
    
    return final_edges
