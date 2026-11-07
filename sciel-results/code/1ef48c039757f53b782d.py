import time

def construct():
    n = 15
    start_time = time.time()
    deadline = start_time + 85
    
    # Base: Complete Bipartite K_{7,8}
    # Part A: 0..6 (size 7)
    # Part B: 7..14 (size 8)
    base_edges = []
    for u in range(7):
        for v in range(7, 15):
            base_edges.append([u, v])
    
    def is_valid(edge_list):
        if not edge_list:
            return True
        adj = {i: set() for i in range(n)}
        deg = [0] * n
        for u, v in edge_list:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1

        for u, v in edge_list:
            for s in adj[u]:
                if s == v:
                    continue
                for t in adj[v]:
                    if t == u or t == s:
                        continue
                    if deg[s] == deg[t]:
                        return False
        return True

    # Try to add as many intra-part edges as possible
    # Intra-A edges: pairs from 0..6
    # Intra-B edges: pairs from 7..14
    
    # Start with base
    best_edges = list(base_edges)
    best_count = len(base_edges)
    
    # Generate all possible intra-part edges
    intra_a_edges = []
    for i in range(7):
        for j in range(i+1, 7):
            intra_a_edges.append([i, j])
    
    intra_b_edges = []
    for i in range(7, 15):
        for j in range(i+1, 15):
            intra_b_edges.append([i, j])
    
    all_intra = intra_a_edges + intra_b_edges
    
    # Greedy: try adding each edge if it keeps the graph valid
    # Sort by potential impact (edges within same part first)
    for edge in all_intra:
        if time.time() > deadline - 5:
            break
        candidate = best_edges + [edge]
        if is_valid(candidate):
            best_edges = candidate
            best_count = len(candidate)
    
    # Try local improvement: remove one edge, add another
    if time.time() < deadline - 10:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(len(best_edges)):
                if time.time() > deadline - 5:
                    break
                # Try removing edge i and adding a new intra-part edge
                for new_edge in all_intra:
                    if new_edge in best_edges:
                        continue
                    # Check if removing best_edges[i] and adding new_edge improves
                    if best_edges[i] in new_edge or new_edge in best_edges[i]:
                        continue
                    candidate = [e for j, e in enumerate(best_edges) if j != i] + [new_edge]
                    if len(candidate) > best_count and is_valid(candidate):
                        best_edges = candidate
                        best_count = len(candidate)
                        improved = True
                        break
    
    # Final validation
    if not is_valid(best_edges):
        # Fallback to K_{7,8} if validation fails
        return [[u, v] for u in range(7) for v in range(7, 15)]
        
    return best_edges
