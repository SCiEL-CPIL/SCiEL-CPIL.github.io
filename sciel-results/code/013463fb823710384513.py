import time
import random
import sys

def construct():
    n = 16
    deadline = time.time() + 115  # Leave 35s buffer for safety
    
    # Helper: Check if graph is 3-colorable using bitmask MRV
    def is_3_colorable(adj):
        colors = [-1] * n
        remaining = (1 << n) - 1  # bitmask of uncolored vertices
        
        def backtrack(remaining_mask):
            if remaining_mask == 0:
                return True
            
            # MRV: pick vertex with fewest available colors
            best_v = -1
            best_avail = 7  # 111 in binary (3 colors)
            best_used = 0
            
            for v in range(n):
                if (remaining_mask >> v) & 1:
                    used = 0
                    for u in range(n):
                        if (adj[v] >> u) & 1:
                            cu = colors[u]
                            if cu != -1:
                                used |= (1 << cu)
                    avail = 7 ^ used
                    if avail == 0:
                        return False
                    if bin(avail).count('1') < bin(best_avail).count('1'):
                        best_v = v
                        best_avail = avail
                        best_used = used
                        if bin(avail).count('1') == 1:
                            break
            
            if best_v == -1:
                return False
            
            for c in range(3):
                if not (best_used & (1 << c)):
                    colors[best_v] = c
                    if backtrack(remaining_mask & ~(1 << best_v)):
                        return True
                    colors[best_v] = -1
            return False
        
        return backtrack((1 << n) - 1)

    # Helper: Check if adding edge (u,v) creates a triangle
    def creates_triangle(adj, u, v):
        return (adj[u] & adj[v]) != 0

    # Helper: Generate a random maximal triangle-free graph
    def random_maximal_tf():
        adj = [0] * n
        edges = []
        possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            if not creates_triangle(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                edges.append([u, v])
        return adj, edges

    # Helper: Local search with edge swaps
    def local_search(adj, edges):
        current_count = len(edges)
        edge_set = set()
        for e in edges:
            edge_set.add((min(e[0], e[1]), max(e[0], e[1])))
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            # Try removing a random edge and adding back optimally
            if not edges:
                break
            idx = random.randint(0, len(edges)-1)
            u, v = edges[idx]
            
            # Remove
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            edges.pop(idx)
            edge_set.discard((min(u, v), max(u, v)))
            
            # Try to add back all possible edges greedily
            possible_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
            random.shuffle(possible_edges)
            
            for u2, v2 in possible_edges:
                if (u2, v2) not in edge_set and not creates_triangle(adj, u2, v2):
                    adj[u2] |= (1 << v2)
                    adj[v2] |= (1 << u2)
                    edges.append([u2, v2])
                    edge_set.add((u2, v2))
                    improved = True
            
            # If we didn't improve, revert would be ideal but complex; just continue
            if len(edges) <= current_count and not improved:
                break
        
        return adj, edges

    # Initialize with Clebsch graph
    vertices = []
    for i in range(32):
        if bin(i).count('1') % 2 == 0:
            vertices.append(i)
    
    clebsch_adj = [0] * n
    clebsch_edges = []
    for u_idx in range(n):
        for v_idx in range(u_idx + 1, n):
            xor_val = vertices[u_idx] ^ vertices[v_idx]
            if bin(xor_val).count('1') == 4:
                clebsch_adj[u_idx] |= (1 << v_idx)
                clebsch_adj[v_idx] |= (1 << u_idx)
                clebsch_edges.append([u_idx, v_idx])
                
    best_edges_count = len(clebsch_edges)
    best_edges = clebsch_edges
    best_adj = clebsch_adj

    # Verify baseline
    if not is_3_colorable(best_adj):
        pass
    else:
        best_edges_count = 0
        best_edges = []
        best_adj = [0]*n

    # Main search loop
    iterations = 0
    while time.time() < deadline:
        iterations += 1
        
        # Alternate between random restarts and improving current best
        if iterations % 3 == 0 and best_adj is not None and best_edges_count > 0:
            # Try to improve the current best
            adj = best_adj[:]
            edges = [e[:] for e in best_edges]
            adj, edges = local_search(adj, edges)
            
            if len(edges) > best_edges_count:
                if not is_3_colorable(adj):
                    best_edges_count = len(edges)
                    best_edges = edges
                    best_adj = adj
        else:
            # Random restart
            adj, edges = random_maximal_tf()
            adj, edges = local_search(adj, edges)
            
            if len(edges) > best_edges_count:
                if not is_3_colorable(adj):
                    best_edges_count = len(edges)
                    best_edges = edges
                    best_adj = adj
    
    return best_edges
