import time
import random
from itertools import combinations

def construct():
    """
    Find a triangle-free graph on 16 vertices with chromatic number >= 4,
    maximizing the number of edges.
    """
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n = 16
    best_edges = None
    best_score = -1
    
    # Grotzsch graph: 11 vertices, 20 edges, triangle-free, 4-chromatic
    # This is the Mycielski graph of C5
    grotzsch_edges = []
    # Original C5
    for i in range(5):
        grotzsch_edges.append([i, (i+1)%5])
    # Mycielski vertices 5..9
    grotzsch_edges.append([5, 1])
    grotzsch_edges.append([5, 4])
    grotzsch_edges.append([6, 0])
    grotzsch_edges.append([6, 2])
    grotzsch_edges.append([7, 1])
    grotzsch_edges.append([7, 3])
    grotzsch_edges.append([8, 2])
    grotzsch_edges.append([8, 4])
    grotzsch_edges.append([9, 3])
    grotzsch_edges.append([9, 0])
    # 10 connected to 5,6,7,8,9
    for i in range(5):
        grotzsch_edges.append([10, 5+i])
    
    # Build adjacency matrix for Grotzsch
    base_adj = [[0]*n for _ in range(n)]
    for u, v in grotzsch_edges:
        base_adj[u][v] = 1
        base_adj[v][u] = 1
    
    def is_triangle_free(adj):
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        return True
    
    def is_3_colorable_bitmask(adj):
        """
        Fast backtracking 3-coloring using bitmasks.
        Returns True if 3-colorable, False otherwise.
        """
        # Precompute adjacency as bitmasks
        adj_mask = [0] * n
        for i in range(n):
            mask = 0
            row = adj[i]
            for j in range(n):
                if row[j]:
                    mask |= (1 << j)
            adj_mask[i] = mask
        
        color = [-1] * n
        
        # Order vertices by degree (descending) for better pruning
        order = sorted(range(n), key=lambda v: bin(adj_mask[v]).count('1'), reverse=True)
        
        def can_color(idx):
            if idx == n:
                return True
            
            v = order[idx]
            # Compute forbidden colors from already-colored neighbors
            forbidden = 0
            mask = adj_mask[v]
            for u in order[:idx]:
                if color[u] != -1 and (mask >> u) & 1:
                    forbidden |= (1 << color[u])
            
            avail = 7 & ~forbidden
            if avail == 0:
                return False
            
            # Try each available color
            while avail:
                c_bit = avail & (-avail)
                avail ^= c_bit
                c = c_bit.bit_length() - 1
                color[v] = c
                if can_color(idx + 1):
                    return True
                color[v] = -1
            
            return False
        
        return can_color(0)
    
    def edge_count(adj):
        return sum(adj[i][j] for i in range(n) for j in range(i+1, n))
    
    # Strategy: Start with Grotzsch + 5 isolated vertices, then try to add edges
    # Key insight: The Grotzsch graph has a maximum independent set of size 5
    # If we connect the 5 new vertices to this independent set, we add 25 edges
    # and maintain triangle-freeness
    
    def find_max_independent_set(adj, vertices):
        """Find maximum independent set using backtracking."""
        best_set = []
        
        def backtrack(remaining, current_set):
            nonlocal best_set
            if len(current_set) > len(best_set):
                best_set = current_set[:]
            
            if not remaining:
                return
            
            v = remaining[0]
            rest = remaining[1:]
            
            # Option 1: Include v
            new_remaining = [u for u in rest if not adj[v][u]]
            backtrack(new_remaining, current_set + [v])
            
            # Option 2: Exclude v
            backtrack(rest, current_set)
        
        backtrack(list(vertices), [])
        return best_set
    
    # Grotzsch vertices are 0..10
    grotzsch_vertices = list(range(11))
    max_iset = find_max_independent_set(base_adj, grotzsch_vertices)
    
    # The 5 new vertices are 11..15
    new_vertices = list(range(11, 16))
    
    # Connect each new vertex to the maximum independent set
    adj = [row[:] for row in base_adj]
    for nv in new_vertices:
        for iv in max_iset:
            adj[nv][iv] = 1
            adj[iv][nv] = 1
    
    # Check if this is valid
    if is_triangle_free(adj) and not is_3_colorable_bitmask(adj):
        m = edge_count(adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    def get_non_edges(adj):
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    non_edges.append((i, j))
        return non_edges
    
    def get_edges(adj):
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    edges.append((i, j))
        return edges
    
    def can_add_edge(adj, u, v):
        """Check if edge (u,v) can be added without creating a triangle."""
        if adj[u][v]:
            return False
        for k in range(n):
            if k != u and k != v and adj[u][k] and adj[v][k]:
                return False
        return True
    
    def try_add_edge(adj, u, v):
        """Try adding edge (u,v) while maintaining triangle-freeness and 4-chromaticity."""
        if not can_add_edge(adj, u, v):
            return None
        
        adj[u][v] = 1
        adj[v][u] = 1
        
        if not is_3_colorable_bitmask(adj):
            return adj
        else:
            adj[u][v] = 0
            adj[v][u] = 0
            return None
    
    # Systematic edge addition: try all non-edges in order
    improved = True
    while improved and time.time() < deadline:
        improved = False
        non_edges = get_non_edges(adj)
        random.shuffle(non_edges)
        for i, j in non_edges:
            if time.time() > deadline:
                break
            if adj[i][j]:
                continue
            new_adj = try_add_edge(adj, i, j)
            if new_adj is not None:
                adj = new_adj
                improved = True
        
        m = edge_count(adj)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
    
    # Try edge swaps to escape local optima
    if time.time() < deadline:
        edges = get_edges(adj)
        non_edges = get_non_edges(adj)
        random.shuffle(edges)
        random.shuffle(non_edges)
        for rem_e in edges:
            if time.time() > deadline:
                break
            u, v = rem_e
            if not adj[u][v]:
                continue
            for add_e in non_edges:
                if time.time() > deadline:
                    break
                a, b = add_e
                if adj[a][b]:
                    continue
                
                # Temporarily remove edge (u,v)
                adj[u][v] = 0
                adj[v][u] = 0
                
                # Check if we can add edge (a,b)
                if can_add_edge(adj, a, b):
                    adj[a][b] = 1
                    adj[b][a] = 1
                    
                    if not is_3_colorable_bitmask(adj):
                        # Success
                        m = edge_count(adj)
                        if m > best_score:
                            best_score = m
                            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj[i][j]]
                        break
                    else:
                        adj[a][b] = 0
                        adj[b][a] = 0
                
                # Revert removal
                adj[u][v] = 1
                adj[v][u] = 1
            
            if time.time() > deadline:
                break
    
    # Random restarts with different initial configurations
    random.seed(42)
    
    def generate_random_triangle_free(density):
        adj = [[0]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < density:
                    if can_add_edge(adj, i, j):
                        adj[i][j] = 1
                        adj[j][i] = 1
        return adj
    
    for restart in range(200):
        if time.time() > deadline:
            break
        
        density = 0.2 + 0.5 * random.random()
        adj_try = generate_random_triangle_free(density)
        
        if is_3_colorable_bitmask(adj_try):
            continue
        
        # Local search from this starting point
        improved = True
        while improved and time.time() < deadline:
            improved = False
            non_edges = get_non_edges(adj_try)
            random.shuffle(non_edges)
            for i, j in non_edges:
                if time.time() > deadline:
                    break
                if adj_try[i][j]:
                    continue
                new_adj = try_add_edge(adj_try, i, j)
                if new_adj is not None:
                    adj_try = new_adj
                    improved = True
            
            m = edge_count(adj_try)
            if m > best_score:
                best_score = m
                best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj_try[i][j]]
        
        if time.time() > deadline:
            break
    
    # Final exhaustive polish: try all non-edges one more time
    if best_edges is not None and time.time() < deadline:
        adj_final = [[0]*n for _ in range(n)]
        for u, v in best_edges:
            adj_final[u][v] = 1
            adj_final[v][u] = 1
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            non_edges = get_non_edges(adj_final)
            for i, j in non_edges:
                if time.time() > deadline:
                    break
                if adj_final[i][j]:
                    continue
                new_adj = try_add_edge(adj_final, i, j)
                if new_adj is not None:
                    adj_final = new_adj
                    improved = True
        
        m = edge_count(adj_final)
        if m > best_score:
            best_score = m
            best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if adj_final[i][j]]
    
    # Fallback: return Grotzsch + isolated
    if best_edges is None:
        best_edges = [[i, j] for i in range(n) for j in range(i+1, n) if base_adj[i][j]]
    
    return best_edges
