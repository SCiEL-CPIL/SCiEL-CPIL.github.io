import random
import time
import itertools

def construct():
    n = 20
    # adj[u] is a set of neighbors
    adj = [set() for _ in range(n)]
    # common[u][v] is the number of common neighbors of u and v
    common = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        # Check 1: common neighbors of u and v must be <= 2
        if common[u][v] > 2:
            return False
        
        # Check 2: For all y in N(v), common[u][y] must be <= 1
        # Adding edge (u,v) increases common(u,y) by 1 for all y in N(v) \ {u}
        # and common(v,y) by 1 for all y in N(u) \ {v}
        # Actually, common(u,y) = |N(u) ∩ N(y)|.
        # New N(u) = N(u) ∪ {v}.
        # New common(u,y) = |N(u) ∩ N(y)| + (1 if v ∈ N(y) else 0).
        # If y ∈ N(v), then v ∈ N(y). So term is 1.
        # We need |N(u) ∩ N(y)| + 1 <= 2 => |N(u) ∩ N(y)| <= 1.
        for y in adj[v]:
            if y == u: continue
            if common[u][y] > 1:
                return False
        
        # Check 3: For all y in N(u), common[v][y] must be <= 1
        for y in adj[u]:
            if y == v: continue
            if common[v][y] > 1:
                return False
        
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        # Update common counts
        # For y in N(v) (excluding u): common[u][y] += 1
        for y in adj[v]:
            if y == u: continue
            common[u][y] += 1
            common[y][u] += 1
            
        # For y in N(u) (excluding v): common[v][y] += 1
        for y in adj[u]:
            if y == v: continue
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        # Revert updates
        for y in adj[v]: # adj[v] is now without u
            common[u][y] -= 1
            common[y][u] -= 1
        for y in adj[u]: # adj[u] is now without v
            common[v][y] -= 1
            common[y][v] -= 1

    def get_edge_count():
        return sum(len(s) for s in adj) // 2

    def build_clebsch():
        # Clebsch graph on vertices 0-15
        # Vertices are 4-bit strings.
        # Adjacency: Hamming distance 1 or 4.
        for u in range(16):
            for v in range(16):
                if u >= v:
                    continue
                # Calculate Hamming distance
                diff = u ^ v
                popcount = bin(diff).count('1')
                if popcount == 1 or popcount == 4:
                    add_edge(u, v)

    def build_initial_graph():
        # Build Clebsch graph
        build_clebsch()
        
        # Find all edges in Clebsch graph to use as neighbor sets for new vertices
        # We need 4 disjoint edges (cliques of size 2 in C16)
        # Since C16 is 1-regular? No, 5-regular.
        # We just need to pick 4 edges that are vertex-disjoint.
        edges_in_clebsch = [(u, v) for u in range(16) for v in adj[u] if u < v]
        
        # Greedily pick 4 disjoint edges
        used_vertices = set()
        selected_pairs = []
        for u, v in edges_in_clebsch:
            if u not in used_vertices and v not in used_vertices:
                selected_pairs.append((u, v))
                used_vertices.add(u)
                used_vertices.add(v)
                if len(selected_pairs) == 4:
                    break
            
        # If we didn't find 4 disjoint edges (unlikely for Clebsch), we can pick overlapping ones
        # but that might violate constraints if not careful. 
        # However, for now, let's assume we found 4.
        # Connect new vertices 16-19 to these pairs
        for i, (u, v) in enumerate(selected_pairs):
            new_v = 16 + i
            add_edge(new_v, u)
            add_edge(new_v, v)
            
        # Connect new vertices among themselves (K4)
        for i in range(4):
            for j in range(i + 1, 4):
                u = 16 + i
                v = 16 + j
                if can_add(u, v):
                    add_edge(u, v)

    def reset_graph():
        nonlocal adj, common
        adj = [set() for _ in range(n)]
        common = [[0] * n for _ in range(n)]

    def polish_graph():
        """Systematic pass trying all missing edges in order of feasibility."""
        added = 0
        for u in range(n):
            for v in range(u + 1, n):
                if v not in adj[u]:
                    if can_add(u, v):
                        add_edge(u, v)
                        added += 1
        return added

    start_time = time.time()
    time_limit = 115.0 # Safety margin for 120s limit
    
    # Build initial structure
    build_initial_graph()
    
    # Initial polish
    polish_graph()
    
    best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
    best_count = get_edge_count()
    
    # Local search with multiple restarts
    while time.time() - start_time < time_limit:
        # Try to add a random edge
        improved = False
        # Try many random additions
        for _ in range(200):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                break
        
        if improved:
            curr_count = get_edge_count()
            if curr_count > best_count:
                best_count = curr_count
                best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
        else:
            # Stuck. Try to remove an edge to allow new ones
            if get_edge_count() > 0:
                # Pick a random edge to remove
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if edges_list:
                    u, v = random.choice(edges_list)
                    remove_edge(u, v)
                    # Try to add edges again
                    for _ in range(200):
                        u2 = random.randint(0, n - 1)
                        v2 = random.randint(0, n - 1)
                        if can_add(u2, v2):
                            add_edge(u2, v2)
                            improved = True
                            break
                    if improved:
                        curr_count = get_edge_count()
                        if curr_count > best_count:
                            best_count = curr_count
                            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
            else:
                # Empty graph, restart
                reset_graph()
                build_initial_graph()
                polish_graph()
    
    # Final polish pass
    polish_graph()
    curr_count = get_edge_count()
    if curr_count > best_count:
        best_count = curr_count
        best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]

    # Final check: ensure the best_edges is valid
    if not best_edges:
        # Fallback to seed if something went wrong
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Verification step (optional but good for safety)
    adj_check = [set() for _ in range(n)]
    for u, v in best_edges:
        adj_check[u].add(v)
        adj_check[v].add(u)
    
    valid = True
    for u in range(n):
        for v in range(u + 1, n):
            common_count = len(adj_check[u] & adj_check[v])
            if common_count > 2:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
