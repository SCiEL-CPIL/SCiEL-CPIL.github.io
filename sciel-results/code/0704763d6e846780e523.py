import random
import time
import itertools

def construct():
    """
    Constructs a dense K2,3-free graph on 20 vertices.
    Uses greedy construction followed by local search with edge removals 
    to escape local maxima.
    """
    n = 20
    # Adjacency matrix for O(1) lookup and easy updates
    adj = [[0] * n for _ in range(n)]
    
    # Helper to check if adding edge (u, v) is valid
    def is_valid_add(u, v):
        # 1. Check pair (u, v)
        common_uv = 0
        for w in range(n):
            if adj[u][w] and adj[v][w]:
                common_uv += 1
                if common_uv > 2:
                    return False
        
        # 2. Check pairs (u, w) for w in N(v) and (v, w) for w in N(u)
        # When we add (u, v), u is added to N(v) and v is added to N(u).
        # For any w, if w is a neighbor of v, then v becomes a common neighbor of u and w.
        # So we check: for all w in N(v), |N(u) intersect N(w)| + (1 if v in N(w) else 0) <= 2
        # Note: v is in N(w) iff w is in N(v). So for all w in N(v), the condition is:
        # |N(u) intersect N(w)| + 1 <= 2  => |N(u) intersect N(w)| <= 1
        
        for w in range(n):
            if w == u or w == v:
                continue
            # Check pair (u, w)
            if adj[v][w]: # w is in N(v)
                common_uw = 0
                for x in range(n):
                    if adj[u][x] and adj[w][x]:
                        common_uw += 1
                        if common_uw > 1: # Since v will be added, total will be common_uw + 1
                            return False
            
            # Check pair (v, w)
            if adj[u][w]: # w is in N(u)
                common_vw = 0
                for x in range(n):
                    if adj[v][x] and adj[w][x]:
                        common_vw += 1
                        if common_vw > 1: # Since u will be added, total will be common_vw + 1
                            return False
                            
        return True

    def add_edge(u, v):
        adj[u][v] = 1
        adj[v][u] = 1

    def remove_edge(u, v):
        adj[u][v] = 0
        adj[v][u] = 0

    def count_edges():
        return sum(sum(row) for row in adj) // 2

    def get_edges():
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v]:
                    edges.append([u, v])
        return edges

    # Start with a random initial graph to break symmetry
    # Or start empty and greedy fill.
    # Let's try starting with a random sparse graph.
    best_edges = 0
    best_adj = [row[:] for row in adj]
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    # Greedy Initialization
    # Try to add all possible edges in random order
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if time.time() - start_time > timeout:
            break
        if not adj[u][v]:
            if is_valid_add(u, v):
                add_edge(u, v)
    
    if count_edges() > best_edges:
        best_edges = count_edges()
        best_adj = [row[:] for row in adj]

    # Local Search Phase
    # Randomly remove edges to allow new ones, then greedily add back.
    iterations = 0
    while time.time() - start_time < timeout:
        iterations += 1
        
        # Randomly remove 1-3 edges to escape local optimum
        num_removals = random.randint(1, 3)
        current_edges = get_edges()
        if not current_edges:
            break
            
        removed = random.sample(current_edges, min(num_removals, len(current_edges)))
        for u, v in removed:
            remove_edge(u, v)
            
        # Greedily add back edges
        # Try adding the removed edges first, then others
        # Shuffle all non-edges
        non_edges = [(i, j) for i in range(n) for j in range(i + 1, n) if not adj[i][j]]
        # Prioritize the ones we just removed? Or just shuffle all.
        random.shuffle(non_edges)
        
        for u, v in non_edges:
            if time.time() - start_time > timeout:
                break
            if not adj[u][v]:
                if is_valid_add(u, v):
                    add_edge(u, v)
                    
        current_count = count_edges()
        if current_count > best_edges:
            best_edges = current_count
            best_adj = [row[:] for row in adj]
            
    # Return the best graph found
    return get_edges_from_adj(best_adj)

def get_edges_from_adj(adj):
    n = len(adj)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if adj[u][v]:
                edges.append([u, v])
    return edges
