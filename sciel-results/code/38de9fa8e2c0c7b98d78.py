import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    n = 18
    
    # Helper: Check if adding edge (u, v) creates a K4
    # adj is a list of bitmasks
    def can_add_edge(adj, u, v):
        if adj[u] & (1 << v):
            return True # Edge already exists
        common = adj[u] & adj[v]
        # Check if there is an edge between any two vertices in common
        # Iterate through vertices in common
        c = common
        while c:
            w = (c & -c).bit_length() - 1
            c &= c - 1
            # Check if w has any neighbor in common other than itself
            # Actually, if adj[w] has any bit set that is in common (and not w), then edge exists
            if adj[w] & (common & ~(1 << w)):
                return False
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    # Helper: Find all triangles and return their bitmasks
    def get_triangles(adj):
        triangles = []
        for u in range(n):
            # Only consider v > u
            v_mask = adj[u] & ~((1 << (u + 1)) - 1)
            while v_mask:
                v = (v_mask & -v_mask).bit_length() - 1
                v_mask &= v_mask - 1
                # Common neighbors of u and v, with index > v
                common = adj[u] & adj[v] & ~((1 << (v + 1)) - 1)
                while common:
                    w = (common & -common).bit_length() - 1
                    common &= common - 1
                    mask = (1 << u) | (1 << v) | (1 << w)
                    triangles.append(mask)
        return triangles

    # Helper: Evaluate the graph (max size of triangle-free induced subgraph)
    def evaluate(adj):
        triangles = get_triangles(adj)
        if not triangles:
            return n
        
        # Check subsets from largest to smallest
        for k in range(n, 0, -1):
            for subset in combinations(range(n), k):
                mask = 0
                for v in subset:
                    mask |= (1 << v)
                
                is_tf = True
                for t_mask in triangles:
                    if (mask & t_mask) == t_mask:
                        is_tf = False
                        break
                if is_tf:
                    return k
        return 0

    # Initial Graph: Random Greedy K4-free
    def generate_random_k4_free():
        adj = [0] * n
        edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        random.shuffle(edges)
        for u, v in edges:
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
        return adj

    # Generate initial candidate
    best_adj = generate_random_k4_free()
    best_score = evaluate(best_adj)
    
    current_adj = best_adj[:]
    current_score = best_score
    
    # Local Search
    t = 1.0
    cooling = 0.99
    
    while time.time() < deadline:
        # Try to add a random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        
        if not (current_adj[u] & (1 << v)):
            if can_add_edge(current_adj, u, v):
                # Tentatively add
                old_adj_u = current_adj[u]
                old_adj_v = current_adj[v]
                add_edge(current_adj, u, v)
                
                new_score = evaluate(current_adj)
                
                if new_score < current_score:
                    best_adj = current_adj[:]
                    best_score = new_score
                    current_score = new_score
                else:
                    # Simulated Annealing: accept worse with prob exp(-delta/t)
                    # Here delta = current_score - new_score (positive if worse)
                    delta = current_score - new_score
                    if delta > 0:
                        if random.random() < pow(2.71828, -delta / t):
                            current_score = new_score
                        else:
                            # Revert
                            current_adj[u] = old_adj_u
                            current_adj[v] = old_adj_v
                    else:
                        # Revert
                        current_adj[u] = old_adj_u
                        current_adj[v] = old_adj_v
                
                t *= cooling

    # Convert best_adj to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u] & (1 << v):
                edges.append([u, v])
                
    return edges
