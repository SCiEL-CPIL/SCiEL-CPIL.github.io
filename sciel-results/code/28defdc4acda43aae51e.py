import random
import time

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
        # Because adding v to N(u) means v becomes a common neighbor of u and y if y is neighbor of v.
        # Wait, if y in N(v), then v in N(y).
        # New common(u, y) = old common(u, y) + (1 if v in N(y) else 0).
        # Since y in N(v) => v in N(y), the term is 1.
        # So we need old common(u, y) + 1 <= 2 => old common(u, y) <= 1.
        for y in adj[v]:
            if common[u][y] > 1:
                return False
        
        # Check 3: For all y in N(u), common[v][y] must be <= 1
        for y in adj[u]:
            if common[v][y] > 1:
                return False
        
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        # Update common counts
        # common[u][v] doesn't change (as derived)
        # For y in N(v): common[u][y] increases by 1 (because v is now in N(u) and v is in N(y))
        # Wait, common[u][y] = |N(u) ∩ N(y)|.
        # N(u) gained v. v is in N(y) (since y in N(v)).
        # So intersection gains v. Count +1.
        for y in adj[v]:
            common[u][y] += 1
            common[y][u] += 1
            # Note: y != u, y != v.
            # If y == u, u not in N(v).
            # If y == v, v not in N(v).
            
        # For y in N(u): common[v][y] increases by 1
        for y in adj[u]:
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

    start_time = time.time()
    time_limit = 110.0 # Safety margin
    best_edges = []
    best_count = 0
    
    # Initial greedy fill
    for _ in range(1000):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if can_add(u, v):
            add_edge(u, v)
            
    best_count = get_edge_count()
    if best_count > 0:
        best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]

    # Local search
    while time.time() - start_time < time_limit:
        # Try to add a random edge
        improved = False
        for _ in range(100):
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if can_add(u, v):
                add_edge(u, v)
                improved = True
                break
        
        if not improved:
            # Stuck. Try to remove an edge to allow new ones
            if get_edge_count() > 0:
                # Pick a random edge to remove
                # Flatten edges for random pick
                # This is O(E)
                edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
                if edges_list:
                    u, v = random.choice(edges_list)
                    remove_edge(u, v)
                    # Try to add edges again
                    for _ in range(100):
                        u2 = random.randint(0, n - 1)
                        v2 = random.randint(0, n - 1)
                        if can_add(u2, v2):
                            add_edge(u2, v2)
                            improved = True
                            break
                    if improved:
                        # Check if we improved the best
                        curr_count = get_edge_count()
                        if curr_count > best_count:
                            best_count = curr_count
                            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]
            else:
                # Empty graph, restart
                pass

    # Final check: ensure the best_edges is valid
    # Re-construct graph from best_edges to verify
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
