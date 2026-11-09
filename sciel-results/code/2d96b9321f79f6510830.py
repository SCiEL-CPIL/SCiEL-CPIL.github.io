import random
import time
import math

def construct():
    n = 20
    adj = [set() for _ in range(n)]
    common = [[0] * n for _ in range(n)]
    
    def can_add(u, v):
        if u == v:
            return False
        if v in adj[u]:
            return False
        if common[u][v] > 2:
            return False
        
        for y in adj[v]:
            if common[u][y] > 1:
                return False
        
        for y in adj[u]:
            if common[v][y] > 1:
                return False
        
        return True

    def add_edge(u, v):
        adj[u].add(v)
        adj[v].add(u)
        for y in adj[v]:
            common[u][y] += 1
            common[y][u] += 1
            
        for y in adj[u]:
            common[v][y] += 1
            common[y][v] += 1

    def remove_edge(u, v):
        adj[u].remove(v)
        adj[v].remove(u)
        for y in adj[v]:
            common[u][y] -= 1
            common[y][u] -= 1
        for y in adj[u]:
            common[v][y] -= 1
            common[y][v] -= 1

    def get_edge_count():
        return sum(len(s) for s in adj) // 2

    start_time = time.time()
    time_limit = 140.0
    best_edges = []
    best_count = 0
    
    def save_best():
        nonlocal best_count, best_edges
        curr_count = get_edge_count()
        if curr_count > best_count:
            best_count = curr_count
            best_edges = [[u, v] for u in range(n) for v in adj[u] if u < v]

    # Simulated Annealing with edge swaps
    # Use time-based cooling: T = T_min * (remaining/total)^alpha
    T_start = 15.0
    T_min = 0.05
    total_time = time_limit
    
    # Initial greedy fill
    for _ in range(3000):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if can_add(u, v):
            add_edge(u, v)
    
    save_best()

    current_count = get_edge_count()
    
    while time.time() - start_time < time_limit:
        elapsed = time.time() - start_time
        remaining = max(0, total_time - elapsed)
        if remaining <= 0:
            break
        # Time-based temperature
        T = T_min + (T_start - T_min) * (remaining / total_time) ** 0.8
        
        # Generate a move
        move_type = random.random()
        if move_type < 0.5:
            # Add edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v or v in adj[u]:
                continue
            if can_add(u, v):
                add_edge(u, v)
                current_count += 1
                save_best()
        elif move_type < 0.8:
            # Remove edge
            if current_count == 0:
                continue
            edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if not edges_list:
                continue
            u, v = random.choice(edges_list)
            remove_edge(u, v)
            new_count = current_count - 1
            # Accept with probability exp(-delta/T) where delta = -1
            # prob = exp(1/T)
            if random.random() < math.exp(1.0 / T):
                current_count = new_count
            else:
                # Reject, revert
                add_edge(u, v)
        else:
            # Edge swap: remove one edge, add another (net zero change in count)
            # This helps escape local optima
            if current_count == 0:
                continue
            edges_list = [(u, v) for u in range(n) for v in adj[u] if u < v]
            if not edges_list:
                continue
            u1, v1 = random.choice(edges_list)
            remove_edge(u1, v1)
            
            # Try to add a random edge
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 != v2 and v2 not in adj[u2] and can_add(u2, v2):
                add_edge(u2, v2)
                # Net change: 0 (removed 1, added 1)
                # Accept with some probability to allow restructuring
                if random.random() < math.exp(0.5 / T):
                    # Keep the swap
                    pass
                else:
                    # Revert
                    remove_edge(u2, v2)
                    add_edge(u1, v1)
            else:
                # Couldn't add, revert
                add_edge(u1, v1)
    
    # Final polish: exhaustive greedy add
    for _ in range(1000):
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if can_add(u, v):
            add_edge(u, v)
    
    # More aggressive final polish: try all pairs
    for u in range(n):
        for v in range(u + 1, n):
            if v not in adj[u] and can_add(u, v):
                add_edge(u, v)
    
    save_best()

    if not best_edges:
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    # Verification
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
        edges = []
        for k in range(5):
            edges += [[4 * k + a, 4 * k + b] for a in range(4) for b in range(a + 1, 4)]
        return edges

    return best_edges
