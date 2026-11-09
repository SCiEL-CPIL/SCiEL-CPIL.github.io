import random
import time
import math

def construct():
    """
    Find a K2,3-free graph on 20 vertices maximizing edges.
    
    Strategy:
    1. Initialize with 5 disjoint K4s (30 edges).
    2. Use Simulated Annealing with:
       - Add moves (70%)
       - Swap moves (20%): remove one edge, add one edge
       - Remove moves (10%)
    3. Fast validity checks using bitsets.
    4. Track best solution found.
    """
    n = 20
    start_time = time.time()
    time_limit = 140.0  # Leave margin for evaluator overhead
    
    # Initialize with 5 disjoint K4s
    adj = [0] * n
    for k in range(5):
        base = 4 * k
        for a in range(4):
            for b in range(a + 1, 4):
                u = base + a
                v = base + b
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
    
    def count_edges(adj_list):
        return sum(bin(x).count('1') for x in adj_list) // 2
    
    current_edges = count_edges(adj)
    best_edges = current_edges
    best_adj = adj[:]
    
    def can_add_edge(adj_list, u, v):
        """Check if adding edge (u,v) maintains K2,3-free property."""
        if adj_list[u] & (1 << v):
            return True
        
        # 1. Common neighbors of u and v must be <= 2
        common_uv = adj_list[u] & adj_list[v]
        if bin(common_uv).count('1') > 2:
            return False
        
        # 2. For w in N(u), common neighbors of w and v must be <= 1
        #    (since u becomes a common neighbor of w and v)
        neighbors_u = adj_list[u]
        temp = neighbors_u
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == v:
                continue
            common_wv = adj_list[w] & adj_list[v]
            if bin(common_wv).count('1') > 1:
                return False
        
        # 3. For w in N(v), common neighbors of w and u must be <= 1
        neighbors_v = adj_list[v]
        temp = neighbors_v
        while temp:
            lsb = temp & -temp
            w = lsb.bit_length() - 1
            temp -= lsb
            if w == u:
                continue
            common_wu = adj_list[w] & adj_list[u]
            if bin(common_wu).count('1') > 1:
                return False
        
        return True
    
    def add_edge(adj_list, u, v):
        adj_list[u] |= (1 << v)
        adj_list[v] |= (1 << u)
    
    def remove_edge(adj_list, u, v):
        adj_list[u] &= ~(1 << v)
        adj_list[v] &= ~(1 << u)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9995
    
    while T > T_min and (time.time() - start_time < time_limit):
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to add a random non-edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if adj[u] & (1 << v):
                continue
            
            if can_add_edge(adj, u, v):
                add_edge(adj, u, v)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                # Invalid. Try to fix by removing a conflicting edge.
                incident_edges = []
                temp = adj[u]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    incident_edges.append((u, w))
                    temp -= lsb
                temp = adj[v]
                while temp:
                    lsb = temp & -temp
                    w = lsb.bit_length() - 1
                    if w != u:
                        incident_edges.append((v, w))
                    temp -= lsb
                
                if incident_edges:
                    rem_u, rem_v = random.choice(incident_edges)
                    remove_edge(adj, rem_u, rem_v)
                    current_edges -= 1
                    
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
                    else:
                        add_edge(adj, rem_u, rem_v)
                        current_edges += 1
        
        elif move_type < 0.9:
            # Try to remove a random edge
            u = random.randint(0, n - 1)
            v = random.randint(0, n - 1)
            if u == v:
                continue
            if adj[u] & (1 << v):
                remove_edge(adj, u, v)
                current_edges -= 1
        
        else:
            # Swap: Remove random edge, add random non-edge
            u1 = random.randint(0, n - 1)
            v1 = random.randint(0, n - 1)
            if u1 == v1 or not (adj[u1] & (1 << v1)):
                continue
                
            u2 = random.randint(0, n - 1)
            v2 = random.randint(0, n - 1)
            if u2 == v2 or (adj[u2] & (1 << v2)):
                continue
            
            remove_edge(adj, u1, v1)
            current_edges -= 1
            
            if can_add_edge(adj, u2, v2):
                add_edge(adj, u2, v2)
                current_edges += 1
                if current_edges > best_edges:
                    best_edges = current_edges
                    best_adj = adj[:]
            else:
                add_edge(adj, u1, v1)
                current_edges += 1
        
        T *= cooling_rate
        
        if time.time() - start_time > time_limit:
            break
    
    # Final greedy pass
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for u in range(n):
            if time.time() - start_time > time_limit:
                break
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    if can_add_edge(adj, u, v):
                        add_edge(adj, u, v)
                        current_edges += 1
                        improved = True
                        if current_edges > best_edges:
                            best_edges = current_edges
                            best_adj = adj[:]
    
    # Extract edges from best_adj
    edges = []
    for u in range(n):
        temp = best_adj[u]
        while temp:
            lsb = temp & -temp
            v = lsb.bit_length() - 1
            if u < v:
                edges.append([u, v])
            temp -= lsb
            
    return edges
