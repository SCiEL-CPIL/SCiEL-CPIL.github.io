"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    n = 20
    # State: adj[u][v] is color (0-19) or -1 if no edge
    adj = [[-1 for _ in range(n)] for _ in range(n)]
    
    def count_violations_for_edge(u, v, c):
        """
        Count non-rainbow 4-cycles involving edge (u,v) if it had color c.
        """
        violations = 0
        for w in range(n):
            if w == u or w == v: continue
            if adj[v][w] == -1: continue
            c_vw = adj[v][w]
            for x in range(n):
                if x == u or x == v or x == w: continue
                if adj[u][x] == -1: continue
                if adj[w][x] != -1:
                    c_wx = adj[w][x]
                    c_xu = adj[u][x]
                    if len({c, c_vw, c_wx, c_xu}) != 4:
                        violations += 1
        return violations

    def total_violations():
        total_v = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] != -1:
                    total_v += count_violations_for_edge(u, v, adj[u][v])
        return total_v // 4

    def num_edges():
        count = 0
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u][v] != -1:
                    count += 1
        return count

    start_time = time.time()
    time_limit = 115.0
    
    best_adj = [row[:] for row in adj]
    best_edges = 0
    
    # Simulated Annealing
    T = 50.0
    cooling = 0.995
    
    # Initialize with a random sparse graph
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < 0.3:
                c = random.randint(0, 19)
                adj[u][v] = c
                adj[v][u] = c
                
    current_edges = num_edges()
    current_violations = total_violations()
    
    if current_violations == 0 and current_edges > best_edges:
        best_edges = current_edges
        best_adj = [row[:] for row in adj]

    while time.time() - start_time < time_limit:
        if T < 0.1:
            break
            
        # Pick a random move
        move_type = random.randint(0, 2)
        u, v = -1, -1
        old_c = -1
        new_c = -1
        is_add = False
        is_remove = False
        is_change = False
        
        if move_type == 0:
            # Add edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and adj[u][v] == -1:
                    break
            new_c = random.randint(0, 19)
            is_add = True
        elif move_type == 1:
            # Remove edge
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and adj[u][v] != -1:
                    break
            old_c = adj[u][v]
            is_remove = True
        else:
            # Change color
            while True:
                u = random.randint(0, n-1)
                v = random.randint(0, n-1)
                if u != v and adj[u][v] != -1:
                    break
            old_c = adj[u][v]
            new_c = random.randint(0, 19)
            if new_c == old_c:
                continue
            is_change = True
            
        # Calculate delta violations
        delta_v = 0
        if is_add:
            # Violations added
            delta_v = count_violations_for_edge(u, v, new_c)
            delta_e = 1
        elif is_remove:
            # Violations removed
            delta_v = -count_violations_for_edge(u, v, old_c)
            delta_e = -1
        else:
            # Violations changed
            delta_v = count_violations_for_edge(u, v, new_c) - count_violations_for_edge(u, v, old_c)
            delta_e = 0
            
        new_v = current_violations + delta_v
        new_e = current_edges + delta_e
        
        # Energy: -1000 * violations + edges
        # Delta Energy = -1000 * delta_v + delta_e
        delta_energy = -1000 * delta_v + delta_e
        
        # Acceptance
        if delta_energy < 0 or (T > 0 and random.random() < math.exp(-delta_energy / T)):
            # Apply move
            if is_add:
                adj[u][v] = new_c
                adj[v][u] = new_c
            elif is_remove:
                adj[u][v] = -1
                adj[v][u] = -1
            else:
                adj[u][v] = new_c
                adj[v][u] = new_c
                
            current_violations = new_v
            current_edges = new_e
            
            if current_violations == 0 and current_edges > best_edges:
                best_edges = current_edges
                best_adj = [row[:] for row in adj]
                
        T *= cooling

    coloured_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if best_adj[u][v] != -1:
                coloured_edges.append([u, v, best_adj[u][v]])
    
    return coloured_edges
# EVOLVE-BLOCK-END
