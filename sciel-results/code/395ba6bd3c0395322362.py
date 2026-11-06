"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit
    
    n = 20
    
    def count_bad_cycles(adj, u, v):
        """Count non-rainbow 4-cycles that include edge (u,v)."""
        count = 0
        cu = adj[u]
        cv = adj[v]
        for x in range(n):
            if x == u or x == v:
                continue
            if cv[x] == -1:
                continue
            cx = cv[x]
            for y in range(n):
                if y == u or y == v or y == x:
                    continue
                if cu[y] == -1:
                    continue
                if adj[x][y] == -1:
                    continue
                cy = cu[y]
                cx_y = adj[x][y]
                c_uv = cu[v]
                # Check if all four colors are distinct
                if len({c_uv, cx, cx_y, cy}) < 4:
                    count += 1
        return count
    
    def total_bad_cycles(adj):
        """Count total non-rainbow 4-cycles (each counted once)."""
        total = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] == -1:
                    continue
                cu = adj[u]
                cv = adj[v]
                for x in range(n):
                    if x == u or x == v:
                        continue
                    if cv[x] == -1:
                        continue
                    cx = cv[x]
                    for y in range(n):
                        if y <= u or y == v or y == x:
                            continue
                        if cu[y] == -1:
                            continue
                        if adj[x][y] == -1:
                            continue
                        cy = cu[y]
                        cx_y = adj[x][y]
                        c_uv = cu[v]
                        if len({c_uv, cx, cx_y, cy}) < 4:
                            total += 1
        return total
    
    def is_valid(adj):
        return total_bad_cycles(adj) == 0
    
    def edge_count(adj):
        count = 0
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    count += 1
        return count
    
    def make_edges(adj):
        edges = []
        for u in range(n):
            for v in range(u+1, n):
                if adj[u][v] != -1:
                    edges.append([u, v, adj[u][v]])
        return edges
    
    best_edges = [[0, v, 0] for v in range(1, n)]
    best_count = len(best_edges)
    
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        # Initialize with complete graph
        adj = [[-1]*n for _ in range(n)]
        
        # Assign colors - use a structured approach
        strategy = attempt % 5
        
        if strategy == 0:
            # Random color assignment
            for u in range(n):
                for v in range(u+1, n):
                    c = random.randint(0, n-1)
                    adj[u][v] = c
                    adj[v][u] = c
        elif strategy == 1:
            # Color by (u+v) mod n
            for u in range(n):
                for v in range(u+1, n):
                    c = (u + v) % n
                    adj[u][v] = c
                    adj[v][u] = c
        elif strategy == 2:
            # Color by (u*v) mod n (with some randomness)
            for u in range(n):
                for v in range(u+1, n):
                    c = (u * v + random.randint(0, 3)) % n
                    adj[u][v] = c
                    adj[v][u] = c
        elif strategy == 3:
            # Structured: use vertex-based coloring
            # Each vertex gets a "palette offset"
            offsets = list(range(n))
            random.shuffle(offsets)
            for u in range(n):
                for v in range(u+1, n):
                    c = (offsets[u] + v) % n
                    adj[u][v] = c
                    adj[v][u] = c
        else:
            # Hybrid: start with structured, then randomize some
            for u in range(n):
                for v in range(u+1, n):
                    if random.random() < 0.5:
                        c = (u + v) % n
                    else:
                        c = random.randint(0, n-1)
                    adj[u][v] = c
                    adj[v][u] = c
        
        # Greedy deletion: repeatedly remove the edge in the most bad cycles
        max_iterations = 500
        for iteration in range(max_iterations):
            if time.time() > deadline:
                break
            
            bad_count = total_bad_cycles(adj)
            if bad_count == 0:
                break
            
            # Find edge with maximum bad cycle involvement
            max_bad = -1
            worst_edge = None
            for u in range(n):
                for v in range(u+1, n):
                    if adj[u][v] == -1:
                        continue
                    b = count_bad_cycles(adj, u, v)
                    if b > max_bad:
                        max_bad = b
                        worst_edge = (u, v)
            
            if worst_edge is None or max_bad <= 0:
                # No single edge can fix it; try recoloring
                # Pick a random edge in a bad cycle and recolor it
                found = False
                for _ in range(100):
                    u = random.randint(0, n-1)
                    v = random.randint(0, n-1)
                    if u == v or adj[u][v] == -1:
                        continue
                    # Check if this edge is in a bad cycle
                    old_c = adj[u][v]
                    for new_c in range(n):
                        if new_c == old_c:
                            continue
                        adj[u][v] = new_c
                        adj[v][u] = new_c
                        new_bad = total_bad_cycles(adj)
                        if new_bad < bad_count:
                            found = True
                            break
                        adj[u][v] = old_c
                        adj[v][u] = old_c
                    if found:
                        break
                if not found:
                    # Just remove the worst edge
                    if worst_edge:
                        u, v = worst_edge
                        adj[u][v] = -1
                        adj[v][u] = -1
                    else:
                        break
            else:
                u, v = worst_edge
                # Try recoloring first before deleting
                old_c = adj[u][v]
                improved = False
                for new_c in range(n):
                    if new_c == old_c:
                        continue
                    adj[u][v] = new_c
                    adj[v][u] = new_c
                    new_bad = total_bad_cycles(adj)
                    if new_bad < bad_count:
                        improved = True
                        break
                    adj[u][v] = old_c
                    adj[v][u] = old_c
                
                if not improved:
                    adj[u][v] = -1
                    adj[v][u] = -1
        
        # Final check and update best
        if is_valid(adj):
            ec = edge_count(adj)
            if ec > best_count:
                best_count = ec
                best_edges = make_edges(adj)
    
    return best_edges
# EVOLVE-BLOCK-END
