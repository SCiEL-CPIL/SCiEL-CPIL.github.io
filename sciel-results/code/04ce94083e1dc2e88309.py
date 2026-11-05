import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 20
    # Initialize complete graph
    # color[u][v] = c
    color = [[None] * n for _ in range(n)]
    
    for u in range(n):
        for v in range(u + 1, n):
            c = random.randint(0, n - 1)
            color[u][v] = c
            color[v][u] = c
            
    # Precompute all 4-cycles
    cycles = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    def is_valid_cycle(cyc, active_edges_set=None):
        if active_edges_set is not None:
            for u, v in cyc:
                if (u, v) not in active_edges_set and (v, u) not in active_edges_set:
                    return True # Cycle is broken
        cols = []
        for u, v in cyc:
            cols.append(color[u][v])
        return len(set(cols)) == 4
    
    # Initial check for full graph
    active_edges_set = set((u, v) for u in range(n) for v in range(u+1, n))
    invalid_cycles = [c for c in cycles if not is_valid_cycle(c, active_edges_set)]
    
    # Local Search on Full Graph
    # Try to fix all conflicts without removing edges
    stuck_count = 0
    while time.time() < deadline and invalid_cycles and stuck_count < 1000:
        # Pick a random invalid cycle
        cyc = random.choice(invalid_cycles)
        
        # Identify edges in this cycle that can be changed
        # We just pick a random edge
        edge_idx = random.randint(0, 3)
        u, v = cyc[edge_idx]
        
        # Try to find a new color that makes this cycle valid
        # And ideally doesn't break other cycles (but we check globally)
        fixed = False
        for _ in range(20):
            new_c = random.randint(0, n - 1)
            if color[u][v] == new_c:
                continue
            
            old_c = color[u][v]
            color[u][v] = new_c
            color[v][u] = new_c
            
            if is_valid_cycle(cyc, active_edges_set):
                # Check if we broke others? 
                # We update the global list
                invalid_cycles = [c for c in cycles if not is_valid_cycle(c, active_edges_set)]
                fixed = True
                stuck_count = 0
                break
            else:
                color[u][v] = old_c
                color[v][u] = old_c
        
        if not fixed:
            stuck_count += 1
            # If stuck, try a different strategy or break to remove edges
            if stuck_count > 100:
                break
                
    # If still invalid, remove edges
    if time.time() < deadline and invalid_cycles:
        while time.time() < deadline and invalid_cycles:
            # Pick a random invalid cycle
            cyc = random.choice(invalid_cycles)
            
            # Pick a random edge in this cycle to remove
            edge_idx = random.randint(0, 3)
            u, v = cyc[edge_idx]
            
            # Remove edge
            active_edges_set.discard((u, v))
            active_edges_set.discard((v, u))
            
            # Update invalid cycles
            # Only cycles that contained this edge are affected (they become broken)
            # So we can just filter
            invalid_cycles = [c for c in cycles if not is_valid_cycle(c, active_edges_set)]
            
    # Construct the result
    result = []
    for u, v in active_edges_set:
        if u < v:
            result.append([u, v, color[u][v]])
            
    return result
