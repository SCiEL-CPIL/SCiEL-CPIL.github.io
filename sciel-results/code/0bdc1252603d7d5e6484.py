import random
import time
import math

def construct():
    N = 31
    E_target = 91
    deadline = time.time() + 140.0  # 10s margin
    
    # Graph representation
    # neighbors[i] is a set of neighbors of i
    neighbors = [set() for _ in range(N)]
    edges = set()
    
    # C[u][v] is the number of common neighbors of u and v
    C = [[0]*N for _ in range(N)]
    
    def add_edge(u, v):
        if u == v or (u, v) in edges or (v, u) in edges:
            return False
        neighbors[u].add(v)
        neighbors[v].add(u)
        edges.add((u, v))
        
        # Update C matrix
        # C[u][w] increases by 1 for all w in neighbors[v]
        for w in neighbors[v]:
            C[u][w] += 1
            C[w][u] += 1
        # C[v][w] increases by 1 for all w in neighbors[u]
        for w in neighbors[u]:
            C[v][w] += 1
            C[w][v] += 1
            
        # Note: C[u][v] and C[v][u] should reflect common neighbors.
        # The loops above handle this.
        # For w=u: C[v][u] += 1 (from second loop, w=u is in neighbors[u]? No, u is not in neighbors[u]).
        # Wait, neighbors[u] does not contain u.
        # So C[u][v] is updated in the first loop when w=v? No, v is in neighbors[v]? No.
        # Let's trace:
        # add_edge(u, v)
        # neighbors[u] = {v, ...}
        # neighbors[v] = {u, ...}
        # Loop 1: for w in neighbors[v]: C[u][w] += 1.
        #   If w=u, C[u][u] += 1. (Self common neighbor? We usually don't care about C[u][u] for C4 count as it's not an edge term, but it might be used in sums? No, sums are over edges. C[u][u] is not used in S.)
        #   If w is another neighbor of v, C[u][w] increases.
        # Loop 2: for w in neighbors[u]: C[v][w] += 1.
        #   If w=v, C[v][v] += 1.
        
        # What about C[u][v]?
        # C[u][v] is the number of common neighbors of u and v.
        # Before adding (u, v), let k be the number of common neighbors.
        # After adding, u and v are neighbors. Do they share new common neighbors?
        # No, the edge (u, v) itself doesn't create new common neighbors for u and v.
        # However, the *definition* of C[u][v] is |N(u) cap N(v)|.
        # N(u) gained v. N(v) gained u.
        # So N(u) cap N(v) now includes u (since u in N(v) and u in N(u)? No, u not in N(u)) and v (since v in N(u) and v in N(v)? No, v not in N(v)).
        # So C[u][v] does NOT change due to the addition of the edge (u, v) itself?
        # Wait.
        # Common neighbors of u and v are nodes w such that w in N(u) and w in N(v).
        # Adding edge (u, v) means v in N(u) and u in N(v).
        # Does v become a common neighbor of u and v?
        # v is in N(u). Is v in N(v)? No. So v is not a common neighbor.
        # Does u become a common neighbor?
        # u is in N(v). Is u in N(u)? No.
        # So C[u][v] remains unchanged by the addition of the edge (u, v).
        
        # But my update logic:
        # Loop 1: for w in neighbors[v] (which now includes u? No, we added u to neighbors[v] BEFORE the loop?
        # Yes: neighbors[v].add(u) is done.
        # So w can be u.
        # If w=u, C[u][u] += 1.
        # This is correct.
        
        return True

    def remove_edge(u, v):
        if (u, v) not in edges and (v, u) not in edges:
            return False
        neighbors[u].remove(v)
        neighbors[v].remove(u)
        edges.discard((u, v))
        edges.discard((v, u))
        
        # Update C matrix
        # C[u][w] decreases by 1 for all w in neighbors[v] (current, which excludes u)
        for w in neighbors[v]:
            C[u][w] -= 1
            C[w][u] -= 1
        # C[v][w] decreases by 1 for all w in neighbors[u] (current, which excludes v)
        for w in neighbors[u]:
            C[v][w] -= 1
            C[w][v] -= 1
            
        return True

    def calc_S():
        s = 0
        for u, v in edges:
            c = C[u][v]
            s += c * (c - 1) // 2
        return s

    # Initialize graph
    # Random start
    random.seed(42)
    all_pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
    random.shuffle(all_pairs)
    
    for u, v in all_pairs:
        if len(edges) >= E_target:
            break
        add_edge(u, v)
        
    current_S = calc_S()
    best_S = current_S
    best_edges = set(edges)
    best_neighbors = [set(s) for s in neighbors]
    best_C = [row[:] for row in C]
    
    # Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.1
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        # Pick random edge to remove
        if not edges:
            break
        rem_u, rem_v = random.choice(list(edges))
        
        # Pick random non-edge to add
        # Generate random non-edge
        while True:
            add_u = random.randint(0, N-1)
            add_v = random.randint(0, N-1)
            if add_u == add_v:
                continue
            if (add_u, add_v) in edges or (add_v, add_u) in edges:
                continue
            break
        
        # Calculate Delta S for removal
        # Delta S_rem = - binom(C(rem_u, rem_v), 2) - sum_{w in N(rem_u)\{rem_v}} (C(rem_u, w) - 1) - sum_{w in N(rem_v)\{rem_u}} (C(rem_v, w) - 1)
        
        c_rem = C[rem_u][rem_v]
        delta_S = -(c_rem * (c_rem - 1) // 2)
        
        for w in neighbors[rem_u]:
            if w == rem_v: continue
            cw = C[rem_u][w]
            delta_S -= (cw - 1)
            
        for w in neighbors[rem_v]:
            if w == rem_u: continue
            cw = C[rem_v][w]
            delta_S -= (cw - 1)
            
        # Perform removal to update C for addition calculation?
        # Or calculate addition delta based on current C?
        # If we calculate addition delta based on current C, we assume the removal didn't happen.
        # But the addition delta depends on the state AFTER removal.
        # So we must either:
        # 1. Update C for removal, calc add delta, then update C for addition (if accepted) or revert (if not).
        # 2. Calc both deltas based on "before" state, accounting for interactions.
        
        # Option 1 is safer.
        # But we need to revert if not accepted.
        # Reverting C is expensive if we don't save it.
        # Saving C (31x31 matrix) is cheap.
        
        C_backup = [row[:] for row in C]
        edges_backup = set(edges)
        neighbors_backup = [set(s) for s in neighbors]
        
        # Remove
        remove_edge(rem_u, rem_v)
        
        # Calculate Delta S for addition
        # Delta S_add = binom(C(add_u, add_v), 2) + sum_{w in N(add_v)} C(add_u, w) + sum_{w in N(add_u)} C(add_v, w)
        
        c_add = C[add_u][add_v]
        delta_S_add = c_add * (c_add - 1) // 2
        
        for w in neighbors[add_v]:
            delta_S_add += C[add_u][w]
            
        for w in neighbors[add_u]:
            delta_S_add += C[add_v][w]
            
        total_delta_S = delta_S + delta_S_add
        
        # Acceptance
        accepted = False
        if total_delta_S < 0:
            accepted = True
        else:
            if random.random() < math.exp(-total_delta_S / T):
                accepted = True
                
        if accepted:
            add_edge(add_u, add_v)
            current_S += total_delta_S
            if current_S < best_S:
                best_S = current_S
                best_edges = set(edges)
                best_neighbors = [set(s) for s in neighbors]
                best_C = [row[:] for row in C]
        else:
            # Revert
            # Restore C, edges, neighbors
            C[:] = [row[:] for row in C_backup]
            edges = edges_backup
            neighbors = neighbors_backup
            # Note: current_S remains unchanged
            
        T *= cooling
        
    # Return the best graph found
    # Format: list of [u, v]
    result = []
    for u, v in best_edges:
        result.append([u, v])
    
    return result

# EVOLVE-BLOCK-START
# The construct function is defined above.
# We just need to expose it.
# EVOLVE-BLOCK-END
