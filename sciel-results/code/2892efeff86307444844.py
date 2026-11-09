import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    N = 256
    squares = set()
    i = 2
    while i * i <= 2 * N:
        squares.add(i * i)
        i += 1
    
    # Filter self-conflicts
    valid_nodes = [v for v in range(1, N + 1) if 2 * v not in squares]
    node_set = set(valid_nodes)
    
    # Build adjacency masks
    adj_mask = [0] * (N + 1)
    
    for v in valid_nodes:
        for u in valid_nodes:
            if u > v and (v + u) in squares:
                adj_mask[v] |= (1 << u)
                adj_mask[u] |= (1 << v)
                
    initial_mask = 0
    for v in valid_nodes:
        initial_mask |= (1 << v)
        
    best_set = set()
    best_size = 0
    
    # Mod 3 baseline
    mod3_set = set(v for v in range(1, N+1) if v % 3 == 1)
    best_set = mod3_set
    best_size = len(mod3_set)
    
    def greedy_one_restart():
        current_mask = initial_mask
        sol = []
        while current_mask:
            min_deg = N + 1
            candidates = []
            for v in valid_nodes:
                if current_mask & (1 << v):
                    deg = (adj_mask[v] & current_mask).bit_count()
                    if deg < min_deg:
                        min_deg = deg
                        candidates = [v]
                    elif deg == min_deg:
                        candidates.append(v)
            if not candidates:
                break
            v = random.choice(candidates)
            sol.append(v)
            current_mask &= ~(adj_mask[v] | (1 << v))
        return sol
    
    def local_improve(sol_set):
        """Try to add nodes back or swap to improve the set."""
        sol_set = set(sol_set)
        improved = True
        while improved:
            improved = False
            # Try adding each valid node not in the set
            for v in valid_nodes:
                if v in sol_set:
                    continue
                # Check if v conflicts with any node in sol_set
                conflict = False
                for u in sol_set:
                    if (v + u) in squares:
                        conflict = True
                        break
                if not conflict:
                    sol_set.add(v)
                    improved = True
            # Try swapping: remove one, add another
            if not improved:
                for v_out in list(sol_set):
                    for v_in in valid_nodes:
                        if v_in in sol_set:
                            continue
                        # Check v_in doesn't conflict with sol_set - {v_out}
                        ok = True
                        for u in sol_set:
                            if u == v_out:
                                continue
                            if (v_in + u) in squares:
                                ok = False
                                break
                        if ok:
                            sol_set.discard(v_out)
                            sol_set.add(v_in)
                            improved = True
                            break
                    if improved:
                        break
        return sol_set
    
    num_restarts = 5000  # Each restart is very fast
    
    for r in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        sol = greedy_one_restart()
        
        if len(sol) > best_size:
            # Apply local improvement
            improved_sol = local_improve(set(sol))
            if len(improved_sol) > best_size:
                best_size = len(improved_sol)
                best_set = improved_sol
            else:
                best_size = len(sol)
                best_set = set(sol)
        elif len(sol) == best_size - 1:
            # Close enough, try local improvement
            improved_sol = local_improve(set(sol))
            if len(improved_sol) > best_size:
                best_size = len(improved_sol)
                best_set = improved_sol
    
    return list(best_set)
