import math
import time

def construct():
    start_time = time.time()
    deadline = 85.0
    
    moduli = list(range(2, 129, 2))
    n_mods = len(moduli)
    
    nodes = []
    for m in moduli:
        for r in range(m):
            nodes.append((m, r))
            
    N = len(nodes)
    
    offsets = {}
    curr_idx = 0
    for m in moduli:
        offsets[m] = curr_idx
        curr_idx += m
        
    # Greedy to get initial solution
    greedy_solution = []
    # Try to pick moduli in some order
    for m in moduli:
        found = False
        for r in range(m):
            compatible = True
            for (m2, r2) in greedy_solution:
                g = math.gcd(m, m2)
                if r % g == r2 % g:
                    compatible = False
                    break
            if compatible:
                greedy_solution.append((m, r))
                found = True
                break
        if not found:
            continue
            
    best_size = len(greedy_solution)
    best_nodes = [offsets[m] + r for m, r in greedy_solution]
    
    # Build neighbors
    neighbors = [0] * N
    mod_to_idx = {m: i for i, m in enumerate(moduli)}
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            if i != j:
                gcd_table[(i, j)] = math.gcd(m1, m2)
                
    for i, (m1, r1) in enumerate(nodes):
        start_idx = offsets[m1]
        end_idx = start_idx + m1
        for j in range(start_idx, end_idx):
            if i != j:
                neighbors[i] |= (1 << j)
        for m2 in moduli:
            if m2 == m1:
                continue
            g = gcd_table[(mod_to_idx[m1], mod_to_idx[m2])]
            target_res = r1 % g
            start_idx2 = offsets[m2]
            for r2 in range(target_res, m2, g):
                j = start_idx2 + r2
                neighbors[i] |= (1 << j)

    def solve_mis(candidates, current_nodes, current_size):
        nonlocal best_size, best_nodes
        
        if time.time() > deadline - 1:
            return
            
        if current_size + candidates.bit_count() <= best_size:
            return
            
        if candidates == 0:
            if current_size > best_size:
                best_size = current_size
                best_nodes = list(current_nodes)
            return
            
        lsb = candidates & -candidates
        u = lsb.bit_length() - 1
        
        new_candidates = candidates & ~neighbors[u] & ~lsb
        current_nodes.append(u)
        solve_mis(new_candidates, current_nodes, current_size + 1)
        current_nodes.pop()
        
        new_candidates2 = candidates & ~lsb
        solve_mis(new_candidates2, current_nodes, current_size)

    initial_candidates = (1 << N) - 1
    solve_mis(initial_candidates, [], 0)
    
    result = []
    for u in best_nodes:
        m, r = nodes[u]
        result.append([r, m])
        
    if not result:
        result = [[1, 2], [3, 4], [5, 8], [7, 16], [15, 32], [31, 64], [63, 128]]
        
    return result
