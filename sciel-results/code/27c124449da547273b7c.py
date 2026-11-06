import time
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin before 150s kill

    # Try n=18 first (new territory), then n=17 (known), then n=16
    for n in [18, 17, 16]:
        if time.time() - start_time > time_limit - 10:
            break
        result = try_find_coloring(n, start_time, time_limit)
        if result:
            return result

    # Fallback: return a valid small coloring
    n = 5
    # Simple 4-coloring of K5: color by (i+j) mod 4? 
    # Actually, for K5, we can just use a proper 4-edge-coloring or any.
    # Let's use a cyclic coloring: color(i,j) = (j - i) % 4
    # This might have C6? No, n=5 < 6, so no C6 possible.
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            colors.append((j - i) % 4)
    return {'n': n, 'colors': colors}

def try_find_coloring(n, start_time, time_limit):
    # Precompute all C6 cycles and edge-to-cycle mapping
    # This is the expensive part. For n=18, C(18,6)=18564, *60 = 1,113,840 cycles.
    # This might be too slow in pure Python. Let's estimate.
    # 1M cycles * ~10 operations = 10M ops. In Python, ~1s per 10M simple ops? Maybe 2-5s.
    # If it takes too long, we abort and return None.
    
    precomp_start = time.time()
    
    cycles = []
    edge_indices = {}
    
    # Optimization: Use a list of lists for edge_indices to avoid dict overhead in inner loop?
    # But we need to map edge -> list of cycle indices.
    # Let's use a dict of edge -> list.
    
    # Generate all 6-vertex subsets
    # For each, generate 60 canonical cycles
    
    # To speed up generation, we can use a precomputed list of relative cycle structures?
    # There are only 60 distinct relative cycles on {0,1,2,3,4,5}.
    # Let's generate them once.
    
    rel_cycles = []
    verts_rel = (0, 1, 2, 3, 4, 5)
    v0_rel = 0
    rest_rel = (1, 2, 3, 4, 5)
    for perm in permutations(rest_rel):
        if perm[0] < perm[-1]:
            cycle = (v0_rel, perm[0], perm[1], perm[2], perm[3], perm[4])
            rel_cycles.append(cycle)
            
    # Now for each subset of 6 vertices, map the relative cycle to actual vertices
    # Subset is sorted: v0 < v1 < ... < v5
    # Relative cycle has 0 at pos 0.
    # We need to map relative index k to actual vertex verts[k].
    
    for verts in combinations(range(n), 6):
        # verts is a tuple of 6 sorted ints
        for rc in rel_cycles:
            # Map rc to actual cycle
            # rc[0] is 0, rc[1..5] are a permutation of 1..5
            # actual_cycle[i] = verts[rc[i]]
            ac = tuple(verts[rc[i]] for i in range(6))
            cycles.append(ac)
            
            # Update edge indices
            # Edges in cycle: (0,1), (1,2), (2,3), (3,4), (4,5), (5,0)
            for k in range(6):
                u = ac[k]
                v = ac[(k + 1) % 6]
                key = (min(u, v), max(u, v))
                if key not in edge_indices:
                    edge_indices[key] = []
                edge_indices[key].append(len(cycles) - 1)
                
    if time.time() - precomp_start > 5.0:
        # Precomputation took too long, abort for this n
        return None
        
    # Local Search with Simulated Annealing / Greedy Repair
    # Initialize random coloring
    colors = np.random.randint(0, 4, size=(n, n))
    colors = (colors + colors.T) // 2
    np.fill_diagonal(colors, 0)
    
    # We need to quickly compute bad_count.
    # bad_count = sum over cycles of [all edges same color]
    
    # To make this fast, we can use numpy?
    # But cycles are irregular.
    # Let's stick to Python loops but optimize the check.
    
    def count_bad(colors, cycles):
        count = 0
        for c in cycles:
            col = colors[c[0]][c[1]]
            # Unrolled check
            if (col == colors[c[1]][c[2]] and col == colors[c[2]][c[3]] and
                col == colors[c[3]][c[4]] and col == colors[c[4]][c[5]] and
                col == colors[c[5]][c[0]]):
                count += 1
        return count

    bad_count = count_bad(colors, cycles)
    
    if bad_count == 0:
        return {'n': n, 'colors': [int(colors[i, j]) for i in range(n) for j in range(i + 1, n)]}
    
    best_bad = bad_count
    best_colors = colors.copy()
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9999
    
    # We will pick a random edge and try to change its color.
    # We need to update bad_count efficiently.
    # When edge (u,v) changes color from c_old to c_new:
    # Only cycles containing (u,v) are affected.
    # For each such cycle, we check if it was monochromatic before and after.
    
    while time.time() - start_time < time_limit:
        # Pick random edge
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        if u == v:
            continue
        if u > v:
            u, v = v, u
            
        key = (u, v)
        if key not in edge_indices:
            continue
            
        affected = edge_indices[key]
        if not affected:
            continue
            
        c_old = colors[u, v]
        
        # Find best new color
        best_delta = float('inf')
        best_new_color = c_old
        
        for c_new in range(4):
            if c_new == c_old:
                continue
                
            delta = 0
            for idx in affected:
                c = cycles[idx]
                # Check if cycle is monochromatic with c_old (before)
                # and c_new (after)
                # We know edge (u,v) is c_old before, c_new after.
                # Other edges are unchanged.
                
                # Check other 5 edges
                # We need to know if all other edges are c_old, and if all other edges are c_new.
                # If all other 5 are c_old:
                #   Before: all 6 are c_old -> Bad.
                #   After: 5 are c_old, 1 is c_new -> Not Bad (unless c_old==c_new, which is false).
                #   So delta -= 1.
                # If all other 5 are c_new:
                #   Before: 5 are c_new, 1 is c_old -> Not Bad.
                #   After: all 6 are c_new -> Bad.
                #   So delta += 1.
                # Else: No change in bad status.
                
                # Efficient check:
                # We can precompute for each cycle a bitmask or something?
                # Or just iterate the 5 edges.
                
                match_old = True
                match_new = True
                
                # Edges in cycle: (c0,c1), (c1,c2), (c2,c3), (c3,c4), (c4,c5), (c5,c0)
                # One of them is (u,v) or (v,u).
                for k in range(6):
                    e_u = c[k]
                    e_v = c[(k + 1) % 6]
                    if (e_u == u and e_v == v) or (e_u == v and e_v == u):
                        continue
                    ec = colors[e_u][e_v]
                    if ec != c_old:
                        match_old = False
                    if ec != c_new:
                        match_new = False
                    if not match_old and not match_new:
                        break # Early exit: can't be either
                    
                if match_old:
                    delta -= 1
                if match_new:
                    delta += 1
                    
            if delta < best_delta:
                best_delta = delta
                best_new_color = c_new
                
        # Apply move with SA acceptance
        if best_delta < 0:
            # Always accept improving moves
            colors[u, v] = best_new_color
            colors[v, u] = best_new_color
            bad_count += best_delta
            if bad_count < best_bad:
                best_bad = bad_count
                best_colors = colors.copy()
                if bad_count == 0:
                    return {'n': n, 'colors': [int(best_colors[i, j]) for i in range(n) for j in range(i + 1, n)]}
        else:
            # Accept worsening moves with probability exp(-delta/T)
            if random.random() < np.exp(-best_delta / T):
                colors[u, v] = best_new_color
                colors[v, u] = best_new_color
                bad_count += best_delta
                
        T *= cooling
        if T < T_min:
            T = T_min
            # Random restart? Or just keep going?
            # Let's add a small random kick if stuck for a while
            # For now, just continue.
            
    # Return best found if it's valid? No, must be valid.
    if best_bad == 0:
        return {'n': n, 'colors': [int(best_colors[i, j]) for i in range(n) for j in range(i + 1, n)]}
        
    return None
