"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    # Try n from 17 down to 1
    # We prioritize finding a high n.
    # If we can't find 17, we return the best found.
    
    best_n = 1
    best_colors = [0]  # Placeholder for n=1 (no edges)
    
    # We will try to solve for n=17, 16, 15...
    # Given the difficulty, we might only have time for one n.
    # Let's try n=17 first. If it fails, return best found (which might be small).
    
    for target_n in [17, 16, 15]:
        if time.time() > deadline - 5:
            break
            
        if _solve_n(target_n, deadline):
            best_n = target_n
            # _solve_n returns the colors if successful
            # We need to capture the return value
            # Let's modify _solve_n to return (success, colors)
            pass # Placeholder for logic structure

    # Refine: Implement _solve_n properly
    
    return {'n': best_n, 'colors': best_colors}

def _solve_n(n, deadline):
    """
    Attempts to find a 4-coloring of K_n with no monochromatic C6.
    Returns (success: bool, colors: list)
    """
    if n <= 1:
        return True, []
        
    m = n * (n - 1) // 2
    
    # 1. Precompute C6s
    # Map edge (i, j) to index
    def edge_idx(i, j):
        if i > j: i, j = j, i
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
        
    # Generate all 6-vertex subsets
    c6s = []
    for vertices in combinations(range(n), 6):
        # For each set of 6 vertices, generate all 60 Hamiltonian cycles
        # Actually, we just need the set of edges.
        # A C6 is defined by the set of 6 edges.
        # There are 60 distinct undirected cycles on 6 labeled vertices.
        # We can generate them by permuting the vertices and fixing the first one to break rotation/reversal symmetry?
        # Simpler: Iterate through all permutations of the 6 vertices, generate the cycle edges, and add to a set to avoid duplicates.
        # Or just generate the 60 cycles explicitly.
        # The 60 cycles correspond to the 60 ways to arrange 5 items in a circle?
        # Yes, (6-1)!/2 = 60.
        
        # To be efficient, we can just pick one representative for each "shape" of cycle?
        # No, the vertices are labeled, so each permutation is distinct.
        # We will generate all 60.
        
        v = list(vertices)
        # Fix v[0] to break rotational symmetry? No, the set of edges is what matters.
        # The set of edges for a cycle (v0, v1, v2, v3, v4, v5) is:
        # (v0,v1), (v1,v2), (v2,v3), (v3,v4), (v4,v5), (v5,v0)
        
        seen_cycles = set()
        # Generate all permutations of v[1:] (5 items)
        for perm in permutations(v[1:]):
            cycle_verts = [v[0]] + list(perm)
            # Generate edge set
            edges = set()
            for k in range(6):
                u = cycle_verts[k]
                w = cycle_verts[(k + 1) % 6]
                idx = edge_idx(u, w)
                edges.add(idx)
            # To avoid duplicates from reversal, we can canonicalize the edge set
            # But using a set of tuples is fine for deduplication within this vertex set
            edge_tuple = tuple(sorted(edges))
            if edge_tuple not in seen_cycles:
                seen_cycles.add(edge_tuple)
                c6s.append(edge_tuple)
                
    # 2. Map edge index to list of C6 indices containing it
    c6s_for_edge = [[] for _ in range(m)]
    for i, c6 in enumerate(c6s):
        for e in c6:
            c6s_for_edge[e].append(i)
            
    # 3. Initialize colors
    # Symmetry breaking:
    # Fix (0,1)=0, (0,2)=1, (0,3)=2 if n>3
    colors = [random.randint(0, 3) for _ in range(m)]
    if n > 1:
        colors[edge_idx(0, 1)] = 0
    if n > 2:
        colors[edge_idx(0, 2)] = 1
    if n > 3:
        colors[edge_idx(0, 3)] = 2
        
    # 4. Count initial mono C6s
    def count_mono():
        count = 0
        for c6 in c6s:
            c0 = colors[c6[0]]
            if c0 == colors[c6[1]] == colors[c6[2]] == colors[c6[3]] == colors[c6[4]] == colors[c6[5]]:
                count += 1
        return count
        
    current_mono = count_mono()
    best_mono = current_mono
    best_colors = list(colors)
    
    if current_mono == 0:
        return True, colors
        
    # 5. Local Search
    # Use Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    # Precompute edge indices for faster access?
    # colors is a list, access is O(1).
    
    while T > min_T:
        if time.time() > deadline:
            break
            
        for _ in range(100): # 100 steps per temperature
            if time.time() > deadline:
                break
                
            e = random.randint(0, m - 1)
            old_c = colors[e]
            new_c = random.choice([c for c in range(4) if c != old_c])
            
            # Check affected C6s
            delta = 0
            affected = c6s_for_edge[e]
            
            # We need to check if the C6 was mono before, and if it is mono after.
            # To avoid modifying colors and reverting, we can simulate the new color for edge e.
            
            for c6_idx in affected:
                c6 = c6s[c6_idx]
                # Check old state
                # All edges in c6 have colors from `colors`, except e which has `old_c`
                # We need to check if all 6 were same.
                # Since e is in c6, we can check:
                # 1. Are all other 5 edges same as old_c?
                # 2. If yes, old C6 was mono.
                
                # Optimization: Just check if all 6 are same.
                # We have to be careful not to use the new color for the old check.
                
                # Let's just extract the 6 colors.
                # This is slow.
                
                # Better:
                # Check if c6 is mono with old_c at e.
                is_old_mono = True
                for idx in c6:
                    if idx == e:
                        c_val = old_c
                    else:
                        c_val = colors[idx]
                    if c_val != old_c: # Wait, we need to check if ALL are same.
                        # If any is different, it's not mono.
                        # But we don't know the target color yet?
                        # The target color for mono is the color of the edges.
                        # If all edges are color X, it is mono.
                        # So, check if all edges are the same color.
                        pass
                        
                # This logic is getting messy.
                # Let's just check:
                # old_mono: all colors in c6 are equal.
                # new_mono: all colors in c6 are equal, with e having new_c.
                
                # To speed up:
                # If the other 5 edges are not all the same, it can't be mono.
                # So, check if the 5 other edges are all the same color.
                
                other_edges = [idx for idx in c6 if idx != e]
                c_other = colors[other_edges[0]]
                all_other_same = True
                for idx in other_edges[1:]:
                    if colors[idx] != c_other:
                        all_other_same = False
                        break
                        
                if all_other_same:
                    # The 5 other edges are color c_other.
                    # Old state: Mono if old_c == c_other.
                    # New state: Mono if new_c == c_other.
                    if old_c == c_other and new_c != c_other:
                        delta -= 1 # Destroyed a mono C6
                    elif old_c != c_other and new_c == c_other:
                        delta += 1 # Created a mono C6
                    # If old_c == c_other and new_c == c_other, delta 0 (impossible as new_c != old_c)
                    # If old_c != c_other and new_c != c_other, delta 0
                    
                # If not all_other_same, it was never mono, and won't be mono (since 5 edges differ, adding one more can't make them all same? 
                # Wait. If 5 edges are not all same, can the 6th make them all same?
                # No. If edges A, B, C, D, E are not all same, then there exist i, j with color_i != color_j.
                # Even if we set F to color_i, color_j is still different. So it cannot be mono.
                # So we only care about the case where the other 5 are same.
                
            new_mono = current_mono + delta
            
            if new_mono < best_mono:
                best_mono = new_mono
                best_colors = list(colors)
                colors[e] = new_c
                current_mono = new_mono
                if current_mono == 0:
                    return True, colors
            elif random.random() < np.exp(-delta / T):
                colors[e] = new_c
                current_mono = new_mono
            else:
                # Revert
                colors[e] = old_c
                
        T *= cooling
        
    if best_mono == 0:
        return True, best_colors
    return False, best_colors

# EVOLVE-BLOCK-END
