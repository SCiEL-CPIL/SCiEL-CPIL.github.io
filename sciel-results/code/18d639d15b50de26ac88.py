import numpy as np
import itertools
import time
import random

# Precompute K6 cycles (60 of them)
# Vertices 0..5
# A cycle is a sequence of 6 vertices.
# Canonical form: start with 0, then the smallest possible next vertex?
# Actually, just generate all permutations of 1..5, prepend 0, and check uniqueness.
# Or simpler: use a set of frozensets of edges? No, edges are undirected.
# A cycle is defined by the set of edges.
# Two cycles are the same if they have the same set of edges.
# In K6, each cycle uses 6 edges.
# Total edges in K6 is 15.
# We can represent a cycle by a frozenset of its 6 edges (local indices).

def get_k6_cycles():
    cycles = set()
    # Permutations of 1,2,3,4,5
    for perm in itertools.permutations([1, 2, 3, 4, 5]):
        v = (0,) + perm
        # Edges: (v[i], v[i+1]) for i=0..4 and (v[5], v[0])
        edges = set()
        for i in range(6):
            u, w = v[i], v[(i + 1) % 6]
            if u > w: u, w = w, u
            edges.add((u, w))
        cycles.add(frozenset(edges))
    return [list(c) for c in cycles]

K6_CYCLES = get_k6_cycles() # Should be 60

def construct():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    # Try n=17 first
    n = 17
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for all C6 in Kn
    # idx(i, j) = i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    def get_idx(i, j):
        return i * (n - 1) - i * (i - 1) // 2 + (j - i - 1)
    
    c6_edges_list = []
    # Iterate over all combinations of 6 vertices
    for comb in itertools.combinations(range(n), 6):
        # For each of the 60 K6 cycles
        for cycle_edges in K6_CYCLES:
            global_edges = []
            for u, v in cycle_edges:
                # u, v are local indices 0..5
                # Map to global
                gu = comb[u]
                gv = comb[v]
                if gu > gv: gu, gv = gv, gu
                global_edges.append(get_idx(gu, gv))
            c6_edges_list.append(global_edges)
            
    c6_edges = np.array(c6_edges_list, dtype=np.int32)
    
    # Local search
    best_score = float('inf')
    best_colors = None
    
    # Initialize random
    colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
    
    # Evaluate function
    def evaluate(c):
        c6_c = c[c6_edges]
        # Check if all 6 are same
        # c6_c[:, 0] is the first color
        # Broadcasting comparison
        is_mono = np.all(c6_c == c6_c[:, 0:1], axis=1)
        return np.sum(is_mono)
    
    current_score = evaluate(colors)
    if current_score == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    best_score = current_score
    best_colors = colors.copy()
    
    # Simulated Annealing / Hill Climbing
    # We will just do hill climbing with random restarts if stuck
    # Since evaluation is fast, we can do many steps
    
    steps_per_restart = 5000
    num_restarts = 100
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            # Use initial random
            pass
        else:
            colors = np.random.randint(0, 4, num_edges, dtype=np.int8)
            current_score = evaluate(colors)
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
            if current_score == 0:
                return {'n': n, 'colors': colors.tolist()}
                
        for step in range(steps_per_restart):
            if time.time() > deadline:
                break
                
            # Pick random edge
            idx = np.random.randint(num_edges)
            old_color = colors[idx]
            
            # Pick new color different from old
            new_color = np.random.randint(0, 3)
            if new_color >= old_color:
                new_color += 1
            # new_color is now in {0,1,2,3} \ {old_color}
            
            colors[idx] = new_color
            new_score = evaluate(colors)
            
            if new_score < current_score:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
                if current_score == 0:
                    return {'n': n, 'colors': colors.tolist()}
            else:
                # Revert
                colors[idx] = old_color
                
        # If we are stuck at a local minimum, restart with new random
        # Or try to escape by accepting worse moves?
        # Given the speed, simple restarts might work.
        
    # If n=17 failed, try n=16?
    # The score is n/9. 17 is much better than 16.
    # If we found a valid 17, we returned.
    # If not, we return the best found?
    # The task says "Maximize n".
    # If we don't find a valid coloring for 17, we should return a valid one for smaller n?
    # The seed returns n=9.
    # If we return an invalid coloring for 17, score is 0.
    # So we must ensure we return a VALID coloring.
    # If we didn't find 0 violations for 17, we should fall back to a known valid one, e.g., n=9 from seed, or try to find valid 16, 15, etc.
    
    # Let's try to find valid colorings for smaller n if 17 fails quickly?
    # No, 17 is the target. If we can't find it in 115s, we return seed.
    
    if best_score == 0:
        return {'n': n, 'colors': best_colors.tolist()}
    else:
        # Fallback to seed
        n_seed = 9
        return {'n': n_seed, 'colors': [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]}
