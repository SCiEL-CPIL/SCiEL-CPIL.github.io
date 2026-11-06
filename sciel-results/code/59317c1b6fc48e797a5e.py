"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Find a valid 2-colouring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Maximise n. The known construction has n=34.
    """
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin before 120s limit
    
    # We'll try to find a valid colouring for n=34 using simulated annealing.
    # If we can't find one in time, fall back to a smaller n.
    
    target_n = 34
    
    # Precompute all triples in lexicographic order
    triples = list(combinations(range(target_n), 3))
    num_triples = len(triples)
    
    # Map triple -> index
    triple_index = {}
    for i, t in enumerate(triples):
        triple_index[t] = i
    
    # For each vertex, list of triples it appears in
    vertex_triples = [[] for _ in range(target_n)]
    for i, (a, b, c) in enumerate(triples):
        vertex_triples[a].append(i)
        vertex_triples[b].append(i)
        vertex_triples[c].append(i)
    
    # For each 4-set (a,b,c,d), the 4 triples that form K4^(3)
    # K4^(3) on {a,b,c,d}: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
    k4_sets = []
    k4_triple_indices = []
    for quad in combinations(range(target_n), 4):
        a, b, c, d = quad
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, c, d)],
            triple_index[(b, c, d)]
        ]
        k4_sets.append(quad)
        k4_triple_indices.append(idxs)
    
    # For each 5-set (a,b,c,d,e), the 10 triples that form K5^(3)
    k5_sets = []
    k5_triple_indices = []
    for quint in combinations(range(target_n), 5):
        a, b, c, d, e = quint
        idxs = [
            triple_index[(a, b, c)],
            triple_index[(a, b, d)],
            triple_index[(a, b, e)],
            triple_index[(a, c, d)],
            triple_index[(a, c, e)],
            triple_index[(a, d, e)],
            triple_index[(b, c, d)],
            triple_index[(b, c, e)],
            triple_index[(b, d, e)],
            triple_index[(c, d, e)]
        ]
        k5_sets.append(quint)
        k5_triple_indices.append(idxs)
    
    num_k4 = len(k4_sets)
    num_k5 = len(k5_sets)
    
    # Count how many red triples in each K4 (should be < 4)
    # Count how many blue triples in each K5 (should be < 10)
    # red_count[k4_i] = number of red triples in that K4
    # blue_count[k5_i] = number of blue triples in that K5
    
    # Initialize with random colouring
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Build initial counts
    red_k4_count = [0] * num_k4
    blue_k5_count = [0] * num_k5
    
    # For each triple, determine which K4s and K5s it belongs to
    # Actually, let's build reverse maps:
    # triple_to_k4s: for each triple, list of K4 indices it's part of
    # triple_to_k5s: for each triple, list of K5 indices it's part of
    
    triple_to_k4s = [[] for _ in range(num_triples)]
    for k4_i, idxs in enumerate(k4_triple_indices):
        for tri_idx in idxs:
            triple_to_k4s[tri_idx].append(k4_i)
    
    triple_to_k5s = [[] for _ in range(num_triples)]
    for k5_i, idxs in enumerate(k5_triple_indices):
        for tri_idx in idxs:
            triple_to_k5s[tri_idx].append(k5_i)
    
    # Build initial counts
    for tri_idx in range(num_triples):
        if colors[tri_idx] == 0:  # red
            for k4_i in triple_to_k4s[tri_idx]:
                red_k4_count[k4_i] += 1
        else:  # blue
            for k5_i in triple_to_k5s[tri_idx]:
                blue_k5_count[k5_i] += 1
    
    # Violation count: number of K4s with all 4 red + number of K5s with all 10 blue
    def compute_violations(red_k4_count, blue_k5_count):
        v = 0
        for i in range(num_k4):
            if red_k4_count[i] == 4:
                v += 1
        for i in range(num_k5):
            if blue_k5_count[i] == 10:
                v += 1
        return v
    
    current_violations = compute_violations(red_k4_count, blue_k5_count)
    best_violations = current_violations
    best_colors = colors[:]
    
    # Simulated annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Precompute for speed: for each triple, the K4s and K5s it affects
    # When we flip triple t from red to blue:
    #   For each K4 containing t: red_k4_count decreases by 1
    #   For each K5 containing t: blue_k5_count decreases by 1
    # When we flip triple t from blue to red:
    #   For each K4 containing t: red_k4_count increases by 1
    #   For each K5 containing t: blue_k5_count increases by 1
    
    # But we need to track violations carefully.
    # Let's define:
    # red_k4_count[i] = number of red triples in K4 i
    # blue_k5_count[i] = number of blue triples in K5 i
    # Violation for K4 i: red_k4_count[i] == 4
    # Violation for K5 i: blue_k5_count[i] == 10
    
    # When flipping triple t:
    # If t goes red->blue:
    #   For each K4 i containing t: red_k4_count[i] -= 1
    #     If red_k4_count[i] was 4, it becomes 3 -> violation removed (-1)
    #     If red_k4_count[i] was 3, it becomes 2 -> no change
    #   For each K5 i containing t: blue_k5_count[i] -= 1
    #     If blue_k5_count[i] was 10, it becomes 9 -> violation removed (-1)
    #     If blue_k5_count[i] was 9, it becomes 8 -> no change
    # If t goes blue->red:
    #   For each K4 i containing t: red_k4_count[i] += 1
    #     If red_k4_count[i] was 3, it becomes 4 -> violation added (+1)
    #     If red_k4_count[i] was 4, it becomes 5 -> no change (but can't be 5)
    #   For each K5 i containing t: blue_k5_count[i] += 1
    #     If blue_k5_count[i] was 9, it becomes 10 -> violation added (+1)
    #     If blue_k5_count[i] was 10, it becomes 11 -> no change
    
    # Actually max red_k4_count is 4, max blue_k5_count is 10.
    
    step = 0
    no_improve_steps = 0
    max_no_improve = 50000
    
    while time.time() < deadline and no_improve_steps < max_no_improve:
        # Pick a random triple to flip
        t = random.randrange(num_triples)
        
        # Compute delta violation
        delta_v = 0
        old_color = colors[t]
        
        if old_color == 0:  # red -> blue
            for k4_i in triple_to_k4s[t]:
                if red_k4_count[k4_i] == 4:
                    delta_v -= 1
                # red_k4_count[k4_i] -= 1 (will apply)
            for k5_i in triple_to_k5s[t]:
                if blue_k5_count[k5_i] == 10:
                    delta_v -= 1
                # blue_k5_count[k5_i] -= 1 (will apply)
        else:  # blue -> red
            for k4_i in triple_to_k4s[t]:
                if red_k4_count[k4_i] == 3:
                    delta_v += 1
                # red_k4_count[k4_i] += 1 (will apply)
            for k5_i in triple_to_k5s[t]:
                if blue_k5_count[k5_i] == 9:
                    delta_v += 1
                # blue_k5_count[k5_i] += 1 (will apply)
        
        # Acceptance decision
        if delta_v <= 0:
            accept = True
        else:
            # Metropolis criterion
            if T > T_min:
                import math
                if random.random() < math.exp(-delta_v / T):
                    accept = True
                else:
                    accept = False
            else:
                accept = False
        
        if accept:
            # Apply the flip
            colors[t] = 1 - old_color
            if old_color == 0:  # was red, now blue
                for k4_i in triple_to_k4s[t]:
                    red_k4_count[k4_i] -= 1
                for k5_i in triple_to_k5s[t]:
                    blue_k5_count[k5_i] -= 1
            else:  # was blue, now red
                for k4_i in triple_to_k4s[t]:
                    red_k4_count[k4_i] += 1
                for k5_i in triple_to_k5s[t]:
                    blue_k5_count[k5_i] += 1
            
            current_violations += delta_v
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                no_improve_steps = 0
            else:
                no_improve_steps += 1
            
            # Cool down
            T *= cooling
        else:
            no_improve_steps += 1
            T *= cooling
        
        step += 1
        
        # Check if we found a valid solution
        if best_violations == 0:
            break
    
    # Check if we have a valid solution
    if best_violations == 0:
        return {'n': target_n, 'colors': best_colors}
    
    # If we couldn't find a valid n=34, try smaller n
    # Fall back to a known small valid construction
    # For n=9, the sum mod 3 construction works
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
