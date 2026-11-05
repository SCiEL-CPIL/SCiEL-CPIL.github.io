import time
import random
import sys
from itertools import combinations

def construct():
    # Target n
    target_n = 34
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s margin
    
    # Helper to generate all triples and their indices
    def get_triples(n):
        triples = []
        idx_map = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx = len(triples)
                    triples.append((a, b, c))
                    idx_map[(a, b, c)] = idx
        return triples, idx_map

    # Precomputation for n=34
    # This takes a few seconds, acceptable.
    triples, idx_map = get_triples(target_n)
    num_triples = len(triples)
    
    # Precompute 4-sets and 5-sets
    # For each 4-set, store the 4 triple indices
    # For each 5-set, store the 10 triple indices
    
    sets_4 = []
    sets_4_to_triples = []
    for a in range(target_n):
        for b in range(a+1, target_n):
            for c in range(b+1, target_n):
                for d in range(c+1, target_n):
                    t1 = idx_map[(a, b, c)]
                    t2 = idx_map[(a, b, d)]
                    t3 = idx_map[(a, c, d)]
                    t4 = idx_map[(b, c, d)]
                    sets_4.append((a, b, c, d))
                    sets_4_to_triples.append([t1, t2, t3, t4])
                    
    sets_5 = []
    sets_5_to_triples = []
    for a in range(target_n):
        for b in range(a+1, target_n):
            for c in range(b+1, target_n):
                for d in range(c+1, target_n):
                    for e in range(d+1, target_n):
                        t_indices = []
                        # Generate 10 triples from 5 vertices
                        verts = [a, b, c, d, e]
                        for i in range(5):
                            for j in range(i+1, 5):
                                for k in range(j+1, 5):
                                    t_indices.append(idx_map[(verts[i], verts[j], verts[k])])
                        sets_5.append((a, b, c, d, e))
                        sets_5_to_triples.append(t_indices)

    # Reverse mapping: for each triple, which 4-sets and 5-sets contain it?
    # And which other triples are in those sets?
    
    # triple_to_4sets[t] = list of (set_idx, other_triple_indices)
    triple_to_4sets = [[] for _ in range(num_triples)]
    for s_idx, t_list in enumerate(sets_4_to_triples):
        for k in range(4):
            t_idx = t_list[k]
            others = [t_list[j] for j in range(4) if j != k]
            triple_to_4sets[t_idx].append((s_idx, others))
            
    triple_to_5sets = [[] for _ in range(num_triples)]
    for s_idx, t_list in enumerate(sets_5_to_triples):
        for k in range(10):
            t_idx = t_list[k]
            others = [t_list[j] for j in range(10) if j != k]
            triple_to_5sets[t_idx].append((s_idx, others))

    # State
    # colors: list of 0/1
    # v4_count: number of Red K4 violations
    # v5_count: number of Blue K5 violations
    
    # Initialization
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Calculate initial violations
    v4_count = 0
    for t_list in sets_4_to_triples:
        if all(colors[t] == 0 for t in t_list):
            v4_count += 1
            
    v5_count = 0
    for t_list in sets_5_to_triples:
        if all(colors[t] == 1 for t in t_list):
            v5_count += 1
            
    best_v = v4_count + v5_count
    best_colors = colors.copy()
    
    # If we start with 0 violations, we are done
    if best_v == 0:
         return {'n': target_n, 'colors': best_colors}

    # Local Search
    # We use a simple hill climbing with random restarts or simulated annealing
    # Given the structure, simple hill climbing with random restarts often works for these sizes.
    
    max_restarts = 20
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Random restart
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Recalc violations
        v4_count = 0
        for t_list in sets_4_to_triples:
            if all(colors[t] == 0 for t in t_list):
                v4_count += 1
        v5_count = 0
        for t_list in sets_5_to_triples:
            if all(colors[t] == 1 for t in t_list):
                v5_count += 1
        
        current_v = v4_count + v5_count
        if current_v == 0:
            best_v = 0
            best_colors = colors.copy()
            break
            
        if current_v < best_v:
            best_v = current_v
            best_colors = colors.copy()
            
        # Local Search Loop
        stuck_count = 0
        while time.time() < deadline and stuck_count < 1000:
            # Pick random triple
            t_idx = random.randint(0, num_triples - 1)
            current_color = colors[t_idx]
            new_color = 1 - current_color
            
            delta_v = 0
            
            if new_color == 1: # Flipping to Blue (0->1)
                # Can create Blue K5
                # Check 5-sets containing t_idx
                for s_idx, others in triple_to_5sets[t_idx]:
                    # If all others are Blue, this becomes a violation
                    if all(colors[o] == 1 for o in others):
                        delta_v += 1
                # Cannot create Red K4 (flipping away from Red)
                
            else: # Flipping to Red (1->0)
                # Can create Red K4
                # Check 4-sets containing t_idx
                for s_idx, others in triple_to_4sets[t_idx]:
                    if all(colors[o] == 0 for o in others):
                        delta_v += 1
                # Cannot create Blue K5 (flipping away from Blue)
                
            if delta_v < 0:
                # Accept move
                colors[t_idx] = new_color
                v4_count += (1 if new_color == 0 else -1) * 0 # Wait, v4/v5 counts need precise update
                
                # Precise update:
                if new_color == 1: # 0->1
                    # v4_count unchanged
                    # v5_count increases by delta_v
                    v5_count += delta_v
                else: # 1->0
                    # v4_count increases by delta_v
                    v4_count += delta_v
                    # v5_count unchanged
                    
                current_v += delta_v
                stuck_count = 0
                if current_v == 0:
                    best_v = 0
                    best_colors = colors.copy()
                    break
                if current_v < best_v:
                    best_v = current_v
                    best_colors = colors.copy()
                    
            else:
                stuck_count += 1
                # Optional: Random walk to escape local minima
                if stuck_count > 500 and random.random() < 0.01:
                    # Accept a bad move to escape
                    colors[t_idx] = new_color
                    if new_color == 1:
                        v5_count += delta_v
                    else:
                        v4_count += delta_v
                    current_v += delta_v
                    stuck_count = 0

    # If we found a valid solution for n=34, return it
    if best_v == 0:
        return {'n': target_n, 'colors': best_colors}
        
    # If not, we should return the best valid n we can construct.
    # Since the seed is n=9, and we might not have found n=34, 
    # we should fallback to a known valid construction or the seed.
    # To be safe and ensure a score > 1.0 if possible, we can try to return 
    # a valid coloring for a smaller n if we didn't find 34.
    # However, finding ANY valid n > 9 is an improvement.
    # The local search might have found a valid n=33, 32, etc. if we implemented that loop.
    # For this implementation, we only searched n=34.
    # If n=34 fails, we return the seed result to guarantee validity.
    
    # Fallback: Seed construction
    n_seed = 9
    colors_seed = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]
    return {'n': n_seed, 'colors': colors_seed}

# EVOLVE-BLOCK-END
