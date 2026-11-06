import time
import sys
import random
import itertools
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute edge indices for all pairs (i, j) with 0 <= i < j < N
    # We need to map (i, j) to index 0..119
    # Lexicographic order: (0,1), (0,2), ..., (0,15), (1,2), ..., (15,14), (15,15) is not a pair.
    # Total edges = 16*15/2 = 120.
    
    # Let's build a lookup table for edge indices to speed up checks
    # edge_idx[i][j] = index
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles (sets of 4 vertices)
    # A 4-cycle on vertices {a,b,c,d} has 3 possible pairings of opposite edges:
    # 1. (a,b) and (c,d) are opposite? No, in a K4, the 3 perfect matchings are:
    #    (a,b)-(c,d), (a,c)-(b,d), (a,d)-(b,c)
    # Wait, the condition is "No four-cycle may have four distinct edge colors".
    # A 4-cycle in a graph usually refers to a cycle of length 4. In K4, there are 3 cycles of length 4:
    # a-b-c-d-a (edges ab, bc, cd, da)
    # a-b-d-c-a (edges ab, bd, dc, ca)
    # a-c-b-d-a (edges ac, cb, bd, da)
    # The task says "all three cycles on every four-vertex subset are checked".
    # This implies we check the 3 Hamiltonian cycles of the K4 induced by the 4 vertices.
    # Let vertices be v0, v1, v2, v3.
    # Cycle 1: v0-v1-v2-v3-v0. Edges: (v0,v1), (v1,v2), (v2,v3), (v3,v0).
    # Cycle 2: v0-v1-v3-v2-v0. Edges: (v0,v1), (v1,v3), (v3,v2), (v2,v0).
    # Cycle 3: v0-v2-v1-v3-v0. Edges: (v0,v2), (v2,v1), (v1,v3), (v3,v0).
    
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        # Cycle 2: a-b-d-c-a
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        # Cycle 3: a-c-b-d-a
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    # Convert cycles to edge indices for faster lookup
    cycle_indices = []
    for c in cycles:
        cycle_indices.append(tuple(edge_idx[u][v] for u, v in c))

    # Precompute for each edge, the list of cycles it belongs to
    # This allows faster invalidation checking when an edge color changes
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)

    def is_valid(colors):
        # Check if any cycle has 4 distinct colors
        for c_idx in cycle_indices:
            c0 = colors[c_idx[0]]
            c1 = colors[c_idx[1]]
            c2 = colors[c_idx[2]]
            c3 = colors[c_idx[3]]
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Initialize with a known valid coloring (the partition one from the seed)
    # This ensures we start with a valid state.
    def get_seed_colors():
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        return colors

    current_colors = get_seed_colors()
    best_score = len(set(current_colors))
    best_colors = current_colors.copy()
    
    # Local Search: Hill Climbing with Simulated Annealing-like acceptance
    # Strategy: 
    # 1. Pick a random edge.
    # 2. Try to assign it a new color.
    # 3. Accept if valid. If it increases distinct colors, definitely accept.
    # 4. If it decreases distinct colors, accept with probability exp(-delta/T).
    # 5. If same, accept with probability exp(-0/T) = 1 (if valid).
    
    # Let's track frequency of colors to bias selection
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1
    current_distinct = best_score
        
    random.seed(42)
    
    steps = 0
    T = 10.0  # Initial temperature
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        steps += 1
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Try to find a better color
        # Bias towards unused or rare colors to increase diversity
        # Generate a list of candidate colors, weighted by 1/(freq+1)
        # But that's expensive. Let's just pick randomly but with a twist:
        # 50% chance to pick from unused colors, 50% from used.
        
        # Optimization: Precompute lists or just pick a random color and check?
        # Picking from unused is good.
        # To avoid O(N) list creation every step, we can pick a random color in 0..119
        # and check if it's unused.
        # Or just pick a random color.
        
        # Let's pick a random color in 0..119
        new_color = random.randint(0, MAX_COLORS - 1)
            
        if new_color == old_color:
            continue
            
        # Check validity efficiently
        # Only need to check cycles involving edge 'e'
        valid = True
        for c_idx in edge_to_cycles[e]:
            ci = cycle_indices[c_idx]
            c0 = current_colors[ci[0]] if ci[0] != e else new_color
            c1 = current_colors[ci[1]] if ci[1] != e else new_color
            c2 = current_colors[ci[2]] if ci[2] != e else new_color
            c3 = current_colors[ci[3]] if ci[3] != e else new_color
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                valid = False
                break
        
        if valid:
            # Calculate new score incrementally
            old_freq = freq[old_color]
            new_freq = freq[new_color]
            
            # Update freq temporarily to calculate score
            freq[old_color] -= 1
            freq[new_color] += 1
            
            # Update distinct count
            new_distinct = current_distinct
            if old_freq == 1:
                new_distinct -= 1
            if new_freq == 0:
                new_distinct += 1
            
            # Accept if score is better, or with probability exp(-(new_score - old_score)/T) if worse
            # Wait, score is number of distinct colors. We want to maximize it.
            # So if new_distinct > current_distinct, accept.
            # If new_distinct < current_distinct, accept with prob exp((new_distinct - current_distinct)/T) = exp(-|diff|/T)
            # If new_distinct == current_distinct, accept (random walk).
            
            if new_distinct >= current_distinct:
                accept = True
            else:
                delta = new_distinct - current_distinct # negative
                accept = random.random() < np.exp(delta / T)
                
            if accept:
                current_colors[e] = new_color
                current_distinct = new_distinct
                # freq is already updated
                if new_distinct > best_score:
                    best_score = new_distinct
                    best_colors = current_colors.copy()
            else:
                # Revert
                current_colors[e] = old_color
                freq[old_color] += 1
                freq[new_color] -= 1
                # current_distinct reverts implicitly as we didn't change it
        else:
            # Revert (no change to freq needed)
            current_colors[e] = old_color
            
        # Cool down
        T *= cooling
        if T < T_min:
            T = T_min
            
    return best_colors
