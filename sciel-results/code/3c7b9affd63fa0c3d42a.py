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
    deadline = start_time + 55.0
    
    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1
            
    # Precompute all 4-cycles
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cycles.append(((a,b), (b,c), (c,d), (d,a)))
        cycles.append(((a,b), (b,d), (d,c), (c,a)))
        cycles.append(((a,c), (c,b), (b,d), (d,a)))
        
    cycle_indices = []
    for c in cycles:
        cycle_indices.append(tuple(edge_idx[u][v] for u, v in c))
        
    # Map edge to cycles
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for i, c_idx in enumerate(cycle_indices):
        for e_idx in c_idx:
            edge_to_cycles[e_idx].append(i)
            
    # Helper: Check validity of a specific edge change
    def is_valid_change(colors, edge, new_color):
        # Check all cycles containing 'edge'
        for c_idx in edge_to_cycles[edge]:
            ci = cycle_indices[c_idx]
            # Get colors of the 4 edges in the cycle
            # Use new_color for the changed edge
            c0 = new_color if ci[0] == edge else colors[ci[0]]
            c1 = new_color if ci[1] == edge else colors[ci[1]]
            c2 = new_color if ci[2] == edge else colors[ci[2]]
            c3 = new_color if ci[3] == edge else colors[ci[3]]
            
            if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    # Generate Tree-Based Initial Coloring
    def get_tree_colors():
        colors = [0] * NUM_EDGES
        # Assign unique colors to nodes of the split tree
        # We simulate the split tree and assign a random unique color to each split
        assigned_colors = set()
        
        def process_partition(vertices, depth):
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            
            # Pick a unique color not used yet
            # There are 15 nodes, so we can pick from 0..119
            while True:
                c = random.randint(0, MAX_COLORS - 1)
                if c not in assigned_colors:
                    assigned_colors.add(c)
                    break
            
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = c
            
            process_partition(left, depth + 1)
            process_partition(right, depth + 1)
            
        random.seed(42) # Fixed seed for reproducibility of init
        process_partition(list(range(N)), 0)
        return colors

    # Local Search
    best_score = 0
    best_colors = None
    
    # Try multiple restarts with different random seeds for the tree init
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
            
        random.seed(42 + restart)
        np.random.seed(42 + restart)
        
        current_colors = get_tree_colors()
        current_distinct = len(set(current_colors))
        
        if current_distinct > best_score:
            best_score = current_distinct
            best_colors = current_colors.copy()
            
        freq = [0] * MAX_COLORS
        for c in current_colors:
            freq[c] += 1
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5.0:
            if no_improve > max_no_improve:
                break
                
            e = random.randint(0, NUM_EDGES - 1)
            old_color = current_colors[e]
            
            # Bias towards unused colors
            if random.random() < 0.8:
                unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                if unused:
                    new_color = random.choice(unused)
                else:
                    new_color = random.randint(0, MAX_COLORS - 1)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
                
            if new_color == old_color:
                continue
                
            if is_valid_change(current_colors, e, new_color):
                # Update freq
                freq[old_color] -= 1
                freq[new_color] += 1
                
                old_distinct = current_distinct
                if freq[old_color] == 0:
                    new_distinct = old_distinct - 1
                else:
                    new_distinct = old_distinct
                    
                if freq[new_color] == 1:
                    new_distinct += 1
                else:
                    new_distinct = new_distinct # No change if already used
                
                # Wait, if old_color freq was 1, distinct decreases by 1.
                # If new_color freq was 0, distinct increases by 1.
                # Net change:
                delta = 0
                if freq[old_color] == 0: delta -= 1
                if freq[new_color] == 1: delta += 1 # Note: freq[new_color] was 0 before increment?
                # Let's recalculate properly.
                
                # Before: freq_old, freq_new
                # After: freq_old-1, freq_new+1
                # Distinct change:
                # If freq_old == 1: -1
                # If freq_new == 0: +1
                
                new_distinct_calc = old_distinct
                if freq[old_color] == 0: # This is AFTER decrement
                    new_distinct_calc -= 1
                if freq[new_color] == 1: # This is AFTER increment
                    # If it was 0 before, it's 1 now, so +1
                    # If it was >0 before, it's >1 now, so no change
                    # We need to know if it was 0 before.
                    # freq[new_color] is now 1. So it was 0 before.
                    new_distinct_calc += 1
                else:
                    # If it was >= 1 before, it is >= 2 now. No change.
                    pass
                
                if new_distinct_calc > best_score:
                    best_score = new_distinct_calc
                    best_colors = current_colors.copy()
                    
                current_colors[e] = new_color
                current_distinct = new_distinct_calc
                no_improve = 0
            else:
                no_improve += 1
                
            T *= cooling
            if T < T_min:
                T = T_min
                
    return best_colors
