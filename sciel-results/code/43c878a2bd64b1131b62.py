import time
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 60s limit
    
    # Precompute edge index
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return (N - 1) * i - i * (i - 1) // 2 + (j - i - 1)
    
    # Precompute all 4-vertex subsets and their three 4-cycles
    four_sets = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    e1 = (edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a))
                    e2 = (edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a))
                    e3 = (edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a))
                    four_sets.append((e1, e2, e3))
    
    num_four_sets = len(four_sets)
    
    # Precompute for each edge, which 4-cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci in range(num_four_sets):
        e1, e2, e3 = four_sets[ci]
        for e in e1:
            edge_to_cycles[e].append(ci)
        for e in e2:
            edge_to_cycles[e].append(ci)
        for e in e3:
            edge_to_cycles[e].append(ci)
    
    # Convert to numpy arrays for vectorized operations
    # Each four_set has 3 cycles, each cycle has 4 edges
    # Shape: (num_four_sets, 3, 4)
    cycle_edges = np.zeros((num_four_sets, 3, 4), dtype=np.int32)
    for ci in range(num_four_sets):
        e1, e2, e3 = four_sets[ci]
        cycle_edges[ci, 0] = e1
        cycle_edges[ci, 1] = e2
        cycle_edges[ci, 2] = e3
    
    def is_valid(colors):
        """Check if coloring is valid (no 4-cycle has 4 distinct colors)."""
        colors_arr = np.array(colors)
        for ci in range(num_four_sets):
            for c in range(3):
                e_tuple = cycle_edges[ci, c]
                c1 = colors_arr[e_tuple[0]]
                c2 = colors_arr[e_tuple[1]]
                c3 = colors_arr[e_tuple[2]]
                c4 = colors_arr[e_tuple[3]]
                if len({c1, c2, c3, c4}) == 4:
                    return False
        return True
    
    def check_single_cycle(colors, e_tuple):
        """Check if a specific 4-cycle is valid."""
        c1 = colors[e_tuple[0]]
        c2 = colors[e_tuple[1]]
        c3 = colors[e_tuple[2]]
        c4 = colors[e_tuple[3]]
        return len({c1, c2, c3, c4}) != 4
    
    def count_colors(colors):
        return len(set(colors))
    
    # Strategy: Use a constraint-satisfaction-inspired approach with multiple restarts
    # Start with a base valid coloring, then use randomized local search with
    # focus on introducing new colors while maintaining validity.
    
    random.seed(42)
    
    best_colors = None
    best_num_colors = 0
    
    # Base coloring: color edge (i,j) by i
    base_colors = [0] * NUM_EDGES
    for i in range(N - 1):
        for j in range(i + 1, N):
            base_colors[edge_idx(i, j)] = i
    
    # Verify base is valid
    assert is_valid(base_colors), "Base coloring should be valid"
    
    best_colors = base_colors[:]
    best_num_colors = count_colors(best_colors)
    
    # Local search with multiple strategies
    max_iterations = 500000
    
    for restart in range(20):
        if time.time() > deadline - 2:
            break
        
        # Start from best found so far or base
        if restart == 0:
            colors = base_colors[:]
        else:
            # Perturb the best solution
            colors = best_colors[:]
            num_perturb = random.randint(5, 15)
            for _ in range(num_perturb):
                e = random.randint(0, NUM_EDGES - 1)
                colors[e] = random.randint(0, 119)
            # Check validity after perturbation; if invalid, revert to best
            if not is_valid(colors):
                colors = best_colors[:]
        
        current_num_colors = count_colors(colors)
        
        # Local search loop
        stagnation_count = 0
        max_stagnation = 5000
        
        while time.time() < deadline - 1 and stagnation_count < max_stagnation:
            # Pick a random edge
            e = random.randint(0, NUM_EDGES - 1)
            old_color = colors[e]
            
            # Generate candidate colors: prioritize unused colors and colors not in neighboring cycles
            used_colors = set(colors)
            unused_colors = [c for c in range(120) if c not in used_colors]
            
            # Build candidate list
            candidates = []
            # First, unused colors (to maximize color count)
            random.shuffle(unused_colors)
            candidates.extend(unused_colors[:20])  # Up to 20 unused colors
            # Then, some existing colors for diversification
            existing = list(used_colors)
            random.shuffle(existing)
            candidates.extend(existing[:10])
            
            # Also add some random colors
            for _ in range(5):
                candidates.append(random.randint(0, 119))
            
            random.shuffle(candidates)
            
            found = False
            for new_color in candidates:
                if new_color == old_color:
                    continue
                colors[e] = new_color
                
                # Check validity only for affected cycles
                valid = True
                for ci in edge_to_cycles[e]:
                    e1, e2, e3 = four_sets[ci]
                    if not check_single_cycle(colors, e1):
                        valid = False
                        break
                    if not check_single_cycle(colors, e2):
                        valid = False
                        break
                    if not check_single_cycle(colors, e3):
                        valid = False
                        break
                
                if valid:
                    found = True
                    break
                else:
                    colors[e] = old_color
            
            if found:
                new_num_colors = count_colors(colors)
                if new_num_colors > current_num_colors:
                    current_num_colors = new_num_colors
                    stagnation_count = 0
                    if new_num_colors > best_num_colors:
                        best_colors = colors[:]
                        best_num_colors = new_num_colors
                else:
                    stagnation_count += 1
            else:
                colors[e] = old_color
                stagnation_count += 1
    
    # Final validation
    if not is_valid(best_colors):
        # Fallback to base
        for i in range(N - 1):
            for j in range(i + 1, N):
                best_colors[edge_idx(i, j)] = i
    
    return best_colors
