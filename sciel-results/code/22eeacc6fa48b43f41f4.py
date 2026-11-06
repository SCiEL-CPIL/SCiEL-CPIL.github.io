import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
DEADLINE = 45  # seconds, safety margin before 60s kill

def construct():
    start_time = time.time()
    
    # Build edge index mapping
    edge_index = {}
    edge_list = []
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            edge_list.append((i, j))
            idx += 1
    
    # Enumerate all 4-sets
    all_4sets = list(itertools.combinations(range(N), 4))
    
    # For each 4-set, precompute the three cycles as tuples of edge indices
    def get_cycles(a, b, c, d):
        e1 = (edge_index[(a,b)], edge_index[(b,c)], edge_index[(c,d)], edge_index[(d,a)])
        e2 = (edge_index[(a,b)], edge_index[(b,d)], edge_index[(d,c)], edge_index[(c,a)])
        e3 = (edge_index[(a,c)], edge_index[(c,b)], edge_index[(b,d)], edge_index[(d,a)])
        return (e1, e2, e3)
    
    # For each edge, store the list of (4set_index, cycle_index) pairs that involve it
    edge_to_4sets = [[] for _ in range(NUM_EDGES)]
    cycles_by_4set = []
    for s_idx, (a, b, c, d) in enumerate(all_4sets):
        cycles = get_cycles(a, b, c, d)
        cycles_by_4set.append(cycles)
        for c_idx, cycle in enumerate(cycles):
            for e_idx in cycle:
                edge_to_4sets[e_idx].append((s_idx, c_idx))
    
    def is_valid_cycle(colors, cycle):
        """Check if a cycle (tuple of 4 edge indices) has < 4 distinct colors"""
        c0 = colors[cycle[0]]
        c1 = colors[cycle[1]]
        c2 = colors[cycle[2]]
        c3 = colors[cycle[3]]
        if c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3:
            return False
        return True
    
    def check_edge_validity(colors, e_idx):
        """Check all 4-sets/cycles involving edge e_idx"""
        for (s_idx, c_idx) in edge_to_4sets[e_idx]:
            cycle = cycles_by_4set[s_idx][c_idx]
            if not is_valid_cycle(colors, cycle):
                return False
        return True
    
    def full_validation(colors):
        """Full validation of all cycles"""
        for s_idx in range(len(all_4sets)):
            for c_idx in range(3):
                cycle = cycles_by_4set[s_idx][c_idx]
                if not is_valid_cycle(colors, cycle):
                    return False
        return True
    
    # Precompute edge constraint counts for prioritization
    edge_constraint_counts = [(len(edge_to_4sets[e]), e) for e in range(NUM_EDGES)]
    edge_constraint_counts.sort()
    sorted_edges = [e for _, e in edge_constraint_counts]
    
    def local_search(colors, max_iterations=5000):
        """Perform local search with neutral moves and new color additions"""
        current_colors = set(colors)
        num_colors = len(current_colors)
        best_colors = colors[:]
        best_num_colors = num_colors
        no_improve_count = 0
        max_no_improve = 500
        
        for iteration in range(max_iterations):
            if time.time() - start_time > DEADLINE:
                break
            
            # Try to add a new color
            new_color_found = False
            # Pick a random new color from 15-119
            if len(current_colors) < 120:
                available_new = [c for c in range(15, 120) if c not in current_colors]
                if available_new:
                    new_color = random.choice(available_new)
                    # Try edges in random order
                    trial_edges = sorted_edges[:]
                    random.shuffle(trial_edges)
                    
                    for e_idx in trial_edges:
                        old_color = colors[e_idx]
                        colors[e_idx] = new_color
                        
                        if check_edge_validity(colors, e_idx):
                            new_color_found = True
                            break
                        
                        colors[e_idx] = old_color
            
            # Try neutral moves: recolor an edge to a different existing color
            neutral_found = False
            if not new_color_found and len(current_colors) > 1:
                e_idx = random.choice(sorted_edges)
                old_color = colors[e_idx]
                # Pick a different existing color
                other_colors = [c for c in current_colors if c != old_color]
                if other_colors:
                    new_color = random.choice(other_colors)
                    colors[e_idx] = new_color
                    
                    if check_edge_validity(colors, e_idx):
                        neutral_found = True
                    else:
                        colors[e_idx] = old_color
            
            # Update best
            current_colors = set(colors)
            num_colors = len(current_colors)
            if num_colors > best_num_colors:
                best_num_colors = num_colors
                best_colors = colors[:]
                no_improve_count = 0
            elif new_color_found or neutral_found:
                no_improve_count += 1
            else:
                no_improve_count += 1
            
            if no_improve_count > max_no_improve:
                break
        
        return best_colors, best_num_colors
    
    # Initialize with the known valid coloring: color(i,j) = i for i<j
    base_colors = [0] * NUM_EDGES
    for idx, (i, j) in enumerate(edge_list):
        base_colors[idx] = i
    
    # Verify initial coloring is valid
    if not full_validation(base_colors):
        return [0] * NUM_EDGES
    
    best_colors = base_colors[:]
    best_num_colors = len(set(base_colors))
    
    # Multiple random restarts with different initial perturbations
    random.seed(42)
    num_restarts = 0
    
    while time.time() - start_time < DEADLINE:
        # Create a perturbed starting point
        if num_restarts == 0:
            colors = base_colors[:]
        else:
            # Start from best found so far, then apply random neutral moves
            colors = best_colors[:]
            # Apply some random neutral moves to perturb
            for _ in range(random.randint(5, 20)):
                e_idx = random.choice(sorted_edges)
                old_color = colors[e_idx]
                other_colors = [c for c in set(colors) if c != old_color]
                if other_colors:
                    new_color = random.choice(other_colors)
                    colors[e_idx] = new_color
                    if check_edge_validity(colors, e_idx):
                        pass  # keep the move
                    else:
                        colors[e_idx] = old_color
        
        # Run local search from this starting point
        result_colors, result_num = local_search(colors)
        
        if result_num > best_num_colors:
            best_colors = result_colors
            best_num_colors = result_num
        
        num_restarts += 1
    
    # Final validation
    if not full_validation(best_colors):
        best_colors = base_colors[:]
    
    # Ensure all colors are in range 0..119
    for i in range(NUM_EDGES):
        if best_colors[i] < 0 or best_colors[i] > 119:
            best_colors[i] = 0
    
    return best_colors
