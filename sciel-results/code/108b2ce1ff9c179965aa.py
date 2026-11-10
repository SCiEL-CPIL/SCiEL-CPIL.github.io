import random
import time
import itertools

def construct():
    start_time = time.time()
    limit = 145.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_to_idx = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = list(itertools.combinations(range(n), 4))
    k4_edges = []
    for k4 in k4s:
        k4_edges.append([edge_to_idx[(a, b)] for a, b in itertools.combinations(k4, 2)])
        
    num_k4 = len(k4s)
    
    # Precompute for each edge, which K4s it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, e_list in enumerate(k4_edges):
        for e_idx in e_list:
            edge_to_k4s[e_idx].append(k4_idx)

    def initialize_state(coloring):
        """Initialize state arrays from a coloring."""
        k4_color_count = [{} for _ in range(num_k4)]
        k4_distinct = [0] * num_k4
        
        for k4_idx, e_list in enumerate(k4_edges):
            color_counts = {}
            for e_idx in e_list:
                c = coloring[e_idx]
                color_counts[c] = color_counts.get(c, 0) + 1
            k4_color_count[k4_idx] = color_counts
            k4_distinct[k4_idx] = len(color_counts)
        
        return k4_color_count, k4_distinct
    
    def is_valid(k4_distinct):
        """Check if all K4s have at least 5 distinct colors."""
        for d in k4_distinct:
            if d < 5:
                return False
        return True
    
    def update_edge(coloring, k4_color_count, k4_distinct, edge_idx, old_color, new_color):
        """Update state when changing edge color."""
        for k4_idx in edge_to_k4s[edge_idx]:
            cc = k4_color_count[k4_idx]
            old_distinct = k4_distinct[k4_idx]
            
            # Remove old color
            if old_color in cc:
                cc[old_color] -= 1
                if cc[old_color] == 0:
                    del cc[old_color]
                    k4_distinct[k4_idx] = old_distinct - 1
            # Add new color
            if new_color in cc:
                cc[new_color] += 1
            else:
                cc[new_color] = 1
                k4_distinct[k4_idx] = k4_distinct[k4_idx] + 1
            coloring[edge_idx] = new_color

    def revert_edge(coloring, k4_color_count, k4_distinct, edge_idx, old_color, new_color):
        """Revert edge color change."""
        for k4_idx in edge_to_k4s[edge_idx]:
            cc = k4_color_count[k4_idx]
            old_distinct = k4_distinct[k4_idx]
            
            # Remove new color
            if new_color in cc:
                cc[new_color] -= 1
                if cc[new_color] == 0:
                    del cc[new_color]
                    k4_distinct[k4_idx] = old_distinct - 1
            # Add old color
            if old_color in cc:
                cc[old_color] += 1
            else:
                cc[old_color] = 1
                k4_distinct[k4_idx] = k4_distinct[k4_idx] + 1
            coloring[edge_idx] = old_color

    def try_eliminate_color(coloring, k4_color_count, k4_distinct, target_color):
        """Try to eliminate target_color by recoloring all its edges. Returns True if successful."""
        edges_of_color = [i for i, c in enumerate(coloring) if c == target_color]
        if not edges_of_color:
            return True  # Already eliminated
        
        # Save original state for rollback
        saved_coloring = coloring[:]
        saved_k4_cc = [dict(cc) for cc in k4_color_count]
        saved_k4_dist = k4_distinct[:]
        
        # Get available palette (excluding target_color)
        palette = [c for c in set(coloring) if c != target_color]
        if not palette:
            return False
            
        random.shuffle(edges_of_color)
        
        # Try to recolor each edge of target_color
        for e_idx in edges_of_color:
            found_new = False
            random.shuffle(palette)
            
            for new_c in palette:
                old_c = coloring[e_idx]
                
                # Try the change
                update_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
                
                if is_valid(k4_distinct):
                    found_new = True
                    break
                else:
                    # Revert this specific change
                    revert_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
            
            if not found_new:
                # Failed - rollback everything
                coloring[:] = saved_coloring
                for i in range(num_k4):
                    k4_color_count[i] = dict(saved_k4_cc[i])
                    k4_distinct[i] = saved_k4_dist[i]
                return False
        
        return True

    # Initialize with seed coloring
    best_coloring = None
    best_num_colors = 120
    
    # Seed coloring: (i + j) % 17
    seed_coloring = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    # Try multiple strategies
    strategies = [
        "seed_reduce",
        "random_15_repair",
        "random_14_repair"
    ]
    
    for strat in strategies:
        if time.time() - start_time > limit:
            break
            
        if strat == "seed_reduce":
            coloring = seed_coloring[:]
            k4_color_count, k4_distinct = initialize_state(coloring)
            
            # Try to reduce colors from 17 down to 13
            for _ in range(50): # Max 50 elimination attempts
                if time.time() - start_time > limit:
                    break
                if not is_valid(k4_distinct):
                    break
                    
                current_colors = sorted(set(coloring))
                if len(current_colors) <= 13:
                    break
                    
                # Pick a random color to eliminate
                target = random.choice(current_colors)
                if try_eliminate_color(coloring, k4_color_count, k4_distinct, target):
                    if len(set(coloring)) < best_num_colors:
                        best_num_colors = len(set(coloring))
                        best_coloring = coloring[:]
                else:
                    # If elimination fails, try a kick
                    # Randomly change a few edges to random colors from palette
                    palette = list(set(coloring))
                    for _ in range(10):
                        e_idx = random.randint(0, num_edges - 1)
                        old_c = coloring[e_idx]
                        new_c = random.choice(palette)
                        if new_c == old_c: continue
                        update_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
                        if not is_valid(k4_distinct):
                            revert_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
                            
        elif strat == "random_15_repair" or strat == "random_14_repair":
            target_colors = 15 if strat == "random_15_repair" else 14
            coloring = [random.randint(0, target_colors - 1) for _ in range(num_edges)]
            k4_color_count, k4_distinct = initialize_state(coloring)
            
            # Repair phase: fix violations
            max_repair_steps = 5000
            for step in range(max_repair_steps):
                if time.time() - start_time > limit:
                    break
                
                # Find a violating K4
                violations = [i for i, d in enumerate(k4_distinct) if d < 5]
                if not violations:
                    break
                    
                v_idx = random.choice(violations)
                e_indices = k4_edges[v_idx]
                e_idx = random.choice(e_indices)
                old_c = coloring[e_idx]
                
                # Try to change e_idx to a color that helps
                # Ideally a color not present in this K4
                current_colors_in_k4 = set(coloring[ei] for ei in e_indices)
                # Try colors 0..target_colors-1
                for new_c in range(target_colors):
                    if new_c == old_c: continue
                    update_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
                    if is_valid(k4_distinct):
                        break
                    else:
                        revert_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
                
            if is_valid(k4_distinct):
                num_used = len(set(coloring))
                if num_used < best_num_colors:
                    best_num_colors = num_used
                    best_coloring = coloring[:]
                    
    if best_coloring is None:
        return seed_coloring
        
    return best_coloring
