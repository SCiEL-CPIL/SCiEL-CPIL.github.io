import random
import time
import itertools

def construct():
    start_time = time.time()
    limit = 140.0  # Safety margin
    
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

    def count_colors(coloring):
        return len(set(coloring))
    
    # State tracking for fast validity checking
    # k4_distinct[k4_idx] = number of distinct colors in K4 k4_idx
    # k4_color_count[k4_idx][c] = count of color c in K4 k4_idx
    
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
    
    def get_violation_count(k4_distinct):
        """Return number of K4s with fewer than 5 distinct colors."""
        return sum(1 for d in k4_distinct if d < 5)
    
    def update_edge(coloring, k4_color_count, k4_distinct, edge_idx, old_color, new_color):
        """Update state when changing edge color. Returns (old_distinct_values, changed_k4s)."""
        changed = []
        for k4_idx in edge_to_k4s[edge_idx]:
            cc = k4_color_count[k4_idx]
            old_distinct = k4_distinct[k4_idx]
            changed.append((k4_idx, old_distinct))
            
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
        return changed
    
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
    
    def try_eliminate_color(coloring, k4_color_count, k4_distinct, target_color, palette):
        """Try to eliminate target_color by recoloring all its edges. Returns True if successful."""
        edges_of_color = [i for i, c in enumerate(coloring) if c == target_color]
        if not edges_of_color:
            return True  # Already eliminated
        
        # Save original state for rollback
        saved_coloring = coloring[:]
        saved_k4_cc = [dict(cc) for cc in k4_color_count]
        saved_k4_dist = k4_distinct[:]
        
        # Try to recolor each edge of target_color
        for e_idx in edges_of_color:
            found_new = False
            # Try colors in palette (excluding target_color)
            other_colors = [c for c in palette if c != target_color]
            random.shuffle(other_colors)
            
            for new_c in other_colors:
                # Try the change
                old_c = coloring[e_idx]
                changed = update_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
                
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
    
    # Try multiple restarts
    for attempt in range(20):
        if time.time() - start_time > limit:
            break
        
        # Start with seed coloring
        init_colors = 17
        coloring = [(i + j) % init_colors for i in range(n) for j in range(i + 1, n)]
        k4_color_count, k4_distinct = initialize_state(coloring)
        
        if not is_valid(k4_distinct):
            continue
        
        current_num_colors = len(set(coloring))
        
        # Phase 1: Try to eliminate colors one by one
        max_elimination_rounds = 5
        for round_idx in range(max_elimination_rounds):
            if time.time() - start_time > limit:
                break
            
            used_colors = sorted(set(coloring))
            random.shuffle(used_colors)
            
            eliminated_any = False
            for c_to_remove in used_colors:
                if time.time() - start_time > limit:
                    break
                
                if c_to_remove not in set(coloring):
                    continue
                
                palette = set(coloring)
                if try_eliminate_color(coloring, k4_color_count, k4_distinct, c_to_remove, palette):
                    eliminated_any = True
                    current_num_colors = len(set(coloring))
                    if current_num_colors <= 13:
                        break
            
            if not eliminated_any or current_num_colors <= 13:
                break
        
        # Phase 2: Random local search to find better solutions
        # Use simulated annealing style: allow some non-improving moves
        base_temp = 2.0
        for step in range(3000):
            if time.time() - start_time > limit:
                break
            
            e_idx = random.randint(0, num_edges - 1)
            old_c = coloring[e_idx]
            
            # Choose new color from current palette or a random one
            if random.random() < 0.7:
                palette = list(set(coloring))
                new_c = random.choice(palette)
            else:
                new_c = random.randint(0, 119)
            
            if new_c == old_c:
                continue
            
            changed = update_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
            
            if is_valid(k4_distinct):
                new_num_colors = len(set(coloring))
                
                # Accept if improving, or with probability based on "temperature"
                if new_num_colors < current_num_colors:
                    current_num_colors = new_num_colors
                    # Keep the change
                else:
                    # Decide whether to keep or revert
                    # Keep non-improving valid moves with some probability to escape local minima
                    if random.random() < 0.3:
                        # Keep for diversity
                        pass
                    else:
                        revert_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
            else:
                revert_edge(coloring, k4_color_count, k4_distinct, e_idx, old_c, new_c)
            
            # Update best
            if current_num_colors < best_num_colors:
                best_num_colors = current_num_colors
                best_coloring = coloring[:]
            
            if best_num_colors <= 13:
                break
        
        if best_num_colors <= 13:
            break
    
    if best_coloring is None:
        # Fallback to seed
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    return best_coloring
