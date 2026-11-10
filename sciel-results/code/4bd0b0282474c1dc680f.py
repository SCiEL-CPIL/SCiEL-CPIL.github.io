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
    
    def is_valid(coloring, color_counts_per_k4):
        # color_counts_per_k4[k4_idx] is a list of 6 colors
        for k4_idx in range(num_k4):
            c_counts = [0] * 120 # max color index assumed < 120
            for c in color_counts_per_k4[k4_idx]:
                c_counts[c] += 1
            distinct = sum(1 for cnt in c_counts if cnt > 0)
            if distinct < 5:
                return False
        return True

    # Faster validity check using sets
    def get_validity(coloring, k4_edge_sets):
        # k4_edge_sets[k4_idx] is a set of colors
        for s in k4_edge_sets:
            if len(s) < 5:
                return False
        return True

    # State: coloring (list of 120 ints), k4_color_sets (list of 1820 sets)
    
    def update_state(coloring, k4_color_sets, edge_idx, old_color, new_color):
        # Remove old_color from relevant K4s
        for k4_idx in edge_to_k4s[edge_idx]:
            k4_color_sets[k4_idx].discard(old_color)
        # Add new_color to relevant K4s
        for k4_idx in edge_to_k4s[edge_idx]:
            k4_color_sets[k4_idx].add(new_color)
        coloring[edge_idx] = new_color

    def revert_state(coloring, k4_color_sets, edge_idx, old_color, new_color):
        # Remove new_color
        for k4_idx in edge_to_k4s[edge_idx]:
            k4_color_sets[k4_idx].discard(new_color)
        # Add old_color
        for k4_idx in edge_to_k4s[edge_idx]:
            k4_color_sets[k4_idx].add(old_color)
        coloring[edge_idx] = old_color

    def check_validity(k4_color_sets):
        for s in k4_color_sets:
            if len(s) < 5:
                return False
        return True

    # Initialize with a valid coloring
    # Try to start with 15 colors
    best_coloring = None
    best_num_colors = 120
    
    # Random restarts with different initial color counts
    for attempt in range(10):
        if time.time() - start_time > limit:
            break
            
        # Start with a random valid coloring
        # Use the seed coloring as a base if it's valid
        # Seed: (i+j) % 17
        init_colors = 17
        coloring = [(i + j) % init_colors for i in range(n) for j in range(i + 1, n)]
        
        # Build k4_color_sets
        k4_color_sets = [set() for _ in range(num_k4)]
        for k4_idx, e_list in enumerate(k4_edges):
            for e_idx in e_list:
                k4_color_sets[k4_idx].add(coloring[e_idx])
                
        if not check_validity(k4_color_sets):
            # If seed is not valid (it is, but just in case), randomize
            # Actually seed is valid.
            pass
            
        current_num_colors = len(set(coloring))
        
        # Local search to reduce colors
        # Strategy: Try to eliminate one color at a time
        # Or, random local search minimizing color count
        improved = True
        while improved and time.time() - start_time < limit:
            improved = False
            # Identify colors used
            used_colors = list(set(coloring))
            random.shuffle(used_colors)
            
            # Try to remove a color
            for c_to_remove in used_colors:
                if time.time() - start_time > limit:
                    break
                edges_of_color = [i for i, c in enumerate(coloring) if c == c_to_remove]
                if not edges_of_color:
                    continue
                    
                # Try to recolor all edges of this color
                success = True
                for e_idx in edges_of_color:
                    old_c = coloring[e_idx]
                    # Try other colors
                    other_colors = [c for c in used_colors if c != c_to_remove]
                    random.shuffle(other_colors)
                    
                    found_new = False
                    for new_c in other_colors:
                        # Try changing e_idx to new_c
                        update_state(coloring, k4_color_sets, e_idx, old_c, new_c)
                        if check_validity(k4_color_sets):
                            found_new = True
                            break
                        else:
                            revert_state(coloring, k4_color_sets, e_idx, old_c, new_c)
                            
                    if not found_new:
                        # Revert all changes made in this attempt for this color
                        # We need to track changes
                        # Simpler: restart the color removal attempt if one edge fails
                        # But we already modified state for previous edges in this color
                        # Let's restructure: make a copy or careful revert
                        pass 
                # This logic is getting complex with partial reverts.
                # Let's use a simpler random local search.
                break # Break out of color removal loop for now, use random search below
        
        # Random Local Search
        # Minimize number of colors
        for step in range(5000):
            if time.time() - start_time > limit:
                break
                
            e_idx = random.randint(0, num_edges - 1)
            old_c = coloring[e_idx]
            new_c = random.randint(0, 119)
            
            if new_c == old_c:
                continue
                
            update_state(coloring, k4_color_sets, e_idx, old_c, new_c)
            
            if check_validity(k4_color_sets):
                new_num_colors = len(set(coloring))
                if new_num_colors < current_num_colors:
                    current_num_colors = new_num_colors
                    # Keep change
                else:
                    # Revert to maintain quality? Or keep for diversity?
                    # Keeping valid states with same color count might help escape local minima
                    # But we want to minimize. Let's revert if not improving.
                    # Actually, random walk on valid states can help.
                    # But strict minimization is safer.
                    revert_state(coloring, k4_color_sets, e_idx, old_c, new_c)
            else:
                revert_state(coloring, k4_color_sets, e_idx, old_c, new_c)
                
        if current_num_colors < best_num_colors:
            best_num_colors = current_num_colors
            best_coloring = coloring[:]
            
        if best_num_colors <= 13:
            break
            
    if best_coloring is None:
        # Fallback to seed
        return [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        
    return best_coloring
