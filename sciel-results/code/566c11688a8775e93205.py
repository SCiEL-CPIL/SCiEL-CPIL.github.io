import time
import random
import itertools

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    # Baseline coloring: (i + j) % 17
    # Vertices 0..15
    # Edges (i, j) for i < j
    n = 16
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Precompute all K4s and their edge indices
    k4s = []
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        # Edges in this K4
        e_indices = []
        # Map (u,v) to index in edges list
        # We can build a lookup or just calculate index
        # Index of (i,j) is sum_{k=0}^{i-1} (15-k) + (j-i-1)
        # Or just use a dictionary for speed if needed, but 120 edges is small.
        # Let's use a dict for edge index lookup
        pass
        
    # Build edge index lookup
    edge_index = {}
    for idx, (i, j) in enumerate(edges):
        edge_index[(i, j)] = idx
        edge_index[(j, i)] = idx
        
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_indices = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)],
            edge_index[(c, d)]
        ]
        k4s.append(e_indices)

    def is_valid(colors):
        for e_indices in k4s:
            # Check distinct colors
            c_set = set()
            for idx in e_indices:
                c_set.add(colors[idx])
            if len(c_set) < 5:
                return False
        return True

    def count_colors(colors):
        return len(set(colors))

    # Initial state: Baseline
    best_colors = [(i + j) % 17 for i, j in edges]
    best_count = count_colors(best_colors)
    
    # Try to improve
    # We will use a randomized local search
    # State: colors array
    # Move: Merge two colors, or recolor an edge
    
    current_colors = best_colors[:]
    current_count = best_count
    
    # Helper to get used colors
    def get_used(colors):
        return list(set(colors))
    
    used = get_used(current_colors)
    
    # Merging strategy
    # We try to merge colors one by one.
    # To avoid getting stuck, we randomize the order of pairs.
    
    max_iter = 5000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iter:
        iter_count += 1
        if not is_valid(current_colors):
            # Should not happen if we only accept valid moves
            current_colors = best_colors[:]
            used = get_used(current_colors)
            continue
            
        # Try to merge a pair
        # Pick a random pair of used colors
        if len(used) <= 2:
            break
            
        c1, c2 = random.sample(used, 2)
        
        # Create merged version
        merged_colors = [c1 if c == c2 else c for c in current_colors]
        
        # Check validity
        if is_valid(merged_colors):
            # Success
            current_colors = merged_colors
            used = get_used(current_colors)
            current_count = len(used)
            
            if current_count < best_count:
                best_count = current_count
                best_colors = current_colors[:]
                
        else:
            # Failure. Try to repair.
            # Find a violating K4
            # We can try changing one edge in a violating K4 to a different color.
            # Which edge? One that was c2 (now c1) or c1.
            # Let's find the first violating K4 in the merged state.
            viol_k4_idx = -1
            for k_idx, e_indices in enumerate(k4s):
                c_set = set(merged_colors[idx] for idx in e_indices)
                if len(c_set) < 5:
                    viol_k4_idx = k_idx
                    break
                    
            if viol_k4_idx != -1:
                e_indices = k4s[viol_k4_idx]
                # Try to recolor one of these edges to a color NOT in c_set
                # But we want to keep the merge (c2->c1).
                # So we want to introduce a 5th color.
                # Pick a random color from the rest of the graph?
                # Or pick a color from the original current_colors that is not c1 or c2?
                # Actually, if we merge c2->c1, we lost a color.
                # To restore validity, we need to make one edge in this K4 a DIFFERENT color.
                # Let's pick a random edge in this K4.
                # And assign it a random color from the global palette (0..119) 
                # that is NOT in the current K4's colors.
                
                # To minimize total colors, we should prefer existing colors.
                # Let's try colors from `used` first.
                candidate_colors = [c for c in used if c not in c_set]
                if not candidate_colors:
                    # No existing color works? Then we might need a new one, 
                    # but that increases count. 
                    # Maybe this merge is just bad. Skip.
                    continue
                    
                # Pick a random edge from the K4
                edge_to_change = random.choice(e_indices)
                
                # Try each candidate color
                found_repair = False
                for new_c in candidate_colors:
                    test_colors = merged_colors[:]
                    test_colors[edge_to_change] = new_c
                    if is_valid(test_colors):
                        current_colors = test_colors
                        # The count of colors might stay same or change
                        # c2 is still merged into c1, but we added new_c?
                        # If new_c was already in used, count stays same (c2 removed, new_c existed).
                        # If new_c was not in used? It was in `used` list, so it existed.
                        # So count decreases by 1 (c2 gone).
                        used = get_used(current_colors)
                        current_count = len(used)
                        
                        if current_count < best_count:
                            best_count = current_count
                            best_colors = current_colors[:]
                            
                        found_repair = True
                        break
                        
                # If no repair found, we revert to pre-merge state (which is current_colors)
                # Wait, current_colors is still the pre-merge state.
                # So we just skip this merge attempt.

    return best_colors
