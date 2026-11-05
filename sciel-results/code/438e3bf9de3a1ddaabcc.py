import time
import random
import itertools

def construct():
    deadline = time.time() + 110  # Leave safety margin
    
    # Generate all pairs in lexicographic order
    pairs = [(i, j) for i in range(16) for j in range(i + 1, 16)]
    n_edges = len(pairs)  # 120
    
    # Map edge to index for quick lookup
    edge_index = {}
    for idx, (i, j) in enumerate(pairs):
        edge_index[(i, j)] = idx

    # Precompute all K4s (sets of 4 vertices) and their 6 edges
    k4s = list(itertools.combinations(range(16), 4))
    n_k4s = len(k4s)
    k4_edges = []
    for quad in k4s:
        edges = []
        for a, b in itertools.combinations(quad, 2):
            idx = edge_index[(a, b)]
            edges.append(idx)
        k4_edges.append(edges)
        
    # Precompute for each edge, the list of K4 indices containing it
    edge_to_k4s = [[] for _ in range(n_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # State variables
    # colours[e] is the color of edge e
    # k4_color_count[k4_idx][c] is the count of color c in K4 k4_idx
    # k4_distinct[k4_idx] is the number of distinct colors in K4 k4_idx
    
    def run_local_search(max_colors, max_iterations=2000):
        # Initialize
        colours = [random.randint(0, max_colors - 1) for _ in range(n_edges)]
        
        # Initialize counts
        k4_color_count = [[0] * max_colors for _ in range(n_k4s)]
        k4_distinct = [0] * n_k4s
        
        for k4_idx in range(n_k4s):
            for e in k4_edges[k4_idx]:
                c = colours[e]
                if k4_color_count[k4_idx][c] == 0:
                    k4_distinct[k4_idx] += 1
                k4_color_count[k4_idx][c] += 1
        
        total_violations = sum(1 for d in k4_distinct if d < 5)
        if total_violations == 0:
            return colours, max_colors
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
                
            # Find a violated K4
            violated_k4s = [i for i in range(n_k4s) if k4_distinct[i] < 5]
            if not violated_k4s:
                break
                
            k4_idx = random.choice(violated_k4s)
            edges_in_k4 = k4_edges[k4_idx]
            
            # Try to fix by changing one edge in this K4
            fixed = False
            # Shuffle edges to randomize choice
            random.shuffle(edges_in_k4)
            
            for e_idx in edges_in_k4:
                old_c = colours[e_idx]
                # Try to change to a new color not in this K4
                # We want to pick a color that is NOT in k4_color_count[k4_idx]
                # If we pick a color already in the K4, distinct count doesn't change (or might decrease if it was the last of old color and new color is same, but we are changing so new is different).
                # Actually, if we change old_c to new_c:
                # 1. Decrement count of old_c. If it hits 0, distinct decreases.
                # 2. Increment count of new_c. If it was 0, distinct increases.
                # Net change in distinct:
                # If old_c count > 1: distinct unchanged by removal.
                # If old_c count == 1: distinct decreases by 1.
                # If new_c count == 0: distinct increases by 1.
                # If new_c count > 0: distinct unchanged by addition.
                
                # We want to increase distinct count.
                # So we prefer new_c with count 0.
                # And we prefer old_c with count > 1 so we don't lose distinctness.
                
                # Let's just try all colors 0..max_colors-1
                # Optimization: Only try colors not currently in the K4 to guarantee increase?
                # If we pick a color in the K4, we might lose distinctness of old_c and not gain new_c.
                # So, prioritize colors NOT in the K4.
                
                # Get list of colors not in this K4
                colors_not_in_k4 = [c for c in range(max_colors) if k4_color_count[k4_idx][c] == 0]
                
                if not colors_not_in_k4:
                    continue # Cannot fix this K4 with a single edge change to a new color? 
                             # Actually if max_colors >= 5 and distinct < 5, there must be at least 1 color not in K4.
                             # Wait, if distinct is 4, and max_colors is 5, there is 1 color not in K4.
                             # If distinct is 3, there are 2 colors not in K4.
                             # So this list should never be empty if max_colors >= 5.
                
                new_c = random.choice(colors_not_in_k4)
                
                # Apply change
                colours[e_idx] = new_c
                
                # Update counts for all K4s containing e_idx
                affected_k4s = edge_to_k4s[e_idx]
                old_violations = 0
                new_violations = 0
                
                for ak in affected_k4s:
                    # Calculate change in distinct count for ak
                    # Remove old_c
                    if k4_color_count[ak][old_c] == 1:
                        old_violations += 1 if k4_distinct[ak] < 5 else 0 # This logic is tricky, let's just recompute distinct for affected K4s? 
                        # No, let's just update and check.
                        pass
                    
                    # Simpler: Just update the state and re-calculate violations for affected K4s
                    pass
                
                # Let's do a cleaner update
                # We need to know if total violations decreased.
                
                # Store old state of affected K4s
                old_distincts = {ak: k4_distinct[ak] for ak in affected_k4s}
                
                # Perform update
                for ak in affected_k4s:
                    # Decrement old_c
                    k4_color_count[ak][old_c] -= 1
                    if k4_color_count[ak][old_c] == 0:
                        k4_distinct[ak] -= 1
                    # Increment new_c
                    if k4_color_count[ak][new_c] == 0:
                        k4_distinct[ak] += 1
                    k4_color_count[ak][new_c] += 1
                    
                # Calculate new violations for affected K4s
                for ak in affected_k4s:
                    if k4_distinct[ak] < 5:
                        new_violations += 1
                    else:
                        # It was fixed or stayed valid
                        if old_distincts[ak] < 5:
                            pass # Was violated, now valid -> good
                        else:
                            pass # Was valid, now valid -> neutral
                            
                    # Wait, I need to compare total violations.
                    # Let's just calculate the delta.
                
                # Re-calculate total violations is O(N). 
                # Better: track total_violations variable.
                
                # Let's restart the loop logic with a clearer violation tracker.
                
                # Rollback if not beneficial? 
                # For simplicity and speed, let's just accept the move if the specific K4 is fixed, 
                # and rely on the random nature to fix others. 
                # This is a greedy approach.
                
                # To be safer, let's check if we increased total violations.
                # We can calculate the change in violations for affected K4s.
                
                delta_violations = 0
                for ak in affected_k4s:
                    old_v = 1 if old_distincts[ak] < 5 else 0
                    new_v = 1 if k4_distinct[ak] < 5 else 0
                    delta_violations += new_v - old_v
                    
                if delta_violations > 0:
                    # Rollback
                    colours[e_idx] = old_c
                    for ak in affected_k4s:
                        k4_color_count[ak][new_c] -= 1
                        if k4_color_count[ak][new_c] == 0:
                            k4_distinct[ak] -= 1
                        k4_color_count[ak][old_c] += 1
                        if k4_color_count[ak][old_c] == 1: # It was 0 before increment? No, it was 1 before decrement.
                             # Wait, if k4_color_count[ak][old_c] was 1, it became 0.
                             # So incrementing makes it 1. Distinct increases.
                             k4_distinct[ak] += 1
                    # Note: The rollback logic above is slightly complex. 
                    # Let's just store the old counts for affected K4s? 
                    # That's too much memory/state.
                    
                    # Alternative: Just accept the move if it fixes the target K4, 
                    # even if it breaks others, but prefer moves that don't break others.
                    # Given the high connectivity, random walks often work.
                    # Let's stick to: Accept if delta_violations <= 0.
                    # If delta_violations > 0, reject.
                    continue
                else:
                    # Accept move
                    fixed = True
                    break
                    
            if not fixed:
                # Try a random move? Or restart?
                # If we can't fix any violated K4 with a non-breaking move, 
                # we might be stuck.
                # Let's try a random color change on a random edge in a violated K4, 
                # accepting even if it breaks others (simulated annealing style).
                # But for now, let's just break and try a new restart.
                break
                
        # Final check
        valid = all(k4_distinct[i] >= 5 for i in range(n_k4s))
        if valid:
            return colours, max_colors
        return None, None

    best_colours = None
    best_num_colors = 100
    
    # Try 13 colors first
    for max_colors in [13, 14, 15]:
        if time.time() > deadline:
            break
            
        for trial in range(50):
            if time.time() > deadline:
                break
                
            colours, used_colors = run_local_search(max_colors)
            
            if colours is not None:
                # Verify and count actual colors used
                actual_colors = len(set(colours))
                if actual_colors < best_num_colors:
                    best_num_colors = actual_colors
                    best_colours = colours[:]
                    
                    if best_num_colors == 13:
                        return best_colours
                        
    if best_colours is not None:
        return best_colours
    else:
        # Fallback to 17-color solution
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# EVOLVE-BLOCK-END
