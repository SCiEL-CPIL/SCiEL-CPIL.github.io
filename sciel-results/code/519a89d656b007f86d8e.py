import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2
NUM_COLORS = 120

def construct():
    start_time = time.time()
    time_limit = 85.0  # Safety margin for 90s limit
    
    # Precompute edge indices
    edges = []
    edge_index = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
            edge_index[(i, j)] = idx
            edge_index[(j, i)] = idx
            idx += 1
            
    # Precompute 4-cycles and adjacency
    # For each edge, store list of (cycle_id, other_edge_indices)
    # Actually, just store the other 3 edge indices for each cycle containing the edge.
    edge_to_neighbors_in_cycles = [[] for _ in range(NUM_EDGES)]
    
    # We need to iterate over all 4-vertex subsets and their 3 cycles.
    for v0 in range(N):
        for v1 in range(v0 + 1, N):
            for v2 in range(v1 + 1, N):
                for v3 in range(v2 + 1, N):
                    # Cycle 1: v0-v1-v2-v3-v0
                    e1 = edge_index[(v0, v1)]
                    e2 = edge_index[(v1, v2)]
                    e3 = edge_index[(v2, v3)]
                    e4 = edge_index[(v3, v0)]
                    # For e1, others are e2, e3, e4
                    edge_to_neighbors_in_cycles[e1].append((e2, e3, e4))
                    edge_to_neighbors_in_cycles[e2].append((e1, e3, e4))
                    edge_to_neighbors_in_cycles[e3].append((e1, e2, e4))
                    edge_to_neighbors_in_cycles[e4].append((e1, e2, e3))
                    
                    # Cycle 2: v0-v1-v3-v2-v0
                    e1 = edge_index[(v0, v1)]
                    e2 = edge_index[(v1, v3)]
                    e3 = edge_index[(v3, v2)]
                    e4 = edge_index[(v2, v0)]
                    edge_to_neighbors_in_cycles[e1].append((e2, e3, e4))
                    edge_to_neighbors_in_cycles[e2].append((e1, e3, e4))
                    edge_to_neighbors_in_cycles[e3].append((e1, e2, e4))
                    edge_to_neighbors_in_cycles[e4].append((e1, e2, e3))
                    
                    # Cycle 3: v0-v2-v1-v3-v0
                    e1 = edge_index[(v0, v2)]
                    e2 = edge_index[(v2, v1)]
                    e3 = edge_index[(v1, v3)]
                    e4 = edge_index[(v3, v0)]
                    edge_to_neighbors_in_cycles[e1].append((e2, e3, e4))
                    edge_to_neighbors_in_cycles[e2].append((e1, e3, e4))
                    edge_to_neighbors_in_cycles[e3].append((e1, e2, e4))
                    edge_to_neighbors_in_cycles[e4].append((e1, e2, e3))

    def is_safe_for_new_color(e_idx, colors):
        """
        Returns True if assigning a NEW color to e_idx is valid.
        This is true if and only if for ALL cycles containing e_idx,
        the other 3 edges do NOT have 3 distinct colors.
        """
        for e1, e2, e3 in edge_to_neighbors_in_cycles[e_idx]:
            c1 = colors[e1]
            c2 = colors[e2]
            c3 = colors[e3]
            if c1 != c2 and c1 != c3 and c2 != c3:
                return False
        return True

    def count_distinct_colors(colors):
        return len(set(colors))

    best_result = [0] * NUM_EDGES
    best_score = 1
    
    for attempt in range(20):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(attempt)
        
        # Initialize with random 0/1 coloring to break symmetry
        colors = [random.randint(0, 1) for _ in range(NUM_EDGES)]
        used_colors = set(colors)
        next_color = max(used_colors) + 1 if used_colors else 1
        
        # If max used is 1, next is 2. If we start with 0s, next is 1.
        # Let's start with all 0s for deterministic baseline, then randomize if needed.
        # Actually, random 0/1 is better.
        
        current_score = len(used_colors)
        
        # Local Search Phase
        iterations = 0
        max_iterations = 5000
        while time.time() - start_time < time_limit and iterations < max_iterations:
            iterations += 1
            
            # Try to add a new color
            # Find an edge that is safe for a new color
            safe_edges = []
            for e in range(NUM_EDGES):
                if is_safe_for_new_color(e, colors):
                    safe_edges.append(e)
            
            if safe_edges:
                # Pick a random safe edge
                e = random.choice(safe_edges)
                colors[e] = next_color
                used_colors.add(next_color)
                next_color += 1
                current_score += 1
            else:
                # Stuck. Need to repair.
                # Find an edge that is "almost" safe (has few bad cycles)
                # Or just pick a random edge and change its color to break a bad cycle.
                
                # Strategy: Pick a random edge that has a color > 0.
                # Change it to 0.
                # This might make some edges safe.
                
                non_zero_edges = [e for e in range(NUM_EDGES) if colors[e] > 0]
                if non_zero_edges:
                    e = random.choice(non_zero_edges)
                    # Change to a random existing color (0 or 1 or another low one)
                    # To minimize damage, change to 0.
                    old_color = colors[e]
                    colors[e] = 0
                    # If old_color was the only one, remove from used?
                    # We don't strictly need to remove, but for score counting,
                    # we should update used_colors accurately if we want to try to reuse it?
                    # No, we want to MAXIMIZE distinct colors.
                    # If we remove a color, we lose 1 from count, but might gain 2 later.
                    # For now, just track the set of currently used colors.
                    # We can recalculate score periodically.
                    
                # To avoid getting stuck in a loop of same repair,
                # we can also try to swap colors of two edges.
                # But simple reset to 0 is a good start.
                
                # Update used_colors to reflect current state?
                # It's expensive to update every time.
                # Let's just recalculate score every 100 iterations.
                if iterations % 100 == 0:
                    current_score = count_distinct_colors(colors)
                    if current_score > best_score:
                        best_score = current_score
                        best_result = colors[:]
                        # Update next_color to be above max used
                        max_used = max(colors)
                        if max_used + 1 > next_color:
                            next_color = max_used + 1
                        # Also, we should ensure next_color is not in used_colors
                        # But since we just set it to max+1, it's likely not.
                        # However, if we removed a color, max might decrease.
                        # Let's just use a counter that goes up.
                        # If we hit a color that is already used, we skip it?
                        # No, we want NEW colors.
                        # So next_color should be the smallest integer NOT in used_colors.
                        # This is O(K). K is small.
                        used_set = set(colors)
                        while next_color in used_set:
                            next_color += 1

        # Final score check
        final_score = count_distinct_colors(colors)
        if final_score > best_score:
            best_score = final_score
            best_result = colors[:]

    return best_result
