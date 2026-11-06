import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n = 16
    num_edges = n * (n - 1) // 2
    
    # Precompute edge indices for all K4s
    def get_idx(u, v):
        if u > v:
            u, v = v, u
        return u * (n - 1) - u * (u - 1) // 2 + (v - u - 1)

    k4_edges = []
    for a, b, c, d in itertools.combinations(range(n), 4):
        edges = [
            get_idx(a, b), get_idx(a, c), get_idx(a, d),
            get_idx(b, c), get_idx(b, d), get_idx(c, d)
        ]
        k4_edges.append(edges)
        
    num_k4 = len(k4_edges)
    
    # Convert to numpy for vectorized operations where possible, 
    # but we need fine-grained control for incremental updates.
    k4_edges_np = np.array(k4_edges, dtype=np.int32)
    
    # Map edge index to list of K4 indices it belongs to
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k4_idx, edges in enumerate(k4_edges):
        for e in edges:
            edge_to_k4s[e].append(k4_idx)

    # State:
    # coloring: list of int, length num_edges
    # k4_counts: list of dicts {color: count}, length num_k4
    # k4_distinct: list of int, length num_k4 (number of distinct colors)
    
    def init_state(coloring, max_colors):
        k4_counts = [{} for _ in range(num_k4)]
        k4_distinct = [0] * num_k4
        for k4_idx, edges in enumerate(k4_edges):
            for e in edges:
                c = coloring[e]
                if c not in k4_counts[k4_idx]:
                    k4_counts[k4_idx][c] = 0
                    k4_distinct[k4_idx] += 1
                k4_counts[k4_idx][c] += 1
        return k4_counts, k4_distinct

    def get_violating_k4s(k4_distinct):
        return [i for i, d in enumerate(k4_distinct) if d < 5]

    def is_valid(k4_distinct):
        return all(d >= 5 for d in k4_distinct)

    # Helper to update state when an edge changes
    def apply_change(coloring, k4_counts, k4_distinct, edge_idx, new_color):
        old_color = coloring[edge_idx]
        if old_color == new_color:
            return False
        
        for k4_idx in edge_to_k4s[edge_idx]:
            counts = k4_counts[k4_idx]
            # Remove old
            counts[old_color] -= 1
            if counts[old_color] == 0:
                del counts[old_color]
                k4_distinct[k4_idx] -= 1
            
            # Add new
            if new_color not in counts:
                counts[new_color] = 0
                k4_distinct[k4_idx] += 1
            counts[new_color] += 1
            
        coloring[edge_idx] = new_color
        return True

    def revert_change(coloring, k4_counts, k4_distinct, edge_idx, old_color):
        new_color = coloring[edge_idx]
        if old_color == new_color:
            return False
        
        for k4_idx in edge_to_k4s[edge_idx]:
            counts = k4_counts[k4_idx]
            # Remove new
            counts[new_color] -= 1
            if counts[new_color] == 0:
                del counts[new_color]
                k4_distinct[k4_idx] -= 1
            
            # Add old
            if old_color not in counts:
                counts[old_color] = 0
                k4_distinct[k4_idx] += 1
            counts[old_color] += 1
            
        coloring[edge_idx] = old_color
        return True

    # 1. Initial Construction
    # Start with a random coloring using 15 colors (upper bound)
    current_max_colors = 15
    coloring = [random.randint(0, current_max_colors - 1) for _ in range(num_edges)]
    k4_counts, k4_distinct = init_state(coloring, current_max_colors)
    
    best_coloring = coloring.copy()
    best_max_colors = current_max_colors
    best_k4_distinct = k4_distinct.copy()
    
    # 2. Local Search to satisfy constraints (>= 5 colors per K4)
    # We focus on violating K4s.
    
    while time.time() < deadline - 10.0:
        violating = get_violating_k4s(k4_distinct)
        if not violating:
            break
        
        # Pick a random violating K4
        target_k4 = random.choice(violating)
        edges_in_k4 = k4_edges[target_k4]
        
        # Find edges in this K4 that can be recolored to increase distinctness
        # We want to change an edge color to a new color not present in this K4
        # or to a color that is present but allows removing another duplicate?
        # Actually, simplest: change an edge to a color NOT in the K4's current set.
        current_colors_in_k4 = set(k4_counts[target_k4].keys())
        
        # Find candidate edges
        candidates = []
        for e in edges_in_k4:
            old_c = coloring[e]
            # We can pick any new color from 0..current_max_colors-1
            # But to be effective, we should pick one not in current_colors_in_k4
            # If we pick one in current_colors_in_k4, distinct count stays same or decreases (if we break a unique count)
            # So let's prioritize colors NOT in current_colors_in_k4
            
            # If distinct < 5, we need to add a new color.
            # Let's pick a random edge and a random new color not in the set.
            pass
        
        # Better strategy: Pick a random edge in the violating K4.
        # Pick a new color uniformly from [0, current_max_colors) that is NOT in current_colors_in_k4.
        # If no such color exists (distinct == current_max_colors), we are stuck with this max_colors.
        
        available_new_colors = [c for c in range(current_max_colors) if c not in current_colors_in_k4]
        
        if available_new_colors:
            e_idx = random.choice(edges_in_k4)
            new_c = random.choice(available_new_colors)
            
            old_c = coloring[e_idx]
            # Apply change
            apply_change(coloring, k4_counts, k4_distinct, e_idx, new_c)
            
            # Check if this improved the global state (reduced total violations)
            # Since we only touched one K4's distinct count (and possibly others), 
            # we need to check if we made things worse elsewhere.
            # For simplicity in this loop, we accept if the target K4 is fixed, 
            # but we should revert if it creates new violations elsewhere.
            
            # Let's do a quick check: 
            # Did we fix the target? 
            # Did we break others?
            
            # To keep it simple and fast, let's just accept if the target K4 distinct count increased.
            # If it created new violations elsewhere, the outer loop will handle them.
            # This is a "greedy repair" approach.
            
            # However, if we accept blindly, we might oscillate.
            # Let's track a "score" = sum(max(0, 5 - distinct))
            
            # Actually, let's just do standard local search:
            # 1. Identify all violating K4s.
            # 2. Pick one.
            # 3. Try to fix it by changing one of its edges to a new color.
            # 4. If it fixes the target without breaking others (or breaking fewer), accept.
            
            # To check "breaking others", we need to know which K4s share edges with e_idx.
            # We already have edge_to_k4s.
            
            # Let's calculate the change in total penalty.
            # Penalty = sum(max(0, 5 - d_i))
            
            # This is getting complex for incremental updates in Python lists.
            # Let's stick to a simpler heuristic: 
            # Accept if the specific K4 we targeted is now valid (>=5).
            # This is "greedy". It might get stuck.
            
            # To avoid getting stuck, we can allow "worsening" moves with a probability (SA-like).
            # But let's first try pure greedy repair.
            
            if k4_distinct[target_k4] >= 5:
                # Accept
                continue
            else:
                # Revert
                revert_change(coloring, k4_counts, k4_distinct, e_idx, old_c)
        
        # If we can't fix it with a new color, we might need to recolor an edge to a DIFFERENT existing color?
        # No, that doesn't increase distinctness.
        
        # If available_new_colors is empty, it means the K4 already uses all current_max_colors.
        # This implies current_max_colors < 5? No, distinct <= current_max_colors.
        # If distinct < 5 and available_new_colors is empty, then current_max_colors < 5? 
        # Or distinct == current_max_colors < 5.
        # In our case, current_max_colors starts at 15, so available_new_colors will never be empty 
        # unless distinct == 15. But we stop when distinct >= 5.
        # So this branch is safe.
        
        # If we are stuck (e.g. oscillating), we might need to restart or use SA.
        # Let's add a step counter. If we don't improve in X steps, restart the repair phase with a new random coloring?
        # Or just keep going.
        
    # 3. Color Reduction Phase
    # Now we have a valid coloring with current_max_colors.
    # Try to reduce the number of colors used.
    
    # We want to eliminate color (current_max_colors - 1).
    # Find all edges with this color.
    
    def try_reduce_color(coloring, k4_counts, k4_distinct, color_to_eliminate, max_attempts=1000):
        edges_to_change = [i for i, c in enumerate(coloring) if c == color_to_eliminate]
        
        if not edges_to_change:
            return True # Already eliminated
        
        # We need to recolor these edges to colors in [0, color_to_eliminate - 1]
        # such that the coloring remains valid.
        
        # Strategy: Randomly pick an edge, try a random lower color, check validity.
        # If valid, keep. If not, revert.
        # Repeat until all edges are changed or we give up.
        
        # To make it efficient, we can process edges one by one.
        
        for _ in range(max_attempts):
            if not edges_to_change:
                return True
            
            # Pick a random edge from the list
            idx = random.randint(0, len(edges_to_change) - 1)
            e_idx = edges_to_change[idx]
            
            old_c = coloring[e_idx]
            
            # Try a random new color
            new_c = random.randint(0, color_to_eliminate - 1)
            
            if new_c == old_c:
                continue
            
            # Apply
            apply_change(coloring, k4_counts, k4_distinct, e_idx, new_c)
            
            # Check if still valid
            # We only need to check K4s affected by this change
            # But checking all is safer and num_k4 is small (1820)
            # Actually, checking all is O(num_k4) which is fine.
            
            if is_valid(k4_distinct):
                # Accept
                edges_to_change.pop(idx)
            else:
                # Revert
                revert_change(coloring, k4_counts, k4_distinct, e_idx, old_c)
                
        return len(edges_to_change) == 0

    # Iteratively reduce colors
    while time.time() < deadline - 5.0:
        if current_max_colors <= 13:
            break
            
        color_to_eliminate = current_max_colors - 1
        
        if try_reduce_color(coloring, k4_counts, k4_distinct, color_to_eliminate, max_attempts=500):
            # Successfully eliminated
            current_max_colors -= 1
            # Update best
            best_coloring = coloring.copy()
            best_max_colors = current_max_colors
            best_k4_distinct = k4_distinct.copy()
        else:
            # Failed to eliminate
            # Maybe try a different strategy? Or just stop?
            # Let's try a few more times with different random seeds/paths
            success = False
            for _ in range(5):
                # Reset to best known valid state
                coloring = best_coloring.copy()
                k4_counts, k4_distinct = init_state(coloring, current_max_colors)
                
                if try_reduce_color(coloring, k4_counts, k4_distinct, color_to_eliminate, max_attempts=100):
                    success = True
                    break
            
            if success:
                current_max_colors -= 1
                best_coloring = coloring.copy()
                best_max_colors = current_max_colors
                best_k4_distinct = k4_distinct.copy()
            else:
                break # Can't reduce further with this method

    # Return the best coloring found
    # Ensure it is valid
    if is_valid(best_k4_distinct):
        return best_coloring
    else:
        # Fallback: Return a known valid coloring (e.g. (i+j)%17)
        # But we need to minimize colors. If our search failed, return the best we had.
        # If even the best is invalid, return the hardcoded one.
        return [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]

# Need to import itertools
import itertools
