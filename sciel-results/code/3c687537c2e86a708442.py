import time
import random
import numpy as np
from collections import Counter

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    n = 16
    # Generate all edges
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(edges)  # 120
    
    # Precompute K4s and their edge indices
    # Map (i,j) to edge index
    edge_idx_map = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_idx_map[(i, j)] = idx
            idx += 1
            
    k4s = []
    k4_edge_indices = []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    e = [
                        edge_idx_map[(a, b)],
                        edge_idx_map[(a, c)],
                        edge_idx_map[(a, d)],
                        edge_idx_map[(b, c)],
                        edge_idx_map[(b, d)],
                        edge_idx_map[(c, d)]
                    ]
                    k4s.append((a, b, c, d))
                    k4_edge_indices.append(e)
                    
    num_k4s = len(k4_edge_indices) # 1820
    
    # Precompute edge_to_k4s for fast updates
    edge_to_k4s = [[] for _ in range(num_edges)]
    for k_idx, edge_idxs in enumerate(k4_edge_indices):
        for e_idx in edge_idxs:
            edge_to_k4s[e_idx].append(k_idx)

    # Helper to check if a K4 is valid
    # Valid if distinct colors >= 5
    # Since there are 6 edges, distinct colors >= 5 means:
    # - 6 distinct colors
    # - 5 distinct colors (one pair same, others distinct)
    # Invalid if:
    # - <= 4 distinct colors
    # Which happens if:
    # - Any color appears >= 3 times (max distinct = 6 - 3 + 1 = 4)
    # - Or two colors appear >= 2 times each (max distinct = 6 - 2 - 2 + 1 + 1 = 4)
    
    def is_violation(colors, k4_edges):
        c = [colors[e] for e in k4_edges]
        # Use a small local check for speed
        # Since len is 6, we can just check conditions
        # Count frequencies
        # We can do this faster than Counter for 6 elements
        # But Counter is okay if we don't call it too often in inner loop?
        # Let's try a manual check for speed in the search loop.
        
        # Check for 3 same
        # Check for 2 same AND 2 same
        # We can just use set logic or sorting?
        # Sorting 6 elements is fast.
        # Or just:
        # if len(set(c)) < 5: return True
        # This is likely fastest in Python for small lists.
        return len(set(c)) < 5

    def count_violations(colors):
        count = 0
        for e in k4_edge_indices:
            if is_violation(colors, e):
                count += 1
        return count

    # Initial solution from Inspiration 1: (i + j) % 17
    # This uses colors 1..16, 0. Total 17 colors?
    # i in 0..15, j in i+1..15
    # Max sum 15+14=29. 29%17=12.
    # Let's check distinct colors.
    init_colors = [(i + j) % 17 for i, j in edges]
    current_colors = np.array(init_colors)
    
    best_colors = current_colors.copy()
    best_num_colors = len(set(best_colors))
    # best_num_colors is likely 17? Or less?
    # The seed score is 1.0, which implies value = seed.
    # Seed value is likely the number of colors used by the seed.
    # Let's assume the seed uses ~17 colors.
    # We want to minimize this. Bounds are 13-15.
    # So we need to find a solution with 15, 14, or 13 colors.
    
    # Search strategy:
    # 1. Try to reduce colors from the initial state.
    # 2. Use local search with simulated annealing or hill climbing.
    # 3. Objective: Minimize number of distinct colors used.
    #    Constraint: All K4s must have >= 5 distinct colors.
    
    # We will maintain the color counts per K4 to speed up validation.
    # But updating them is complex.
    # Given 120 edges and 1820 K4s, a full check is 1820 * 6 ops ~ 11k ops.
    # In Python, this might take ~1ms.
    # 110s / 1ms = 110,000 steps.
    # This is feasible.
    
    # However, we can be smarter.
    # We only care about reducing the number of distinct colors.
    # Let's try to remap colors to a smaller palette [0, K-1] and see if it's valid.
    # If we find a valid coloring with K colors, we can try K-1.
    
    # Let's run a search that tries to compress the color space.
    # Start with the initial coloring.
    # Identify the set of used colors.
    # Try to map one color to another, check validity.
    
    # To make this faster, we can use the "violation" check only on affected K4s.
    # When we change color of edge e from c1 to c2:
    # Only K4s containing e are affected.
    # We can check if those K4s become invalid.
    
    # Let's implement a local search that tries to merge colors.
    
    def get_used_colors(colors):
        return set(colors)
    
    def try_merge_color(colors, target_color, source_color):
        # Change all edges with source_color to target_color
        # Return True if valid, False otherwise
        # We need to check validity of K4s containing any edge that changed color.
        # Actually, just check all K4s containing any edge with source_color.
        
        # Find edges with source_color
        edges_to_change = [i for i in range(num_edges) if colors[i] == source_color]
        
        if not edges_to_change:
            return True, colors
        
        # Affected K4s
        affected_k4s = set()
        for e in edges_to_change:
            for k_idx in edge_to_k4s[e]:
                affected_k4s.add(k_idx)
        
        # Apply change temporarily
        for e in edges_to_change:
            colors[e] = target_color
            
        # Check validity of affected K4s
        valid = True
        for k_idx in affected_k4s:
            if is_violation(colors, k4_edge_indices[k_idx]):
                valid = False
                break
                
        if not valid:
            # Revert
            for e in edges_to_change:
                colors[e] = source_color
            return False, colors
        else:
            return True, colors

    # Main search loop
    # We will try to reduce the number of colors.
    # Start with K = len(set(init_colors))
    # Try to reduce K by merging colors.
    
    current_palette = get_used_colors(current_colors)
    
    # Sort colors by frequency, try to merge rare ones first?
    # Or just random.
    
    max_iterations = 50000
    iter_count = 0
    
    while time.time() < deadline and iter_count < max_iterations:
        iter_count += 1
        
        used_colors = list(get_used_colors(current_colors))
        if len(used_colors) <= 13:
            # We hit the lower bound. Stop.
            break
            
        # Pick a random color to eliminate
        # Prefer rare colors?
        # Let's just pick random from used colors, but ensure it's not the only one?
        # If we pick a color, we merge it into another random color.
        
        source_color = random.choice(used_colors)
        # Pick a target color different from source
        other_colors = [c for c in used_colors if c != source_color]
        if not other_colors:
            break
        target_color = random.choice(other_colors)
        
        # Try merge
        success, new_colors = try_merge_color(current_colors, target_color, source_color)
        
        if success:
            current_colors = new_colors
            # Update best
            new_num_colors = len(set(current_colors))
            if new_num_colors < best_num_colors:
                best_colors = current_colors.copy()
                best_num_colors = new_num_colors
                # Optional: re-randomize or continue
        else:
            # Failed to merge.
            # Maybe do a random local move to escape local minima?
            # Change a random edge to a random color from the palette.
            if random.random() < 0.1: # Occasionally
                e_idx = random.randint(0, num_edges - 1)
                old_c = current_colors[e_idx]
                new_c = random.choice(list(get_used_colors(current_colors)))
                
                if new_c != old_c:
                    # Check validity of affected K4s
                    affected = set(edge_to_k4s[e_idx])
                    current_colors[e_idx] = new_c
                    valid = True
                    for k_idx in affected:
                        if is_violation(current_colors, k4_edge_indices[k_idx]):
                            valid = False
                            break
                    if not valid:
                        current_colors[e_idx] = old_c
                    else:
                        # Accept
                        pass

    # Final check: ensure best_colors is valid
    # If best_colors is invalid (shouldn't happen if logic is correct), fallback to initial
    # Also, we want to minimize colors.
    # The best_colors found might have more colors than a fresh random search?
    # Unlikely, but let's ensure we return a valid one.
    
    # Verify best_colors
    if count_violations(best_colors) > 0:
        # Fallback to initial
        best_colors = np.array(init_colors)
        
    return best_colors.tolist()
