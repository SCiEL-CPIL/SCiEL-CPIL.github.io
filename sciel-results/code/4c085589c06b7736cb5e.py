import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    n = 16
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {(i, j): k for k, (i, j) in enumerate(edges)}
    
    # Precompute K4s and their edges
    k4s = []
    k4_edges = []
    edge_to_k4s = [[] for _ in range(num_edges)]
    
    for quad in itertools.combinations(range(n), 4):
        a, b, c, d = quad
        e_idx = [
            edge_index[(a, b)], edge_index[(a, c)], edge_index[(a, d)],
            edge_index[(b, c)], edge_index[(b, d)], edge_index[(c, d)]
        ]
        k4s.append(quad)
        k4_edges.append(e_idx)
        for e in e_idx:
            edge_to_k4s[e].append(len(k4s) - 1)
            
    num_k4s = len(k4s)
    
    # Convert to numpy arrays for fast access
    k4_edges_np = np.array(k4_edges, dtype=np.int32) # Shape (num_k4s, 6)
    
    # Helper to count distinct colors in a K4 given the color array
    # Since we update locally, we might just recalculate the bad count for affected K4s
    # But for simplicity and robustness in the search loop, we can maintain a global bad count
    # and update it by checking only the affected K4s.
    
    def get_k4_distinct_count(colors, k4_idx):
        """Calculate distinct colors for a specific K4."""
        e_indices = k4_edges_np[k4_idx]
        k4_colors = colors[e_indices]
        return len(np.unique(k4_colors))

    def count_bad_k4s(colors):
        """Count total bad K4s. Used for initialization and final verification."""
        k4_colors = colors[k4_edges_np]
        # Sort each row to easily count uniques
        sorted_colors = np.sort(k4_colors, axis=1)
        diff = np.diff(sorted_colors, axis=1)
        unique_counts = 1 + np.sum(diff != 0, axis=1)
        bad_mask = unique_counts < 5
        return np.sum(bad_mask)

    best_result = None
    best_colors_used = 100
    
    # Fallback: The (i+j)%17 coloring is valid and uses 17 colors?
    # Actually (i+j)%17 uses at most 17 colors, but for i,j < 16, max sum is 30, so colors 0..16.
    # Let's just generate a valid baseline if search fails.
    baseline = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    # Verify baseline
    base_colors_np = np.array(baseline, dtype=np.int32)
    if count_bad_k4s(base_colors_np) == 0:
        best_colors_used = len(set(baseline))
        best_result = baseline[:]

    # Search for better colorings
    # We prioritize finding 13 colors. If not found, 14 is the next best.
    # The theoretical lower bound is 13.
    
    target_palettes = [13, 14, 15]
    
    for C in target_palettes:
        if time.time() > deadline - 5:
            break
            
        # If we already have a result with <= C colors, we don't need to search for C if C is larger
        # But we want to minimize. So we search for 13 first. If found, return.
        # If not found in reasonable time, try 14.
        
        # Local search parameters
        restarts = 50 if C == 13 else 20
        max_iterations_per_restart = 2000
        
        for r in range(restarts):
            if time.time() > deadline - 2:
                break
                
            # Initialize random coloring with C colors
            colors = np.array([random.randint(0, C - 1) for _ in range(num_edges)], dtype=np.int32)
            
            # Calculate initial bad count
            # We can do this efficiently by checking all K4s
            current_bad = count_bad_k4s(colors)
            
            if current_bad == 0:
                used_colors = len(set(colors))
                if used_colors < best_colors_used:
                    best_colors_used = used_colors
                    best_result = colors.tolist()
                
                if best_colors_used <= 13:
                    return best_result
                # If we found 14, we might still want to find 13?
                # The loop structure handles this by trying C=13 first.
                # If C=13 fails, we move to C=14. If we find a valid 14, we store it.
                # Then we continue the C=14 loop? No, if we find a valid 14, we should probably
                # try harder for 13 if we have time, but the loop structure is sequential.
                # Let's break out of the restart loop for this C if we found a valid one,
                # unless we are searching for 13 and haven't found it yet.
                # Actually, if C=13 and we found a valid one, used_colors <= 13. Return.
                # If C=14 and we found a valid one, used_colors <= 14. Store.
                # Should we keep searching for better 14s? No, 14 is the target.
                # Should we go back to 13? The loop doesn't go back.
                # So we should structure the search to spend more time on 13.
                
                if C == 13:
                    return best_result
                else:
                    break # Found a valid 14-coloring, no need to search more 14s
            
            # Local Search Loop
            for it in range(max_iterations_per_restart):
                if time.time() > deadline - 2:
                    break
                    
                # Pick a random edge
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                
                # Pick a random new color
                new_color = random.randint(0, C - 1)
                if new_color == old_color:
                    continue
                
                colors[e_idx] = new_color
                
                # Calculate delta in bad count
                # Only K4s containing e_idx are affected
                affected_k4s = edge_to_k4s[e_idx]
                
                # We need to know the bad status of these K4s BEFORE the change
                # and AFTER the change.
                # Since we only changed one edge, we can check the distinct count for each affected K4.
                
                delta_bad = 0
                for k4_idx in affected_k4s:
                    # Get colors of this K4
                    e_indices = k4_edges_np[k4_idx]
                    k4_colors = colors[e_indices]
                    distinct = len(np.unique(k4_colors))
                    is_bad_after = (distinct < 5)
                    
                    # We need to know if it was bad BEFORE.
                    # This requires either storing the bad status or reverting to check.
                    # Reverting is expensive. Storing an array `is_bad` of size num_k4s is better.
                    # Let's maintain an `is_bad` array.
                    
                    # Wait, I didn't initialize `is_bad`. Let's refactor to use an `is_bad` array.
                    # This is a common pattern in local search.
                    
                    # Let's restart this logic with an `is_bad` array.
                    # This inner loop is getting complicated to patch on the fly.
                    # Let's break and write a cleaner version of the search loop.
                    pass

            # The above inner loop logic was flawed because I didn't maintain state.
            # Let's implement a proper local search with an `is_bad` array.
            
            # --- Proper Local Search Implementation ---
            
            # Re-initialize for the proper search
            colors = np.array([random.randint(0, C - 1) for _ in range(num_edges)], dtype=np.int32)
            
            # Initialize is_bad array
            k4_colors = colors[k4_edges_np]
            sorted_colors = np.sort(k4_colors, axis=1)
            diff = np.diff(sorted_colors, axis=1)
            unique_counts = 1 + np.sum(diff != 0, axis=1)
            is_bad = unique_counts < 5
            current_bad = np.sum(is_bad)
            
            if current_bad == 0:
                used_colors = len(set(colors))
                if used_colors < best_colors_used:
                    best_colors_used = used_colors
                    best_result = colors.tolist()
                if C == 13:
                    return best_result
                else:
                    break
            
            for it in range(max_iterations_per_restart):
                if time.time() > deadline - 2:
                    break
                    
                e_idx = random.randint(0, num_edges - 1)
                old_color = colors[e_idx]
                new_color = random.randint(0, C - 1)
                if new_color == old_color:
                    continue
                
                # Update colors
                colors[e_idx] = new_color
                
                # Update is_bad for affected K4s
                affected_k4s = edge_to_k4s[e_idx]
                
                # Calculate new bad status for affected K4s
                # We can vectorize this check if we gather the colors for these K4s
                # But the list of K4s is small (~91). A loop is fine.
                
                delta_bad = 0
                for k4_idx in affected_k4s:
                    # Get colors
                    e_indices = k4_edges_np[k4_idx]
                    k4_colors = colors[e_indices]
                    distinct = len(np.unique(k4_colors))
                    new_is_bad = (distinct < 5)
                    
                    if new_is_bad != is_bad[k4_idx]:
                        if new_is_bad:
                            delta_bad += 1
                        else:
                            delta_bad -= 1
                        is_bad[k4_idx] = new_is_bad
                
                current_bad += delta_bad
                
                # Acceptance criterion:
                # If badness decreases, accept.
                # If badness stays same, accept with some probability (to escape plateaus).
                # If badness increases, reject (or accept with very low prob, but let's stick to simple hill climbing first).
                
                if current_bad == 0:
                    used_colors = len(set(colors))
                    if used_colors < best_colors_used:
                        best_colors_used = used_colors
                        best_result = colors.tolist()
                    if C == 13:
                        return best_result
                    else:
                        break # Found valid 14, stop searching for 14
                
                if delta_bad < 0 or (delta_bad == 0 and random.random() < 0.1):
                    # Accept
                    pass
                else:
                    # Reject: revert
                    colors[e_idx] = old_color
                    # Revert is_bad
                    for k4_idx in affected_k4s:
                        e_indices = k4_edges_np[k4_idx]
                        k4_colors = colors[e_indices]
                        distinct = len(np.unique(k4_colors))
                        is_bad[k4_idx] = (distinct < 5)
                    current_bad -= delta_bad

    if best_result is not None:
        return best_result
    else:
        return baseline

