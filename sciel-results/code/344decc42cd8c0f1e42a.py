"""
Combined approach:
1. Structural idea from Inspiration 1: Use a flat NumPy array for colours and precompute K4 edge indices.
   This allows for faster vectorized checks and easier manipulation.
2. Refinement machinery from Current Program: A robust local search that specifically targets
   "unique" colours (colours used exactly once) to merge them into existing colours, thereby
   reducing the total count. This is a very effective heuristic for minimization.
   Additionally, we add a "polish" phase that tries to reduce the maximum colour index (compactness)
   which often correlates with finding lower colour counts or helps escape local minima.
   We also incorporate a simple simulated annealing-like acceptance for neutral moves to escape
   local minima, but strictly prioritize reducing the number of distinct colours.

The goal is to find a valid edge colouring of K16 where every K4 has >= 5 distinct colours,
minimizing the total number of colours used.
"""

import time
import random
import numpy as np
import itertools

# Precompute K4s and edge mappings globally for speed
N = 16
TOTAL_EDGES = N * (N - 1) // 2  # 120

# Map (i, j) to flat index
def get_edge_index(i, j):
    if i > j:
        i, j = j, i
    # Index calculation for upper triangle
    # 0..0: 0
    # 0..1, 0..2...
    # Formula: i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    return i * (N - 1) - (i * (i + 1) // 2) + (j - i - 1)

# Precompute all K4s as lists of edge indices
K4_EDGE_LISTS = []
for quad in itertools.combinations(range(N), 4):
    a, b, c, d = quad
    edges = [
        get_edge_index(a, b), get_edge_index(a, c), get_edge_index(a, d),
        get_edge_index(b, c), get_edge_index(b, d), get_edge_index(c, d)
    ]
    K4_EDGE_LISTS.append(edges)

NUM_K4S = len(K4_EDGE_LISTS)  # 1820

# Precompute for each edge, which K4s it belongs to
EDGE_TO_K4S = [[] for _ in range(TOTAL_EDGES)]
for k4_idx, edges in enumerate(K4_EDGE_LISTS):
    for e in edges:
        EDGE_TO_K4S[e].append(k4_idx)

# For fast validity checking, we can precompute the "other 5 edges" for each K4?
# Or just iterate the 6 edges. 1820 * 6 is small.
# To check if a change to edge e affects validity:
# We only need to check K4s containing e.
# For each such K4, we count distinct colours among its 6 edges.
# If we change e from c_old to c_new, we can update the count efficiently?
# No, set distinctness is hard to update incrementally without storing the set.
# Given small size (91 K4s per edge), we just recompute the set for the 6 edges of those K4s.

def count_distinct_colors_in_k4(k4_idx, colors):
    """
    Returns the number of distinct colours in the K4 identified by k4_idx.
    colors is a flat numpy array of length 120.
    """
    edges = K4_EDGE_LISTS[k4_idx]
    c_set = set()
    for e in edges:
        c_set.add(colors[e])
    return len(c_set)

def is_valid_coloring(colors):
    """
    Checks if the coloring is valid (every K4 has >= 5 distinct colours).
    colors: numpy array of length 120.
    """
    for k4_idx in range(NUM_K4S):
        if count_distinct_colors_in_k4(k4_idx, colors) < 5:
            return False
    return True

def get_num_colours(colors):
    """
    Returns the number of distinct colours used.
    """
    return len(np.unique(colors))

def construct():
    """
    Returns 120 integer colours in 0..119 for the pairs (i,j), 0<=i<j<16,
    in lexicographic order.
    """
    # Deadline guard
    start_time = time.time()
    timeout = 145.0  # 5s margin for 150s limit

    # Initialize with the known valid 17-colour construction: (i + j) % 17
    colors = np.zeros(TOTAL_EDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = get_edge_index(i, j)
            colors[idx] = (i + j) % 17

    # Verify initial validity (should be true)
    if not is_valid_coloring(colors):
        # Fallback: just return the initial, even if somehow invalid (shouldn't happen)
        return colors.tolist()

    best_colors = colors.copy()
    best_num_colours = get_num_colours(best_colors)
    current_violations = 0  # We start valid

    # Local Search Strategy:
    # 1. Identify "unique" colours (used exactly once).
    # 2. Try to recolor the edge with a unique colour to a non-unique colour.
    #    If valid, accept. This reduces the number of distinct colours by 1.
    # 3. If no unique colours, try "neutral" moves (recolor to an existing colour)
    #    to perturb the state and potentially create new unique colours or different structures.
    # 4. Periodically try to "compact" colours (map used colours to 0..k-1) to ensure
    #    we are tracking the minimum index range, though distinct count is the metric.

    no_improvement_count = 0
    max_no_improvement = 5000

    while time.time() - start_time < timeout:
        # Count occurrences of each colour
        unique_counts = np.unique(colors, return_counts=True)
        colour_vals, counts = unique_counts
        count_dict = dict(zip(colour_vals, counts))
        
        # Find unique colours
        unique_colours = [c for c, cnt in count_dict.items() if cnt == 1]
        
        if unique_colours:
            # Pick a random unique colour
            target_c = random.choice(unique_colours)
            
            # Find the edge with this colour
            edge_idx = np.where(colors == target_c)[0][0]
            
            # Try to change it to a non-unique colour
            non_unique_colours = [c for c, cnt in count_dict.items() if cnt > 1]
            
            if not non_unique_colours:
                # All colours are unique? (120 colours)
                # We must merge two. Pick any other colour.
                non_unique_colours = [c for c in count_dict.keys() if c != target_c]
            
            if not non_unique_colours:
                break
                
            random.shuffle(non_unique_colours)
            
            improved = False
            for new_c in non_unique_colours:
                if new_c == target_c:
                    continue
                
                # Test validity
                old_c = colors[edge_idx]
                colors[edge_idx] = new_c
                
                valid = True
                # Check all K4s containing this edge
                for k4_idx in EDGE_TO_K4S[edge_idx]:
                    if count_distinct_colors_in_k4(k4_idx, colors) < 5:
                        valid = False
                        break
                
                if valid:
                    # Accept move
                    improved = True
                    no_improvement_count = 0
                    
                    # Update best
                    curr_num = get_num_colours(colors)
                    if curr_num < best_num_colours:
                        best_num_colours = curr_num
                        best_colors = colors.copy()
                    break
                else:
                    # Revert
                    colors[edge_idx] = old_c
            
            if not improved:
                no_improvement_count += 1
                if no_improvement_count > max_no_improvement:
                    break
        else:
            # No unique colours. All colours used >= 2 times.
            # Try a neutral move to escape local minimum.
            if no_improvement_count > max_no_improvement:
                break
            
            # Pick a random edge
            edge_idx = random.randint(0, TOTAL_EDGES - 1)
            old_c = colors[edge_idx]
            
            # Pick a random colour from the set of used colours
            used_colours = list(count_dict.keys())
            if len(used_colours) <= 1:
                break
            new_c = random.choice(used_colours)
            
            if new_c == old_c:
                no_improvement_count += 1
                continue
            
            # Test validity
            colors[edge_idx] = new_c
            valid = True
            for k4_idx in EDGE_TO_K4S[edge_idx]:
                if count_distinct_colors_in_k4(k4_idx, colors) < 5:
                    valid = False
                    break
            
            if valid:
                # Accept neutral move
                # This might create a unique colour if old_c count was 2 and becomes 1,
                # or if new_c count was 1 and becomes 2 (but we picked from used, so new_c count >= 1).
                # Actually, if old_c count was 2, it becomes 1 (unique!).
                # If old_c count > 2, it stays > 1.
                # So this move has a chance to create a unique colour if old_c count was 2.
                no_improvement_count += 1 # It's not an immediate improvement in count, but it's progress
            else:
                colors[edge_idx] = old_c
                no_improvement_count += 1

    # Final Polish:
    # Even if we stopped, we can try to compact the colours to ensure we report the minimum.
    # The metric is "number of colours used", so we just need the distinct count.
    # However, the output is the colours themselves.
    # We should return the best_colors found.
    
    # One last check: try to reduce distinct count by merging any pair of colours
    # if we have time.
    if time.time() - start_time < timeout - 2.0:
        # Try to merge colours in best_colors
        candidate = best_colors.copy()
        curr_distinct = get_num_colours(candidate)
        
        # Try to merge each colour into another
        used = set(candidate)
        for c1 in list(used):
            if time.time() - start_time > timeout - 1.0:
                break
            for c2 in list(used):
                if c1 == c2:
                    continue
                # Try to change all c1 to c2
                mask = (candidate == c1)
                if not np.any(mask):
                    continue
                
                candidate[mask] = c2
                
                if is_valid_coloring(candidate):
                    new_distinct = get_num_colours(candidate)
                    if new_distinct < curr_distinct:
                        curr_distinct = new_distinct
                        best_colors = candidate.copy()
                        used = set(best_colors)
                    else:
                        candidate[mask] = c1 # Revert
                else:
                    candidate[mask] = c1 # Revert

    # Verify final solution
    if not is_valid_coloring(best_colors):
        # Fallback to initial
        for i in range(N):
            for j in range(i + 1, N):
                idx = get_edge_index(i, j)
                best_colors[idx] = (i + j) % 17

    return best_colors.tolist()

# EVOLVE-BLOCK-END
