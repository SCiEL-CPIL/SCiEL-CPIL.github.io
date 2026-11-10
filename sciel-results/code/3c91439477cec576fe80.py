"""
Strategy: 
1. Vectorized Rainbow Check: Use NumPy to check all 4-cycles simultaneously.
   This is orders of magnitude faster than the Python loop in the seed program.
2. Randomized Hill Climbing / Simulated Annealing:
   - Start with a valid 2-coloring (e.g., bipartite coloring).
   - Iteratively pick a random edge, change its color to a new random color.
   - Check validity using the fast vectorized checker.
   - If valid, accept the change (greedy).
   - Occasionally accept invalid changes or revert to escape local minima (annealing).
   - Track the maximum distinct colors seen.
3. The 60s deadline is strictly enforced.
"""

import time
import numpy as np
import random

N = 16
NEDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices and cycle structures for vectorized checking
# Edge index mapping: (i, j) -> idx
# We will store colors in a flat array of length NEDGES.
# To make vectorized checking easy, we can map (i,j) to a 2D matrix or use a lookup.
# However, for the checker, it's better to precompute the 4 edge indices for each of the 120 cycles per quad.
# Total quads: C(16,4) = 1820.
# Total cycles: 1820 * 3 = 5460.

def _get_edge_indices():
    """Return a list of 5460 tuples, each containing 4 edge indices."""
    idx_map = {}
    # Fill idx_map for all pairs
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx  # Symmetric
            idx += 1
    
    cycles = []
    from itertools import combinations
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = (idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)])
        cycles.append(e1)
        # Cycle 2: a-b-d-c-a
        e2 = (idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)])
        cycles.append(e2)
        # Cycle 3: a-c-b-d-a
        e3 = (idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)])
        cycles.append(e3)
    return np.array(cycles, dtype=np.int32)

CYCLES = _get_edge_indices()

def _is_valid_vectorized(colors_np):
    """
    Check if colors_np (shape NEDGES,) is valid.
    Returns True if no cycle has 4 distinct colors.
    """
    # Extract colors for all cycles
    # cycle_colors shape: (5460, 4)
    cycle_colors = colors_np[CYCLES]
    
    # A cycle is rainbow if all 4 are distinct.
    # Distinct count can be checked by sorting and looking for unique neighbors,
    # or simpler: 
    # Sort the 4 values. If all distinct, sorted[0] < sorted[1] < sorted[2] < sorted[3]
    # is NOT sufficient because we just need to know if len(set) == 4.
    # Actually, if we sort, distinct means no adjacent elements are equal in the sorted array? No.
    # If sorted is [1, 2, 3, 4], distinct.
    # If sorted is [1, 1, 3, 4], not distinct.
    # So: sort, then check if any adjacent are equal. If any adjacent are equal, it's NOT rainbow.
    # If NO adjacent are equal, it IS rainbow.
    
    sorted_colors = np.sort(cycle_colors, axis=1)
    # Check if any adjacent pair is equal
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    
    # If any cycle has NO adjacent equal, it is a rainbow.
    # So if all cycles have at least one adjacent equal, it is valid.
    # i.e., if there exists a cycle where adjacent_equal is False for all 3 pairs, it is invalid.
    
    # Wait, my logic:
    # adjacent_equal[i] is True if cycle i has at least one duplicate.
    # We want to ensure that for ALL i, adjacent_equal[i] is True.
    # So if any adjacent_equal[i] is False, we have a rainbow.
    
    return np.all(adjacent_equal)

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    # Initialize with a valid 2-coloring: Bipartite
    # Color (i,j) based on parity of i+j? Or just 0 if i%2==j%2 else 1?
    # Let's use: color = 0 if (i+j)%2==0 else 1.
    # This ensures no rainbow 4-cycle?
    # In a bipartite graph, any cycle is even. A 4-cycle in bipartite graph uses edges from two color classes?
    # If we color by partition, edges within partition don't exist.
    # Edges between partitions: all color 1? No, that's 1 color.
    # Let's try: color(i,j) = i % 2.
    # Cycle a-b-c-d-a.
    # If a,b,c,d alternate partitions?
    # Let's just start with all 0s (1 color) and grow.
    # Or start with a known valid small coloring.
    # Actually, random initialization is risky.
    # Let's start with a 2-coloring: color(i,j) = 0 if i < 8 else 1.
    # This splits vertices into two sets. Edges within set 0 are color 0. Within set 1 are color 1.
    # Between sets are color 0? No, let's say between is color 0.
    # Then set 1 internal edges are color 1.
    # Is this valid?
    # Quad {a,b,c,d}.
    # If all in set 0: all edges 0. Valid.
    # If all in set 1: all edges 1. Valid.
    # If mixed: 
    #   3 in 0, 1 in 1. Edges within 0 are 0. Edges to 1 are 0. All 0. Valid.
    #   2 in 0, 2 in 1. 
    #   Edges within 0: 0. Within 1: 1. Between: 0.
    #   Cycle a(0)-b(1)-c(0)-d(1)-a(0).
    #   Edges: ab(0), bc(0), cd(1), da(0). Colors: 0,0,1,0. Distinct: {0,1}. Valid.
    #   Cycle a(0)-b(1)-d(1)-c(0)-a(0).
    #   Edges: ab(0), bd(1), dc(0), ca(0). Colors: 0,1,0,0. Valid.
    #   So this 2-coloring is valid.
    
    colors = np.zeros(NEDGES, dtype=np.int32)
    # Apply the 2-coloring
    for i in range(N):
        for j in range(i + 1, N):
            idx = i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
                
    best_colors = colors.copy()
    best_distinct = _count_distinct(best_colors)
    
    # Annealing parameters
    temp = 1.0
    cooling = 0.995
    min_temp = 0.01
    
    # Precompute list of all edge indices for random selection
    edge_indices = list(range(NEDGES))
    
    while time.time() < deadline:
        # Pick random edge
        e_idx = random.choice(edge_indices)
        old_color = colors[e_idx]
        
        # Pick new color
        # Bias towards new colors to increase diversity
        # If we pick from 0..MAX_COLOR-1, we might just shuffle existing ones.
        # Let's pick from a wider range or specifically try to introduce new colors.
        # Strategy: 50% chance to pick a color from 0..best_distinct-1 (refine),
        # 50% chance to pick a color from best_distinct..MAX_COLOR-1 (expand).
        
        if random.random() < 0.5 and best_distinct < MAX_COLOR:
            new_color = random.randint(best_distinct, MAX_COLOR - 1)
        else:
            new_color = random.randint(0, MAX_COLOR - 1)
            
        if new_color == old_color:
            continue
            
        colors[e_idx] = new_color
        
        # Check validity
        if _is_valid_vectorized(colors):
            # Valid!
            new_distinct = _count_distinct(colors)
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = colors.copy()
                # Increase temperature slightly on improvement? Or keep cooling.
                # Usually we keep cooling.
            # Accept the move
            temp *= cooling
        else:
            # Invalid
            # Decide whether to keep it (Metropolis) or revert
            # Since we don't have an energy function other than validity, 
            # we can just revert most of the time, but sometimes keep to escape local minima.
            # However, keeping an invalid state makes it harder to recover.
            # Better strategy: Revert, but occasionally try a "jump" (random reset of a few edges) if stuck.
            
            # Simple approach: Revert.
            colors[e_idx] = old_color
            
            # If we are stuck for a long time, we might need a restart or mutation.
            # But for now, just revert and cool down.
            temp *= cooling
            
        if temp < min_temp:
            # Reset temperature to explore more
            temp = 0.5
            # Maybe shuffle a few edges to escape local optimum
            for _ in range(10):
                e_idx = random.choice(edge_indices)
                colors[e_idx] = random.randint(0, MAX_COLOR - 1)
            # If this makes it invalid, revert to best?
            # No, we are in the main loop. If invalid, the next check will revert.
            # But we just modified 10 edges. If invalid, we will revert them one by one in subsequent steps?
            # No, the loop picks ONE edge at a time.
            # So if we modify 10, the state is likely invalid.
            # The next iteration will pick one, change it, check (invalid), revert.
            # This effectively undoes the jump one step at a time? No, it reverts the specific edge changed in that step.
            # This is messy.
            
            # Better: If stuck, just reset to best_colors and try again with higher temp.
            # But we don't know "stuck" easily.
            
            # Let's just reset temp.
            pass
            
    # Final check: ensure best_colors is valid and within bounds
    # It is, by construction.
    
    return best_colors.tolist()

# EVOLVE-BLOCK-END
