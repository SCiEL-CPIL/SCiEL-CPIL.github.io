"""
Strategy: 
1. Population-Based Multi-Start Local Search.
2. Constraint-Guided Moves: For each edge, compute the set of 'safe' colors 
   (colors that do not create a rainbow cycle in any cycle containing that edge).
   This is derived by intersecting the color sets of the other 3 edges in each 
   cycle where those 3 edges are already distinct.
3. Diversity Heuristic: When selecting a safe color, prefer unused colors or 
   colors with low frequency to maximize the distinct color count.
4. Vectorized Validation: Use NumPy to quickly verify validity after a move, 
   though the constraint-guided selection makes invalid moves rare.
"""

import time
import numpy as np
import random
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge indices and cycle structures
def _get_edge_indices():
    """Return edge index mapping and cycle structures."""
    idx_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            idx_map[(i, j)] = idx
            idx_map[(j, i)] = idx
            idx += 1
    
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        cycles.append((idx_map[(a,b)], idx_map[(b,c)], idx_map[(c,d)], idx_map[(d,a)]))
        # Cycle 2: a-b-d-c-a
        cycles.append((idx_map[(a,b)], idx_map[(b,d)], idx_map[(c,d)], idx_map[(a,c)]))
        # Cycle 3: a-c-b-d-a
        cycles.append((idx_map[(a,c)], idx_map[(b,c)], idx_map[(b,d)], idx_map[(a,d)]))
    
    cycles_np = np.array(cycles, dtype=np.int32)
    
    # For each edge, find which cycles contain it and at which position
    # edge_cycle_info[e] = list of (cycle_idx, pos_in_cycle)
    edge_cycle_info = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for pos in range(4):
            e = cycle[pos]
            edge_cycle_info[e].append((ci, pos))
    
    return idx_map, cycles_np, edge_cycle_info

IDX_MAP, CYCLES, EDGE_CYCLE_INFO = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _is_valid_vectorized(colors_np):
    """
    Check if colors_np (shape NEDGES,) is valid.
    Returns True if no cycle has 4 distinct colors.
    """
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _initial_valid_coloring():
    """
    Generate a valid 2-coloring.
    Split vertices into two sets of 8.
    Edges within set 0: color 0
    Edges within set 1: color 1
    Edges between sets: color 0
    This is valid because any 4-cycle will have at most 2 colors.
    """
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _local_search(colors, deadline, rng):
    """
    Perform local search on a single coloring.
    Uses constraint-guided moves to ensure validity and maximize diversity.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    
    # Precompute color frequencies for heuristic
    # We will update this incrementally or recompute if needed. 
    # Given N=120, recomputing is fast, but let's try to be smart.
    
    start_time = time.time()
    stagnation_counter = 0
    
    while time.time() < deadline:
        improved = False
        # Shuffle edge order to avoid bias
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() >= deadline:
                break
            
            old_color = colors[e]
            
            # 1. Determine Allowed Colors for edge e
            # Allowed colors are those that do NOT create a rainbow in any cycle containing e.
            # A cycle becomes rainbow if the other 3 edges have 3 distinct colors, 
            # and we pick a 4th distinct color.
            # So, if other 3 are distinct {c1, c2, c3}, we MUST pick from {c1, c2, c3}.
            
            allowed_colors = None  # None means unconstrained (all colors allowed)
            
            for ci, pos in EDGE_CYCLE_INFO[e]:
                cycle = CYCLES[ci]
                other_edges = [cycle[k] for k in range(4) if k != pos]
                other_colors = [colors[oe] for oe in other_edges]
                other_set = set(other_colors)
                
                if len(other_set) == 3:
                    # Constraint: color(e) must be in other_set
                    if allowed_colors is None:
                        allowed_colors = set(other_set)
                    else:
                        allowed_colors &= other_set
                        
                        if not allowed_colors:
                            # No valid color exists for this edge given current state of others
                            # This edge is "blocked". We cannot change it without changing others.
                            # Break out of checking this edge.
                            break
            
            if allowed_colors is not None and not allowed_colors:
                continue # Edge is blocked, skip
            
            # 2. Select a candidate color
            if allowed_colors is not None:
                candidates = list(allowed_colors)
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue
            else:
                # Unconstrained: All colors are safe regarding local cycles.
                # We want to maximize distinct colors.
                # Strategy: 
                # - If we have room, try to introduce a completely new color.
                # - Otherwise, try to balance frequencies or just pick randomly from existing.
                
                # Get current color counts
                # This is O(NEDGES). Doing it inside the loop is O(NEDGES^2) per pass.
                # N=120, so 14400 ops. Very fast.
                color_counts = np.bincount(colors, minlength=MAX_COLOR)
                
                # Find unused colors
                unused_colors = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
                
                if unused_colors:
                    # Prefer unused colors to increase distinct count
                    # Pick one randomly from unused
                    candidates = unused_colors
                else:
                    # All colors used. Try to pick a color that is under-represented?
                    # Or just pick randomly to shuffle?
                    # Picking randomly might lead to less diversity if we collapse.
                    # Let's pick from colors with min frequency to maintain spread.
                    min_freq = np.min(color_counts[color_counts > 0]) if np.any(color_counts > 0) else 0
                    candidates = [c for c in range(MAX_COLOR) if color_counts[c] == min_freq]
                
                # Remove old color if it's in candidates (we want a change)
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue
            
            new_color = rng.choice(candidates)
            
            # 3. Apply move and check global validity (safety net)
            # The constraint logic above should guarantee local validity, 
            # but let's verify.
            colors[e] = new_color
            
            if _is_valid_vectorized(colors):
                current_distinct = _count_distinct(colors)
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()
                    improved = True
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
            else:
                # This should be rare if constraint logic is correct.
                # Revert
                colors[e] = old_color
                stagnation_counter += 1
        
        # If no improvement in this pass, try a perturbation
        if not improved:
            # Randomly change a few edges to escape local minima
            # This might break validity, so we check and revert
            num_perturb = rng.randint(1, 3)
            for _ in range(num_perturb):
                e = rng.randint(0, NEDGES - 1)
                old_c = colors[e]
                new_c = rng.randint(0, MAX_COLOR - 1)
                colors[e] = new_c
                if _is_valid_vectorized(colors):
                    d = _count_distinct(colors)
                    if d > best_distinct:
                        best_distinct = d
                        best_colors = colors.copy()
                else:
                    colors[e] = old_c
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    # Use a fixed seed for reproducibility in the first run, but maybe randomize for robustness?
    # The task implies a single run. Let's use a fixed seed for consistency in development,
    # but for the final submission, a fixed seed is fine as long as it's good.
    # Actually, randomizing the seed per run might be safer against bad luck, 
    # but the evaluator runs it once. Let's stick to a robust strategy.
    
    rng = random.Random(42)
    
    # Multi-start local search
    # We allocate time for multiple independent searches.
    # Each search runs until a sub-deadline.
    
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Generate initial valid coloring
        colors = _initial_valid_coloring()
        
        # Add some random valid perturbations to diversify the starting point
        # Try to introduce some new colors early on
        for _ in range(rng.randint(5, 15)):
            e = rng.randint(0, NEDGES - 1)
            # Pick a random new color
            new_color = rng.randint(2, MAX_COLOR - 1) # Avoid 0 and 1 initially
            colors[e] = new_color
            if _is_valid_vectorized(colors):
                pass # Keep it
            else:
                colors[e] = 0 # Revert to safe
        
        # Run local search with time limit
        # Give each restart at least 2 seconds, max 10 seconds?
        # If time is tight, do fewer, longer searches.
        local_deadline = min(time.time() + 10.0, deadline - 2.0)
        
        colors, distinct = _local_search(colors, local_deadline, rng)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
            
        num_restarts += 1
        
        # If we've done a few restarts and are close to deadline, break
        if num_restarts >= 5 and time.time() > deadline - 10.0:
            break

    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
    
    return best_overall_colors.tolist()

# EVOLVE-BLOCK-END
