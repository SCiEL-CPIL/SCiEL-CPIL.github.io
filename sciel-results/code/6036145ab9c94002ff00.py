"""
Edge-colourings of K16 in which every K4 sees five colours.
We start from the (i+j) mod 17 seed (17 colours) and use simulated annealing
to reduce the number of colours while maintaining the constraint that
every K4 spans at least 5 distinct colours.
"""

import time
import random
import numpy as np
from itertools import combinations

# Precompute all K4s and their edge indices for fast constraint checking
def _precompute_k4s():
    """Precompute all 4-vertex subsets and their 6 edge indices in the 120-edge list."""
    n = 16
    # Map (i, j) with i < j to index in the 120-element list
    edge_index = {}
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            edge_index[(i, j)] = idx
            idx += 1
    
    k4_list = []  # list of tuples of 6 edge indices
    for quad in combinations(range(n), 4):
        edges = []
        for a, b in combinations(quad, 2):
            edges.append(edge_index[(a, b)])
        k4_list.append(tuple(edges))
    
    return k4_list, edge_index

_K4S = None
_EDGE_INDEX = None

def _ensure_precomputed():
    global _K4S, _EDGE_INDEX
    if _K4S is None:
        _K4S, _EDGE_INDEX = _precompute_k4s()

def _count_k4_colors(colors, k4_edges):
    """Count distinct colors in a K4 given the 6 edge indices and color array."""
    c = [colors[e] for e in k4_edges]
    return len(set(c))

def _energy(colors):
    """
    Energy = sum over all K4s of max(0, 5 - num_distinct_colors).
    Also add a penalty for number of colors used (to minimize colors).
    """
    _ensure_precomputed()
    total_violation = 0
    for k4 in _K4S:
        nc = _count_k4_colors(colors, k4)
        if nc < 5:
            total_violation += (5 - nc)
    # Number of distinct colors used
    num_colors = len(set(colors))
    # Weight the color count to encourage fewer colors
    return total_violation * 100 + num_colors

def _is_valid(colors):
    """Check if every K4 has at least 5 distinct colors."""
    _ensure_precomputed()
    for k4 in _K4S:
        if _count_k4_colors(colors, k4) < 5:
            return False
    return True

def _num_colors_used(colors):
    return len(set(colors))

def construct():
    _ensure_precomputed()
    n_edges = 120
    n = 16
    
    # Deadline guard: must finish within ~115s to be safe (evaluator kills at 150s)
    start_time = time.time()
    deadline = start_time + 115.0
    
    # Start from the seed: (i + j) mod 17
    colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    best_colors = colors.copy()
    best_num_colors = _num_colors_used(best_colors)
    best_energy = _energy(best_colors)
    
    # Simulated annealing
    # We want to minimize the number of colors while keeping validity.
    # Strategy: start with 17 colors, try to merge colors or reassign edges to fewer colors.
    
    # First, try a greedy approach: start with m colors and see if we can find a valid coloring
    # Try m from 13 upward (since 13 <= f(16) <= 15)
    
    # But first, let's try to reduce from 17 using annealing with color merging.
    
    def _anneal_from(initial_colors, max_colors, time_limit):
        """Run simulated annealing trying to reduce colors to at most max_colors."""
        colors = initial_colors.copy()
        current_energy = _energy(colors)
        
        # Temperature schedule
        T = 50.0
        T_min = 0.01
        cooling = 0.9995
        
        best_local = colors.copy()
        best_local_energy = current_energy
        best_local_nc = _num_colors_used(best_local)
        
        random.seed(42)
        
        while T > T_min:
            if time.time() > time_limit:
                break
            
            # Random move: pick a random edge and reassign to a random color in [0, max_colors-1]
            e = random.randint(0, n_edges - 1)
            old_color = colors[e]
            new_color = random.randint(0, max_colors - 1)
            
            if new_color == old_color:
                T *= cooling
                continue
            
            colors[e] = new_color
            new_energy = _energy(colors)
            delta = new_energy - current_energy
            
            if delta <= 0 or random.random() < np.exp(-delta / T):
                current_energy = new_energy
                if current_energy < best_local_energy:
                    best_local_energy = current_energy
                    best_local = colors.copy()
                    best_local_nc = _num_colors_used(best_local)
            else:
                colors[e] = old_color
            
            T *= cooling
        
        return best_local, best_local_energy, best_local_nc
    
    # Try to find a valid coloring with 15 colors
    # First, let's try direct annealing from the seed with target 15 colors
    time_limit_15 = start_time + 90.0  # leave 25s buffer
    
    # Strategy: try annealing with max_colors = 15
    # But we need to be careful: the energy function penalizes violations heavily,
    # so we should first find a valid 17-color solution (we already have one),
    # then try to reduce.
    
    # Let's try a different approach: start with 15 colors and anneal to find valid solution
    # Initialize with random assignment to 15 colors, but bias toward the seed structure
    
    for target_colors in [15, 14, 13]:
        if time.time() > deadline - 30:
            break
        
        # Try multiple restarts
        for restart in range(3):
            if time.time() > deadline - 25:
                break
            
            # Initialize: use seed pattern but modulo target_colors
            if target_colors <= 17:
                init = np.array([(i + j) % target_colors for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            else:
                init = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            
            time_limit = min(deadline - 20, time.time() + 25.0)
            
            result_colors, result_energy, result_nc = _anneal_from(init, target_colors, time_limit)
            
            if result_nc <= target_colors and _is_valid(result_colors):
                if result_nc < best_num_colors:
                    best_colors = result_colors.copy()
                    best_num_colors = result_nc
                    best_energy = result_energy
            
            # Also check if we found a valid solution with fewer colors than target
            if result_nc < target_colors and _is_valid(result_colors):
                if result_nc < best_num_colors:
                    best_colors = result_colors.copy()
                    best_num_colors = result_nc
                    best_energy = result_energy
    
    # If we haven't found anything better than 17, do a final annealing pass to try to reduce
    if best_num_colors > 15 and time.time() < deadline - 30:
        # Try more aggressive annealing from the 17-color seed
        time_limit = min(deadline - 20, time.time() + 25.0)
        for target in [15, 14]:
            if time.time() > deadline - 20:
                break
            init = np.array([(i + j) % target for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
            result_colors, result_energy, result_nc = _anneal_from(init, target, time_limit)
            if result_nc <= target and _is_valid(result_colors):
                if result_nc < best_num_colors:
                    best_colors = result_colors.copy()
                    best_num_colors = result_nc
                    best_energy = result_energy
    
    # Final check: make sure best_colors is valid
    if not _is_valid(best_colors):
        # Fall back to the seed
        best_colors = np.array([(i + j) % 17 for i in range(n) for j in range(i + 1, n)], dtype=np.int32)
    
    # Remap colors to 0..k-1 for cleanliness (though not strictly required)
    unique_colors = sorted(set(best_colors.tolist()))
    color_map = {c: i for i, c in enumerate(unique_colors)}
    final_colors = [color_map[c] for c in best_colors.tolist()]
    
    return final_colors
