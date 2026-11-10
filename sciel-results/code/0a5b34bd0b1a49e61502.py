import time
import random
import numpy as np
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119
NUM_COLORS = MAX_COLOR + 1  # 120

def _edge_index(i, j):
    """Map pair (i,j) with i<j to edge index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_cycles():
    """Return list of cycles, each cycle is a tuple of 4 edge indices."""
    cycles = []
    for a in range(N - 3):
        for b in range(a + 1, N - 2):
            for c in range(b + 1, N - 1):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    # Cycle 2: a-b-d-c-a
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    # Cycle 3: a-c-b-d-a
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _is_rainbow(cyc, colors):
    """Check if the cycle has 4 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    return c0 != c1 and c0 != c2 and c0 != c3 and c1 != c2 and c1 != c3 and c2 != c3

def _validate(colors):
    """Full validation of a coloring."""
    for cyc in CYCLES:
        if _is_rainbow(cyc, colors):
            return False
    return True

def _count_distinct(colors):
    return len(set(colors))

def _fast_local_search(colors, color_counts, deadline, rng):
    """
    Fast local search that tries to increase the number of distinct colors.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    current_distinct = best_distinct
    
    # Precompute edge cycle info for speed
    edge_cycle_info = []
    for e in range(NUM_EDGES):
        info = []
        for ci in EDGE_TO_CYCLES[e]:
            cycle = CYCLES[ci]
            for pos in range(4):
                if cycle[pos] == e:
                    info.append((ci, pos))
                    break
        edge_cycle_info.append(info)

    def _is_move_valid(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
        for ci, pos in edge_cycle_info[e]:
            cycle = CYCLES[ci]
            nc0 = new_color if pos == 0 else colors_arr[cycle[0]]
            nc1 = new_color if pos == 1 else colors_arr[cycle[1]]
            nc2 = new_color if pos == 2 else colors_arr[cycle[2]]
            nc3 = new_color if pos == 3 else colors_arr[cycle[3]]
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        return True

    while time.time() < deadline:
        improved = False
        # Try to introduce new colors or shuffle existing ones
        # We prioritize trying to assign unused colors to edges
        unused_colors = [c for c in range(NUM_COLORS) if color_counts[c] == 0]
        rng.shuffle(unused_colors)
        
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() >= deadline:
                break
            
            old_color = colors[e]
            
            # Try to change to an unused color
            # We limit the number of attempts per edge to keep it fast
            attempts = 0
            max_attempts = 5
            while attempts < max_attempts:
                if not unused_colors:
                    break
                new_color = unused_colors[0] # Pick first available unused
                # Note: We don't remove from unused_colors list immediately, 
                # we just check if it's still unused in the global state
                
                if _is_move_valid(e, new_color, colors):
                    # Apply move
                    color_counts[old_color] -= 1
                    color_counts[new_color] += 1
                    colors[e] = new_color
                    
                    # Update distinct count
                    if color_counts[old_color] == 0:
                        current_distinct -= 1
                    if color_counts[new_color] == 1:
                        current_distinct += 1
                    
                    # Update unused_colors list if necessary (lazy update)
                    # We just filter it occasionally or rely on color_counts
                    if new_color in unused_colors:
                        unused_colors.remove(new_color)
                        
                    if current_distinct > best_distinct:
                        best_distinct = current_distinct
                        best_colors = colors.copy()
                        improved = True
                    
                    break # Move applied, try next edge
                else:
                    attempts += 1
                    # If this unused color doesn't work, try another unused color
                    if len(unused_colors) > 1:
                        # Swap to try a different unused color
                        unused_colors[0], unused_colors[attempts] = unused_colors[attempts], unused_colors[0]
            
            # If we didn't apply a move, maybe try a random valid move to perturb?
            # For now, just skip to keep it simple and fast
            
        if not improved and time.time() < deadline - 1.0:
            # Perturbation: random valid move
            e = rng.randint(0, NUM_EDGES - 1)
            old = colors[e]
            new = rng.randint(0, MAX_COLOR)
            if new != old and _is_move_valid(e, new, colors):
                color_counts[old] -= 1
                color_counts[new] += 1
                colors[e] = new
                if color_counts[old] == 0:
                    current_distinct -= 1
                if color_counts[new] == 1:
                    current_distinct += 1

    return best_colors, best_distinct

def _greedy_seed(seed):
    """Build a valid coloring using greedy color introduction."""
    rng = random.Random(seed)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    color_counts = np.zeros(NUM_COLORS, dtype=np.int32)
    color_counts[0] = NUM_EDGES

    # Precompute edge_cycle_info
    edge_cycle_info = []
    for e in range(NUM_EDGES):
        info = []
        for ci in EDGE_TO_CYCLES[e]:
            cycle = CYCLES[ci]
            for pos in range(4):
                if cycle[pos] == e:
                    info.append((ci, pos))
                    break
        edge_cycle_info.append(info)

    def _is_move_valid(e, new_color, colors_arr):
        old_color = colors_arr[e]
        if old_color == new_color:
            return True
        for ci, pos in edge_cycle_info[e]:
            cycle = CYCLES[ci]
            nc0 = new_color if pos == 0 else colors_arr[cycle[0]]
            nc1 = new_color if pos == 1 else colors_arr[cycle[1]]
            nc2 = new_color if pos == 2 else colors_arr[cycle[2]]
            nc3 = new_color if pos == 3 else colors_arr[cycle[3]]
            if nc0 != nc1 and nc0 != nc2 and nc0 != nc3 and nc1 != nc2 and nc1 != nc3 and nc2 != nc3:
                return False
        return True

    # Introduce colors 1..119 one at a time
    for new_c in range(1, MAX_COLOR + 1):
        edge_order = list(range(NUM_EDGES))
        rng.shuffle(edge_order)
        for e in edge_order:
            if _is_move_valid(e, new_c, colors):
                old = colors[e]
                colors[e] = new_c
                color_counts[old] -= 1
                color_counts[new_c] += 1
                break
    return colors, color_counts, edge_cycle_info

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s hard kill

    best_overall_colors = None
    best_overall_distinct = 0

    num_restarts = 0
    while time.time() < deadline - 3.0:
        seed = 1000 + num_restarts
        colors, color_counts, edge_cycle_info = _greedy_seed(seed)
        
        # Initial distinct count
        current_distinct = _count_distinct(colors)
        if current_distinct > best_overall_distinct:
            best_overall_distinct = current_distinct
            best_overall_colors = colors.copy()

        # Local search
        local_deadline = min(time.time() + 8.0, deadline - 2.0)
        rng = random.Random(seed + 7777)
        colors, distinct = _fast_local_search(colors, color_counts, local_deadline, rng)

        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()

        num_restarts += 1

    if best_overall_colors is None:
        # Fallback: all zeros is always valid
        best_overall_colors = np.zeros(NUM_EDGES, dtype=np.int32)

    # Final validation
    if not _validate(best_overall_colors):
        return [0] * NUM_EDGES

    return best_overall_colors.tolist()
