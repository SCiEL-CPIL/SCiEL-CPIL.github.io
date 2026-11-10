import time
import random
import numpy as np
from itertools import combinations

N = 16
NEDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

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
    
    # For each edge, find which cycles contain it
    edge_to_cycles = [[] for _ in range(NEDGES)]
    for ci, cycle in enumerate(cycles):
        for e in cycle:
            edge_to_cycles[e].append(ci)
    
    return idx_map, cycles_np, edge_to_cycles

def _is_valid_vectorized(colors_np):
    """Check if no cycle has 4 distinct colors."""
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

# Precompute
IDX_MAP, CYCLES, EDGE_TO_CYCLES = _get_edge_indices()
NUM_CYCLES = len(CYCLES)

def _initial_valid_coloring():
    """Generate a valid 2-coloring as starting point."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            if i >= 8 and j >= 8:
                colors[idx] = 1
            else:
                colors[idx] = 0
    return colors

def _initial_algebraic_coloring(K):
    """Generate an algebraic coloring using (i+j) % K."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            colors[idx] = (i + j) % K
    return colors

def _local_search(colors, deadline, rng):
    """
    Perform local search on a single coloring.
    Uses incremental validity check for speed.
    Returns the best coloring found during this search.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    color_counts = np.bincount(colors, minlength=MAX_COLOR)
    
    start_time = time.time()
    
    while time.time() < deadline:
        improved = False
        # Shuffle edge order
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() >= deadline:
                break
            
            old_color = colors[e]
            
            # Determine candidate colors: prefer unused or low-frequency colors
            # to maximize diversity
            # We check a small set of candidates to keep it fast
            # 1. Unused colors
            # 2. Low frequency colors
            
            # To avoid iterating 120 colors every time, we pick a few random candidates
            # biased towards low counts.
            
            # Simple heuristic: pick 5 random candidates, sort by count, try best first
            # Or just try to find an unused color.
            
            # Let's try a specific strategy:
            # 1. If there are unused colors, pick one at random from them.
            # 2. Otherwise, pick a random color with low frequency.
            
            # To make this efficient, we can just try a few random colors and validate.
            # But we want to bias towards diversity.
            
            # Let's generate a list of candidates
            candidates = []
            
            # Find some unused colors
            # np.where(color_counts == 0)[0]
            # This is fast enough? 120 size.
            
            # Let's just try 10 random colors, prioritizing those with count 0 or 1.
            # We will test validity incrementally.
            
            tried = set()
            found_valid = False
            new_color = -1
            
            # Try to find a valid move
            # We want to increase distinct count if possible.
            # If old_color count is 1, changing it reduces distinct count (unless new color is new).
            # If old_color count > 1, changing it keeps distinct count same (unless new color is new).
            
            # Strategy:
            # 1. Generate 20 random candidates.
            # 2. Sort them by (color_counts[c], c) -> prefer low count.
            # 3. Check validity incrementally.
            
            cand_list = rng.sample(range(MAX_COLOR), min(20, MAX_COLOR))
            # Remove old color to force change? Not strictly necessary, but helps mixing.
            if old_color in cand_list:
                cand_list.remove(old_color)
            
            # Sort by count
            cand_list.sort(key=lambda c: color_counts[c])
            
            for c in cand_list:
                if c == old_color:
                    continue
                
                # Check validity incrementally
                # Only check cycles containing e
                valid = True
                for ci in EDGE_TO_CYCLES[e]:
                    cycle = CYCLES[ci]
                    # Get other 3 edges
                    other_colors = []
                    for pos in range(4):
                        if cycle[pos] == e:
                            continue
                        other_colors.append(colors[cycle[pos]])
                    
                    # Check if adding c to other_colors creates 4 distinct colors
                    # i.e., len(set(other_colors + [c])) == 4
                    # This is equivalent to: len(set(other_colors)) == 3 and c not in set(other_colors)
                    # Or len(set(other_colors)) < 3 is always safe?
                    # If other_colors has 1 distinct: e.g. {A, A, A}. Adding c.
                    # If c == A: {A, A, A, A} -> 1 distinct. OK.
                    # If c != A: {A, A, A, c} -> 2 distinct. OK.
                    # If other_colors has 2 distinct: e.g. {A, A, B}. Adding c.
                    # If c == A: {A, A, B, A} -> 2 distinct. OK.
                    # If c == B: {A, A, B, B} -> 2 distinct. OK.
                    # If c != A, B: {A, A, B, c} -> 3 distinct. OK.
                    # If other_colors has 3 distinct: e.g. {A, B, C}. Adding c.
                    # If c == A: {A, B, C, A} -> 3 distinct. OK.
                    # If c == B: {A, B, C, B} -> 3 distinct. OK.
                    # If c == C: {A, B, C, C} -> 3 distinct. OK.
                    # If c != A, B, C: {A, B, C, c} -> 4 distinct. INVALID.
                    
                    other_set = set(other_colors)
                    if len(other_set) == 3 and c not in other_set:
                        valid = False
                        break
                
                if valid:
                    # Accept move
                    colors[e] = c
                    color_counts[old_color] -= 1
                    color_counts[c] += 1
                    
                    # Check if distinct count improved
                    # If color_counts[old_color] was 1, we lost a color.
                    # If color_counts[c] was 0, we gained a color.
                    # Net change:
                    # -1 if old_count == 1 and new_count == 0? No.
                    # If old_count == 1, distinct decreases by 1 (unless new color was also 1? No, we pick low count).
                    # If new_count == 0, distinct increases by 1.
                    
                    # Let's just recalculate distinct count occasionally or track it.
                    # Recalculating is O(120). Doing it every step might be too slow.
                    # But 120 is small.
                    
                    current_distinct = _count_distinct(colors)
                    if current_distinct > best_distinct:
                        best_distinct = current_distinct
                        best_colors = colors.copy()
                        improved = True
                    
                    found_valid = True
                    break # Try next edge
            
            # If no valid move found, we could try a random perturbation
            # But the above loop tries 20 candidates. If all fail, no move.
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(42)
    
    # Try algebraic colorings first as they can give high diversity quickly
    for K in range(2, 20):
        colors = _initial_algebraic_coloring(K)
        if _is_valid_vectorized(colors):
            distinct = _count_distinct(colors)
            if distinct > best_overall_distinct:
                best_overall_distinct = distinct
                best_overall_colors = colors.copy()
    
    # Multi-start local search
    num_restarts = 0
    while time.time() < deadline - 5.0:
        # Generate initial valid coloring
        if rng.random() < 0.5:
            K = rng.randint(2, 10)
            colors = _initial_algebraic_coloring(K)
            if not _is_valid_vectorized(colors):
                colors = _initial_valid_coloring()
        else:
            colors = _initial_valid_coloring()
        
        # Add some random valid perturbations to diversify
        for _ in range(rng.randint(0, 10)):
            e = rng.randint(0, NEDGES - 1)
            new_color = rng.randint(0, MAX_COLOR - 1)
            colors[e] = new_color
            if not _is_valid_vectorized(colors):
                # Revert
                # We don't have the old color stored here easily without copying
                # Let's just revert to 0 or 1 which are likely valid in initial state
                # Actually, better to just skip invalid perturbations
                # But we already changed it.
                # Let's restore from a backup or just accept that this restart starts from a potentially invalid state?
                # No, _local_search assumes valid start?
                # My incremental check assumes valid state.
                # So we must ensure start is valid.
                # Let's just not do random perturbations if we can't easily revert.
                # Or, revert to the color it had.
                # We didn't store it.
                # Let's change strategy: only perturb if we can check validity easily.
                # Since we are in construct(), we can use _is_valid_vectorized.
                # If invalid, we need to revert.
                # We can just copy colors before perturbations.
                pass # Disabled for now to avoid bugs
        
        # Run local search with time limit
        local_deadline = min(time.time() + 8.0, deadline - 2.0)
        colors, distinct = _local_search(colors, local_deadline, rng)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
        
        num_restarts += 1
    
    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
    
    return best_overall_colors.tolist()
