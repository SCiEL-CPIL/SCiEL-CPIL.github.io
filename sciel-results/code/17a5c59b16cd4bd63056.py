import time
import numpy as np
import random
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
    
    # Precompute for each edge: list of (cycle_idx, position_in_cycle)
    edge_cycle_info = []
    for e in range(NEDGES):
        info = []
        for ci in edge_to_cycles[e]:
            cycle = cycles_np[ci]
            pos = int(np.where(cycle == e)[0][0])
            info.append((ci, pos))
        edge_cycle_info.append(info)
    
    return idx_map, cycles_np, edge_to_cycles, edge_cycle_info

def _is_valid_vectorized(colors_np):
    """Check if no cycle has 4 distinct colors."""
    cycle_colors = colors_np[CYCLES]
    sorted_colors = np.sort(cycle_colors, axis=1)
    adjacent_equal = (sorted_colors[:, 0] == sorted_colors[:, 1]) | \
                     (sorted_colors[:, 1] == sorted_colors[:, 2]) | \
                     (sorted_colors[:, 2] == sorted_colors[:, 3])
    return bool(np.all(adjacent_equal))

def _is_valid_fast(colors_np, edge_cycle_info):
    """Faster validity check using incremental constraint checking."""
    # Check all cycles
    for ci in range(NUM_CYCLES):
        cycle = CYCLES[ci]
        c0, c1, c2, c3 = (colors_np[cycle[0]], colors_np[cycle[1]], 
                          colors_np[cycle[2]], colors_np[cycle[3]])
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def _count_distinct(colors_np):
    return len(np.unique(colors_np))

def _get_allowed_colors(colors_np, e, edge_cycle_info):
    """
    Get set of colors that edge e can take without creating a rainbow cycle.
    Returns None if all colors are allowed, otherwise a set of allowed colors.
    """
    allowed = None
    for ci, pos in edge_cycle_info[e]:
        cycle = CYCLES[ci]
        other_edges = [cycle[k] for k in range(4) if k != pos]
        other_colors = set(colors_np[other_edges])
        
        if len(other_colors) == 3:
            # Must pick from other_colors
            if allowed is None:
                allowed = set(other_colors)
            else:
                allowed &= other_colors
    
    return allowed  # None means all colors allowed

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

def _initial_algebraic_coloring():
    """Generate an initial coloring using algebraic structure."""
    colors = np.zeros(NEDGES, dtype=np.int32)
    for i in range(N):
        for j in range(i + 1, N):
            idx = IDX_MAP[(i, j)]
            # Use XOR-based coloring for some structure
            colors[idx] = (i * 7 + j * 11) % 16
    # Verify and fix if needed
    if not _is_valid_fast(colors, EDGE_CYCLE_INFO):
        return _initial_valid_coloring()
    return colors

def _diversify_initial(colors, rng, num_perturb=5):
    """Apply random valid perturbations to diversify initial coloring."""
    colors = colors.copy()
    for _ in range(num_perturb):
        e = rng.randint(0, NEDGES - 1)
        new_color = rng.randint(0, MAX_COLOR - 1)
        old_color = colors[e]
        colors[e] = new_color
        if not _is_valid_fast(colors, EDGE_CYCLE_INFO):
            colors[e] = old_color
    return colors

def _local_search(colors, deadline, rng, edge_cycle_info, max_iter=None):
    """
    Perform local search maximizing distinct colors.
    Uses forbidden-set pruning and color-frequency heuristic.
    Returns the best coloring found during this search.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    start_time = time.time()
    iter_count = 0
    stagnation_count = 0
    max_stagnation = 50
    
    while time.time() < deadline:
        if max_iter is not None and iter_count >= max_iter:
            break
        iter_count += 1
        
        improved = False
        # Shuffle edge order
        edge_order = list(range(NEDGES))
        rng.shuffle(edge_order)
        
        for e in edge_order:
            if time.time() >= deadline:
                break
            
            old_color = colors[e]
            
            # Get allowed colors
            allowed = _get_allowed_colors(colors, e, edge_cycle_info)
            
            if allowed is not None:
                # Must pick from allowed colors
                candidates = [c for c in allowed if c != old_color]
                if not candidates:
                    continue
            else:
                # All colors allowed - use frequency heuristic
                color_counts = np.bincount(colors, minlength=MAX_COLOR)
                unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
                if unused:
                    candidates = rng.sample(unused, min(15, len(unused)))
                else:
                    # Pick from low-frequency colors
                    freqs = [(color_counts[c], c) for c in range(MAX_COLOR) if color_counts[c] > 0]
                    freqs.sort()
                    candidates = [c for _, c in freqs[:15]]
                    candidates = [c for c in candidates if c != old_color]
                    if not candidates:
                        continue
            
            # Pick a random candidate
            new_color = rng.choice(candidates)
            colors[e] = new_color
            
            # Check validity
            if _is_valid_fast(colors, edge_cycle_info):
                current_distinct = _count_distinct(colors)
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()
                    improved = True
                    stagnation_count = 0
            else:
                # Revert
                colors[e] = old_color
        
        if not improved:
            stagnation_count += 1
            # Try random perturbation to escape local minimum
            if stagnation_count >= max_stagnation:
                num_perturb = rng.randint(3, 8)
                for _ in range(num_perturb):
                    e = rng.randint(0, NEDGES - 1)
                    old_color = colors[e]
                    new_color = rng.randint(0, MAX_COLOR - 1)
                    colors[e] = new_color
                    if not _is_valid_fast(colors, edge_cycle_info):
                        colors[e] = old_color
                stagnation_count = 0
            else:
                # Gentle perturbation
                e = rng.randint(0, NEDGES - 1)
                old_color = colors[e]
                new_color = rng.randint(0, MAX_COLOR - 1)
                colors[e] = new_color
                if not _is_valid_fast(colors, edge_cycle_info):
                    colors[e] = old_color
    
    return best_colors, best_distinct

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    best_overall_colors = None
    best_overall_distinct = 0
    
    rng = random.Random(123)
    
    # Multi-start local search with diverse initializations
    num_restarts = 0
    while time.time() < deadline - 3.0:
        # Choose initial coloring strategy
        init_type = rng.randint(0, 2)
        if init_type == 0:
            colors = _initial_valid_coloring()
        elif init_type == 1:
            colors = _initial_algebraic_coloring()
        else:
            colors = _initial_valid_coloring()
        
        # Diversify
        colors = _diversify_initial(colors, rng, num_perturb=rng.randint(3, 12))
        
        # Run local search with adaptive time limit
        remaining = deadline - time.time()
        local_deadline = min(time.time() + min(10.0, remaining * 0.6), deadline - 2.0)
        colors, distinct = _local_search(colors, local_deadline, rng, EDGE_CYCLE_INFO)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
        
        num_restarts += 1
        
        # Early termination if we've hit a high score
        if best_overall_distinct >= 40:
            break
    
    # Final polish pass if time remains
    if best_overall_colors is not None and time.time() < deadline - 1.0:
        polish_deadline = deadline - 0.5
        best_overall_colors, best_overall_distinct = _local_search(
            best_overall_colors, polish_deadline, rng, EDGE_CYCLE_INFO
        )
    
    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
    
    return best_overall_colors.tolist()

# Precompute
IDX_MAP, CYCLES, EDGE_TO_CYCLES, EDGE_CYCLE_INFO = _get_edge_indices()
NUM_CYCLES = len(CYCLES)
