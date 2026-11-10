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

def _entropy_penalty(colors_np):
    """Compute negative entropy of color distribution (lower is better for high diversity)."""
    counts = np.bincount(colors_np, minlength=MAX_COLOR)
    nonzero = counts[counts > 0]
    if len(nonzero) == 0:
        return 0.0
    probs = nonzero / nonzero.sum()
    return -np.sum(probs * np.log(probs))

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

def _local_search(colors, deadline, rng, use_entropy=False):
    """
    Perform local search on a single coloring.
    Uses forbidden-set pruning and optional entropy-aware candidate selection.
    Returns the best coloring found during this search.
    """
    best_colors = colors.copy()
    best_distinct = _count_distinct(colors)
    
    # Precompute for each edge: list of (cycle_idx, position_in_cycle)
    edge_cycle_info = []
    for e in range(NEDGES):
        info = []
        for ci in EDGE_TO_CYCLES[e]:
            cycle = CYCLES[ci]
            for pos in range(4):
                if cycle[pos] == e:
                    info.append((ci, pos))
                    break
        edge_cycle_info.append(info)
    
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
            
            # Compute forbidden colors: colors that would create a rainbow cycle
            # For each cycle containing e, if the other 3 edges have 3 distinct colors,
            # then the 4th color must be one of those 3 to avoid rainbow.
            allowed_colors = None  # None means all colors allowed initially
            
            for ci, pos in edge_cycle_info[e]:
                cycle = CYCLES[ci]
                other_edges = [cycle[k] for k in range(4) if k != pos]
                other_colors = [colors[oe] for oe in other_edges]
                other_set = set(other_colors)
                
                if len(other_set) == 3:
                    # This edge's color MUST be in other_set to avoid rainbow
                    if allowed_colors is None:
                        allowed_colors = set(other_set)
                    else:
                        allowed_colors &= set(other_set)
            
            # Determine candidate colors
            if allowed_colors is not None:
                # Must pick from allowed_colors
                candidates = list(allowed_colors)
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue
            else:
                # All colors are allowed (no constraint from this edge's cycles)
                # Use color-frequency heuristic: prefer under-represented colors
                color_counts = np.bincount(colors, minlength=MAX_COLOR)
                # Pick a random unused color first, then under-represented
                unused = [c for c in range(MAX_COLOR) if color_counts[c] == 0]
                if unused:
                    candidates = unused[:20]
                else:
                    # Pick from low-frequency colors
                    freqs = [(color_counts[c], c) for c in range(MAX_COLOR) if color_counts[c] > 0]
                    freqs.sort()
                    candidates = [c for _, c in freqs[:20]]
                
                # Remove old color
                if old_color in candidates:
                    candidates.remove(old_color)
                if not candidates:
                    continue
            
            # If using entropy-aware selection, weight candidates by how much they improve diversity
            if use_entropy and len(candidates) > 1:
                # Score candidates by potential entropy improvement
                current_entropy = _entropy_penalty(colors)
                scored = []
                for c in candidates:
                    colors[e] = c
                    if _is_valid_vectorized(colors):
                        new_entropy = _entropy_penalty(colors)
                        scored.append((new_entropy - current_entropy, c))
                    colors[e] = old_color
                if scored:
                    scored.sort(reverse=True)
                    # Pick from top candidates with some randomness
                    top_k = min(5, len(scored))
                    _, best_c = scored[rng.randint(0, top_k - 1)]
                    new_color = best_c
                else:
                    new_color = rng.choice(candidates)
            else:
                new_color = rng.choice(candidates)
            
            colors[e] = new_color
            
            # Check validity (vectorized)
            if _is_valid_vectorized(colors):
                current_distinct = _count_distinct(colors)
                if current_distinct > best_distinct:
                    best_distinct = current_distinct
                    best_colors = colors.copy()
                    improved = True
            else:
                # Revert
                colors[e] = old_color
        
        if not improved:
            # Try a random perturbation to escape local minimum
            num_perturb = rng.randint(1, 5)
            for _ in range(num_perturb):
                e = rng.randint(0, NEDGES - 1)
                old_color = colors[e]
                new_color = rng.randint(0, MAX_COLOR - 1)
                colors[e] = new_color
                if not _is_valid_vectorized(colors):
                    colors[e] = old_color
    
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
        # Generate initial valid coloring - mix of algebraic and random perturbations
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
                colors[e] = 0  # Revert to safe color
        
        # Run local search with time limit
        local_deadline = min(time.time() + 8.0, deadline - 2.0)
        use_entropy = (num_restarts % 3 == 0)  # Use entropy-aware selection every 3rd restart
        colors, distinct = _local_search(colors, local_deadline, rng, use_entropy=use_entropy)
        
        if distinct > best_overall_distinct:
            best_overall_distinct = distinct
            best_overall_colors = colors.copy()
        
        num_restarts += 1
    
    if best_overall_colors is None:
        best_overall_colors = _initial_valid_coloring()
    
    return best_overall_colors.tolist()
