import time
import random
import numpy as np
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

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
                    cycles.append((_edge_index(a, b), _edge_index(b, c), _edge_index(c, d), _edge_index(a, d)))
                    cycles.append((_edge_index(a, b), _edge_index(b, d), _edge_index(c, d), _edge_index(a, c)))
                    cycles.append((_edge_index(a, c), _edge_index(b, c), _edge_index(b, d), _edge_index(a, d)))
    return cycles

CYCLES = _build_cycles()
NUM_CYCLES = len(CYCLES)

# Precompute cycle matrix: each row is 4 edge indices
CYCLE_MATRIX = np.array(CYCLES, dtype=np.int32)  # shape (NUM_CYCLES, 4)

# Build edge-to-cycles mapping
def _build_edge_to_cycles():
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(CYCLES):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

EDGE_TO_CYCLES = _build_edge_to_cycles()

def _count_violations(colors):
    """Vectorized count of 4-cycles with 4 distinct edge colors."""
    if isinstance(colors, list):
        colors = np.array(colors, dtype=np.int32)
    # Get colors for all cycle edges at once
    cycle_colors = colors[CYCLE_MATRIX]  # shape (NUM_CYCLES, 4)
    # A cycle is a violation if all 4 colors are distinct
    # Sort each row and check if all unique
    sorted_colors = np.sort(cycle_colors, axis=1)
    # Check if any adjacent elements are equal (means not all distinct)
    has_dup = np.any(sorted_colors[:, 1:] == sorted_colors[:, :-1], axis=1)
    violations = ~has_dup  # True where all 4 are distinct
    return int(np.sum(violations))

def _is_valid(colors):
    """Check if coloring has no violations."""
    if isinstance(colors, list):
        colors = np.array(colors, dtype=np.int32)
    cycle_colors = colors[CYCLE_MATRIX]
    sorted_colors = np.sort(cycle_colors, axis=1)
    has_dup = np.any(sorted_colors[:, 1:] == sorted_colors[:, :-1], axis=1)
    return bool(np.all(has_dup))

def _check_edge_change_safe(e, new_color, colors):
    """Check if changing edge e to new_color creates any new violations.
    Returns True if safe (no new violations created)."""
    for ci in EDGE_TO_CYCLES[e]:
        cyc = CYCLES[ci]
        e0, e1, e2, e3 = cyc
        oc0, oc1, oc2, oc3 = colors[e0], colors[e1], colors[e2], colors[e3]
        
        nc0, nc1, nc2, nc3 = oc0, oc1, oc2, oc3
        if e0 == e:
            nc0 = new_color
        elif e1 == e:
            nc1 = new_color
        elif e2 == e:
            nc2 = new_color
        elif e3 == e:
            nc3 = new_color
        
        old_all_distinct = len({oc0, oc1, oc2, oc3}) == 4
        new_all_distinct = len({nc0, nc1, nc2, nc3}) == 4
        
        if not old_all_distinct and new_all_distinct:
            return False
    return True

def _rank1_init(seed=42):
    """Generate an initial coloring using rank-1 algebraic structure.
    
    Uses color(i,j) = (u_i * v_j) mod K for i < j, where u and v are 
    carefully chosen vectors. This tends to produce colorings with many 
    distinct colors while having structure that may avoid 4-distinct cycles.
    """
    rng = np.random.RandomState(seed)
    
    # Try different moduli and vector configurations
    best_colors = None
    best_count = 0
    
    for K in [7, 11, 13, 17, 19, 23, 29, 31]:
        # Generate random vectors
        u = rng.randint(1, K, size=N)
        v = rng.randint(1, K, size=N)
        
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_index(i, j)] = (u[i] * v[j]) % K
        
        # Check validity
        if _is_valid(colors):
            count = len(set(colors))
            if count > best_count:
                best_count = count
                best_colors = colors.copy()
    
    if best_colors is not None:
        return best_colors
    
    # Fallback: try with offset
    for K in [7, 11, 13, 17, 19, 23]:
        u = rng.randint(1, K, size=N)
        v = rng.randint(1, K, size=N)
        offset = rng.randint(0, K)
        
        colors = np.zeros(NUM_EDGES, dtype=np.int32)
        for i in range(N):
            for j in range(i + 1, N):
                colors[_edge_index(i, j)] = ((u[i] * v[j]) + offset) % K
        
        if _is_valid(colors):
            count = len(set(colors))
            if count > best_count:
                best_count = count
                best_colors = colors.copy()
    
    if best_colors is not None:
        return best_colors
    
    # Final fallback: all zeros
    return np.zeros(NUM_EDGES, dtype=np.int32)

def _random_valid_init(seed=42):
    """Try to build a valid coloring by greedily assigning colors."""
    rng = np.random.RandomState(seed)
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Try to assign random colors while maintaining validity
    for _ in range(500):  # Try up to 500 assignments
        e = rng.randint(0, NUM_EDGES)
        c = rng.randint(0, MAX_COLOR + 1)
        if c != colors[e] and _check_edge_change_safe(e, c, colors):
            colors[e] = c
    
    return colors

def _local_search(colors, deadline, seed=42):
    """Fast local search to maximize distinct colors while maintaining validity."""
    rng = np.random.RandomState(seed)
    colors = colors.copy()
    
    # Track color counts
    color_counts = np.zeros(MAX_COLOR + 1, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    
    current_distinct = int(np.sum(color_counts > 0))
    best_colors = colors.copy()
    best_distinct = current_distinct
    
    T = 2.0
    cooling = 0.99995
    min_T = 0.05
    
    # Precompute which edges have which colors for quick reference
    while time.time() < deadline:
        if T < min_T:
            break
        
        # Pick a random edge
        e_idx = rng.randint(0, NUM_EDGES)
        old_color = int(colors[e_idx])
        
        # Choose new color: bias toward unused colors to encourage diversity
        if rng.random() < 0.6:
            # Pick from unused colors
            unused_mask = color_counts == 0
            if np.any(unused_mask):
                unused_indices = np.where(unused_mask)[0]
                new_color = int(rng.choice(unused_indices))
            else:
                new_color = int(rng.randint(0, MAX_COLOR + 1))
        else:
            new_color = int(rng.randint(0, MAX_COLOR + 1))
        
        if new_color == old_color:
            continue
        
        # Check if safe
        if not _check_edge_change_safe(e_idx, new_color, colors):
            continue
        
        # Apply change
        colors[e_idx] = new_color
        color_counts[old_color] -= 1
        color_counts[new_color] += 1
        
        new_distinct = current_distinct
        if color_counts[old_color] == 0:
            new_distinct -= 1
        if color_counts[new_color] == 1:
            new_distinct += 1
        
        # Accept/reject
        delta = new_distinct - current_distinct
        if delta >= 0 or rng.random() < np.exp(delta / T):
            current_distinct = new_distinct
            if new_distinct > best_distinct:
                best_distinct = new_distinct
                best_colors = colors.copy()
        else:
            # Revert
            colors[e_idx] = old_color
            color_counts[old_color] += 1
            color_counts[new_color] -= 1
        
        T *= cooling
    
    return best_colors, best_distinct

def _fix_violations(colors):
    """Try to fix any violations by recoloring edges."""
    colors = colors.copy()
    
    for _ in range(1000):
        viol_count = _count_violations(colors)
        if viol_count == 0:
            break
        
        # Find a violating cycle
        cycle_colors = colors[CYCLE_MATRIX]
        sorted_colors = np.sort(cycle_colors, axis=1)
        has_dup = np.any(sorted_colors[:, 1:] == sorted_colors[:, :-1], axis=1)
        violating_indices = np.where(~has_dup)[0]
        
        if len(violating_indices) == 0:
            break
        
        # Pick a random violating cycle
        ci = int(random.choice(violating_indices))
        cyc = CYCLES[ci]
        
        # Try to recolor one of the edges in this cycle
        for e in cyc:
            old_c = int(colors[e])
            # Try colors that match one of the other edges in the cycle
            other_colors = [int(colors[oe]) for oe in cyc if oe != e]
            for nc in other_colors:
                if nc != old_c and _check_edge_change_safe(e, nc, colors):
                    # Check if this reduces violations
                    colors[e] = nc
                    if _count_violations(colors) < viol_count:
                        break
                    else:
                        colors[e] = old_c
            
            # Also try a random color
            nc = random.randint(0, MAX_COLOR)
            if nc != old_c and _check_edge_change_safe(e, nc, colors):
                colors[e] = nc
                if _count_violations(colors) < viol_count:
                    break
                else:
                    colors[e] = old_c
    
    return colors

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    best_colors = None
    best_distinct = 0
    
    # Multiple initialization strategies
    inits = []
    
    # Strategy 1: Rank-1 algebraic
    for seed in range(20):
        c = _rank1_init(seed)
        if _is_valid(c):
            d = len(set(c))
            inits.append((d, c))
    
    # Strategy 2: Random valid init
    for seed in range(20):
        c = _random_valid_init(seed)
        if _is_valid(c):
            d = len(set(c))
            inits.append((d, c))
    
    # Strategy 3: Simple greedy
    colors = np.zeros(NUM_EDGES, dtype=np.int32)
    for new_c in range(1, MAX_COLOR + 1):
        if time.time() > deadline:
            break
        for e in range(NUM_EDGES):
            if _check_edge_change_safe(e, new_c, colors):
                colors[e] = new_c
                break
    if _is_valid(colors):
        d = len(set(colors))
        inits.append((d, colors))
    
    # Sort by distinct count, take top candidates
    inits.sort(key=lambda x: -x[0])
    
    # Local search on best initializations
    for d, c in inits[:10]:  # Top 10 initializations
        if time.time() > deadline:
            break
        
        result_colors, result_distinct = _local_search(c, deadline, seed=random.randint(0, 10000))
        
        if _is_valid(result_colors):
            if result_distinct > best_distinct:
                best_distinct = result_distinct
                best_colors = result_colors
    
    # Final fallback
    if best_colors is None:
        best_colors = np.zeros(NUM_EDGES, dtype=np.int32)
    
    # Verify and return
    if not _is_valid(best_colors):
        best_colors = _fix_violations(best_colors)
        if not _is_valid(best_colors):
            return [0] * NUM_EDGES
    
    return best_colors.tolist()
