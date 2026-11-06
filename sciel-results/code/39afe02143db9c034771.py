"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

# Precompute all K4s (4-vertex subsets) and their edges
_K4S = list(combinations(range(16), 4))
_K4_EDGES = []
for quad in _K4S:
    edges = []
    for i in range(4):
        for j in range(i+1, 4):
            u = quad[i]
            v = quad[j]
            idx = 15*u - u*(u-1)//2 + (v - u - 1)
            edges.append(idx)
    _K4_EDGES.append(edges)

# Build adjacency: for each edge index, list of k4_idx that contain it
_EDGE_TO_K4S = [[] for _ in range(120)]
for k4_idx, edges in enumerate(_K4_EDGES):
    for e_idx in edges:
        _EDGE_TO_K4S[e_idx].append(k4_idx)


def _count_colours_in_k4(coloring, k4_idx):
    edges = _K4_EDGES[k4_idx]
    cols = set()
    for e_idx in edges:
        cols.add(coloring[e_idx])
    return len(cols)


def _check_valid(coloring):
    """Return True if every K4 has at least 5 distinct colours."""
    for k4_idx in range(len(_K4S)):
        if _count_colours_in_k4(coloring, k4_idx) < 5:
            return False
    return True


def _count_violations(coloring):
    """Return number of K4s with fewer than 5 colours."""
    count = 0
    for k4_idx in range(len(_K4S)):
        if _count_colours_in_k4(coloring, k4_idx) < 5:
            count += 1
    return count


def _num_colours(coloring):
    return len(set(coloring))


def _greedy_construct():
    """
    Deterministic greedy: assign smallest available color to each edge in lex order.
    For each edge, reject a color if it causes any fully-colored K4 to have < 5 colors.
    Also use a look-ahead: for partially colored K4s, ensure the constraint is still satisfiable.
    """
    coloring = [-1] * 120  # -1 means uncolored
    max_color = 0  # track the highest color used
    
    for e_idx in range(120):
        # Try colors 0, 1, 2, ... up to max_color+1
        placed = False
        for c in range(max_color + 2):
            coloring[e_idx] = c
            valid = True
            
            # Check all K4s containing this edge
            for k4_idx in _EDGE_TO_K4S[e_idx]:
                edges = _K4_EDGES[k4_idx]
                # Check if all edges in this K4 are colored
                all_colored = all(coloring[ei] >= 0 for ei in edges)
                
                if all_colored:
                    # All 6 edges colored: need >= 5 distinct colors
                    cols = set(coloring[ei] for ei in edges)
                    if len(cols) < 5:
                        valid = False
                        break
                else:
                    # Partially colored: check if constraint is still satisfiable
                    # Count distinct colors on colored edges
                    colored_edges = [ei for ei in edges if coloring[ei] >= 0]
                    uncolored_count = len(edges) - len(colored_edges)
                    distinct_so_far = len(set(coloring[ei] for ei in colored_edges))
                    # Each uncolored edge can add at most 1 new color
                    # Need: distinct_so_far + uncolored_count >= 5
                    if distinct_so_far + uncolored_count < 5:
                        valid = False
                        break
            
            if valid:
                placed = True
                if c > max_color:
                    max_color = c
                break
        
        if not placed:
            # Should not happen if we try enough colors, but fallback
            coloring[e_idx] = max_color + 1
            max_color += 1
    
    # Compact colors: remap to 0..k-1
    color_map = {}
    next_c = 0
    for c in coloring:
        if c not in color_map:
            color_map[c] = next_c
            next_c += 1
    coloring = [color_map[c] for c in coloring]
    
    if _check_valid(coloring):
        return coloring, _num_colours(coloring)
    return None, 17


def _hill_climb(coloring, max_iters=50000, time_limit=100.0):
    """Hill-climbing: randomly change one edge colour to reduce violations and colour count."""
    start_time = time.time()
    current = coloring[:]
    current_violations = _count_violations(current)
    current_nc = _num_colours(current)
    best = current[:]
    best_violations = current_violations
    best_nc = current_nc
    best_score = best_violations * 1000 + best_nc
    
    rng = random.Random(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        e_idx = rng.randint(0, 119)
        old_c = current[e_idx]
        new_c = rng.randint(0, 119)
        if new_c == old_c:
            continue
        
        current[e_idx] = new_c
        
        delta_viol = 0
        for k4_idx in _EDGE_TO_K4S[e_idx]:
            current[e_idx] = old_c
            before = _count_colours_in_k4(current, k4_idx)
            current[e_idx] = new_c
            after = _count_colours_in_k4(current, k4_idx)
            before_v = 1 if before < 5 else 0
            after_v = 1 if after < 5 else 0
            delta_viol += after_v - before_v
        
        new_violations = current_violations + delta_viol
        new_nc = _num_colours(current)
        new_score = new_violations * 1000 + new_nc
        
        if new_score <= best_score:
            best_score = new_score
            best = current[:]
            best_violations = new_violations
            best_nc = new_nc
            current_violations = new_violations
            current_nc = new_nc
        else:
            current[e_idx] = old_c
    
    return best, best_nc


def _compact_and_hill_climb(coloring, time_limit=60.0):
    """Compact colors then hill climb to reduce color count."""
    # Compact: remap colors to 0..k-1 preserving structure
    color_map = {}
    next_c = 0
    for c in coloring:
        if c not in color_map:
            color_map[c] = next_c
            next_c += 1
    compacted = [color_map[c] for c in coloring]
    
    # Hill climb with focus on reducing color count
    start_time = time.time()
    current = compacted[:]
    current_violations = _count_violations(current)
    current_nc = _num_colours(current)
    best = current[:]
    best_violations = current_violations
    best_nc = current_nc
    best_score = best_violations * 1000 + best_nc
    
    rng = random.Random(777)
    
    while time.time() - start_time < time_limit:
        e_idx = rng.randint(0, 119)
        old_c = current[e_idx]
        
        # Try to assign a color that's already in use (to potentially reduce palette)
        used_colors = set(current)
        candidates = list(used_colors)
        if not candidates:
            candidates = [0]
        new_c = rng.choice(candidates)
        if new_c == old_c:
            # Also try a random new color
            new_c = rng.randint(0, current_nc)
            if new_c == old_c:
                continue
        
        current[e_idx] = new_c
        
        delta_viol = 0
        for k4_idx in _EDGE_TO_K4S[e_idx]:
            current[e_idx] = old_c
            before = _count_colours_in_k4(current, k4_idx)
            current[e_idx] = new_c
            after = _count_colours_in_k4(current, k4_idx)
            before_v = 1 if before < 5 else 0
            after_v = 1 if after < 5 else 0
            delta_viol += after_v - before_v
        
        new_violations = current_violations + delta_viol
        new_nc = _num_colours(current)
        new_score = new_violations * 1000 + new_nc
        
        if new_score <= best_score:
            best_score = new_score
            best = current[:]
            best_violations = new_violations
            best_nc = new_nc
            current_violations = new_violations
            current_nc = new_nc
        else:
            current[e_idx] = old_c
    
    return best, best_nc


def construct():
    deadline = time.time() + 145
    
    # Try greedy construction first
    best = None
    best_nc = 17
    
    c, nc = _greedy_construct()
    if c is not None:
        best = c
        best_nc = nc
    
    # If greedy didn't produce valid, use seed
    if best is None:
        best = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
        best_nc = _num_colours(best)
    
    # Hill climb to reduce colors
    if time.time() < deadline - 10:
        remaining = deadline - time.time() - 10
        c, nc = _hill_climb(best, time_limit=min(remaining, 50.0))
        if nc < best_nc:
            best = c
            best_nc = nc
    
    # Compact and hill climb
    if time.time() < deadline - 5:
        remaining = deadline - time.time() - 5
        c, nc = _compact_and_hill_climb(best, time_limit=min(remaining, 40.0))
        if nc < best_nc:
            best = c
            best_nc = nc
    
    # Final polish
    if time.time() < deadline - 3:
        remaining = deadline - time.time() - 3
        c, nc = _hill_climb(best, time_limit=min(remaining, 20.0))
        if nc < best_nc:
            best = c
            best_nc = nc
    
    # Ensure validity
    if not _check_valid(best):
        best = [(i + j) % 17 for i in range(16) for j in range(i + 1, 16)]
    
    # Ensure all colours are in 0..119
    best = [c % 120 for c in best]
    
    return best
# EVOLVE-BLOCK-END
