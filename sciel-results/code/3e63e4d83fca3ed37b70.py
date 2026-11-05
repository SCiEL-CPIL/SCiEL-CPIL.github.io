"""Program for the edge-coloring task: maximize distinct colors in K16 with no rainbow 4-cycle."""
import time
import random
import itertools

N = 16
NUM_COLORS = 120
NUM_EDGES = N * (N - 1) // 2  # 120

def _edge_index(i, j):
    """Get the index for edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + j - i - 1

def _precompute_cycles():
    """Precompute all 4-cycles as lists of 4 edge indices."""
    cycles = []
    for a, b, c, d in itertools.combinations(range(N), 4):
        # Three 4-cycles on {a,b,c,d}
        # Cycle 1: a-b, b-c, c-d, d-a
        cycles.append([_edge_index(a,b), _edge_index(b,c), _edge_index(c,d), _edge_index(d,a)])
        # Cycle 2: a-b, b-d, d-c, c-a
        cycles.append([_edge_index(a,b), _edge_index(b,d), _edge_index(c,d), _edge_index(c,a)])
        # Cycle 3: a-c, c-b, b-d, d-a
        cycles.append([_edge_index(a,c), _edge_index(b,c), _edge_index(b,d), _edge_index(d,a)])
    return cycles

def _precompute_edge_cycles():
    """For each edge, precompute the list of cycles it participates in."""
    all_cycles = _precompute_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(all_cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return all_cycles, edge_to_cycles

def _check_validity(colors, all_cycles):
    """Check that no 4-cycle has four distinct colors."""
    for cyc in all_cycles:
        if len(set(colors[e] for e in cyc)) == 4:
            return False
    return True

def _is_valid_change(colors, edge_idx, new_color, all_cycles, edge_to_cycles):
    """Check if changing edge_idx to new_color keeps all cycles valid."""
    for ci in edge_to_cycles[edge_idx]:
        cyc = all_cycles[ci]
        # Get the other 3 edges in this cycle
        other_edges = [e for e in cyc if e != edge_idx]
        other_colors = set(colors[e] for e in other_edges)
        # The cycle would have colors: {new_color} | other_colors
        if len(other_colors) == 3 and new_color not in other_colors:
            return False
    return True

def _count_colors(colors):
    return len(set(colors))

def _greedy_coloring(all_cycles, edge_to_cycles, edge_order, color_order):
    """
    Greedy edge-coloring: process edges in edge_order, assign smallest color from color_order
    that doesn't create a rainbow 4-cycle.
    """
    colors = [None] * NUM_EDGES
    
    for e in edge_order:
        # Find the first valid color from color_order
        assigned = False
        for c in color_order:
            # Check if assigning color c to edge e creates a rainbow 4-cycle
            valid = True
            for ci in edge_to_cycles[e]:
                cyc = all_cycles[ci]
                other_edges = [x for x in cyc if x != e]
                # Only consider edges that have been colored so far
                other_colored = [x for x in other_edges if colors[x] is not None]
                if len(other_colored) < 3:
                    continue
                other_colors = set(colors[x] for x in other_colored)
                if len(other_colors) == 3 and c not in other_colors:
                    valid = False
                    break
            if valid:
                colors[e] = c
                assigned = True
                break
        if not assigned:
            # Should not happen with enough colors, but fallback
            colors[e] = color_order[0]
    
    return colors

def _local_improve(colors, all_cycles, edge_to_cycles, time_limit):
    """Try to improve the coloring by swapping colors or trying new colors."""
    start_time = time.time()
    best_count = _count_colors(colors)
    best_colors = colors[:]
    
    # Try to introduce new colors
    used = set(colors)
    for new_c in range(NUM_COLORS):
        if new_c in used:
            continue
        if time.time() - start_time > time_limit:
            break
        
        # Try to add new_c to one edge at a time
        for e in range(NUM_EDGES):
            if time.time() - start_time > time_limit:
                break
            old = colors[e]
            if _is_valid_change(colors, e, new_c, all_cycles, edge_to_cycles):
                colors[e] = new_c
                used.add(new_c)
                cur_count = _count_colors(colors)
                if cur_count > best_count:
                    best_count = cur_count
                    best_colors = colors[:]
                break
    
    return best_colors, best_count

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline for safety (90s limit)
    
    all_cycles, edge_to_cycles = _precompute_edge_cycles()
    
    # Strategy 1: Greedy with lexicographic edge order
    edge_order_lex = list(range(NUM_EDGES))
    color_order_asc = list(range(NUM_COLORS))
    
    colors1 = _greedy_coloring(all_cycles, edge_to_cycles, edge_order_lex, color_order_asc)
    count1 = _count_colors(colors1)
    
    # Strategy 2: Greedy with reversed edge order
    edge_order_rev = list(range(NUM_EDGES - 1, -1, -1))
    colors2 = _greedy_coloring(all_cycles, edge_to_cycles, edge_order_rev, color_order_asc)
    count2 = _count_colors(colors2)
    
    # Strategy 3: Greedy with random edge order
    random.seed(42)
    edge_order_rand = list(range(NUM_EDGES))
    random.shuffle(edge_order_rand)
    colors3 = _greedy_coloring(all_cycles, edge_to_cycles, edge_order_rand, color_order_asc)
    count3 = _count_colors(colors3)
    
    # Pick the best initial solution
    best_count = max(count1, count2, count3)
    if best_count == count1:
        best_colors = colors1[:]
    elif best_count == count2:
        best_colors = colors2[:]
    else:
        best_colors = colors3[:]
    
    # Local improvement phase
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        best_colors, best_count = _local_improve(best_colors, all_cycles, edge_to_cycles, remaining_time - 5)
    
    # Verify validity
    if not _check_validity(best_colors, all_cycles):
        # Fallback to a simple valid coloring
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
