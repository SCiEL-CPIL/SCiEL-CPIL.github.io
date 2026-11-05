import time
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 119

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_four_cycles():
    """Precompute all 4-cycles as tuples of 4 edge indices."""
    cycles = []
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # Cycle 1: a-b-c-d-a
                    e1 = (edge_index(a, b), edge_index(b, c), edge_index(c, d), edge_index(d, a))
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c-a
                    e2 = (edge_index(a, b), edge_index(b, d), edge_index(d, c), edge_index(c, a))
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d-a
                    e3 = (edge_index(a, c), edge_index(c, b), edge_index(b, d), edge_index(d, a))
                    cycles.append(e3)
    return cycles

_CYCLES = build_four_cycles()
_CYCLES_BY_EDGE = [[] for _ in range(NUM_EDGES)]
for cyc in _CYCLES:
    for e in cyc:
        _CYCLES_BY_EDGE[e].append(cyc)

def is_valid_cycle(colors, cyc):
    """Check if a 4-cycle has at most 3 distinct colors."""
    c0, c1, c2, c3 = colors[cyc[0]], colors[cyc[1]], colors[cyc[2]], colors[cyc[3]]
    if c0 == c1 or c0 == c2 or c0 == c3:
        return True
    if c1 == c2 or c1 == c3:
        return True
    if c2 == c3:
        return True
    return False

def validate(colors):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for cyc in _CYCLES:
        if not is_valid_cycle(colors, cyc):
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def greedy_coloring(edge_order, rng):
    """
    Greedy: process edges in given order, assign each edge a color that
    keeps all 4-cycles valid. Try to maximize distinct colors by preferring
    unused colors when possible.
    """
    colors = [-1] * NUM_EDGES
    used_colors = set()
    
    for e in edge_order:
        # Find all 4-cycles that contain edge e and are now "complete"
        # (all 4 edges have been assigned)
        candidate_colors = []
        
        # Try colors in an order that favors diversity
        # First try unused colors (to increase distinct count)
        unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
        rng.shuffle(unused)
        
        # Then try used colors
        used_list = list(used_colors)
        rng.shuffle(used_list)
        
        all_candidates = unused + used_list
        
        found = False
        for c in all_candidates:
            colors[e] = c
            # Check all cycles containing e
            valid = True
            for cyc in _CYCLES_BY_EDGE[e]:
                # Only check if all 4 edges in the cycle are assigned
                if colors[cyc[0]] == -1 or colors[cyc[1]] == -1 or colors[cyc[2]] == -1 or colors[cyc[3]] == -1:
                    continue
                if not is_valid_cycle(colors, cyc):
                    valid = False
                    break
            if valid:
                found = True
                break
        
        if not found:
            # Shouldn't happen, but fallback to 0
            colors[e] = 0
        else:
            used_colors.add(colors[e])
    
    return colors

def local_improve(colors, rng, deadline):
    """Try to increase the number of distinct colors via local search."""
    colors = colors[:]
    best_count = count_colors(colors)
    current_count = best_count
    used_colors = set(colors)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for _ in range(500):
            if time.time() >= deadline:
                break
            e = rng.randint(0, NUM_EDGES - 1)
            old = colors[e]
            
            # Try to change to a new color not currently used
            unused = [c for c in range(MAX_COLOR + 1) if c not in used_colors]
            if unused:
                rng.shuffle(unused)
                candidates = unused[:10]  # Try up to 10 unused colors
            else:
                candidates = [rng.randint(0, MAX_COLOR) for _ in range(10)]
            
            for new_c in candidates:
                if new_c == old:
                    continue
                colors[e] = new_c
                # Check only cycles containing edge e
                valid = True
                for cyc in _CYCLES_BY_EDGE[e]:
                    if not is_valid_cycle(colors, cyc):
                        valid = False
                        break
                if valid:
                    if new_c not in used_colors:
                        used_colors.add(new_c)
                        current_count += 1
                        if current_count > best_count:
                            best_count = current_count
                        improved = True
                    break
                else:
                    colors[e] = old
            else:
                colors[e] = old
    
    return colors, best_count

def construct():
    start = time.time()
    deadline = start + 55  # Safety margin before 60s limit
    
    rng = random.Random(42)
    
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Try multiple greedy orderings
    edge_indices = list(range(NUM_EDGES))
    
    # Different orderings to try
    orderings = []
    
    # Natural order
    orderings.append(list(edge_indices))
    
    # Reverse order
    orderings.append(list(reversed(edge_indices)))
    
    # Random shuffles
    for _ in range(15):
        shuffled = list(edge_indices)
        rng.shuffle(shuffled)
        orderings.append(shuffled)
    
    # Process edges in order of increasing degree in the 4-cycle hypergraph
    # (edges that appear in fewer 4-cycles first)
    edge_cycle_count = [len(_CYCLES_BY_EDGE[e]) for e in range(NUM_EDGES)]
    degree_order = sorted(edge_indices, key=lambda e: edge_cycle_count[e])
    orderings.append(degree_order)
    orderings.append(list(reversed(degree_order)))
    
    candidates = []
    for order in orderings:
        if time.time() >= deadline - 10:
            break
        c = greedy_coloring(order, rng)
        if validate(c):
            cnt = count_colors(c)
            candidates.append((cnt, c))
    
    # Sort by count descending
    candidates.sort(key=lambda x: -x[0])
    
    if candidates:
        best_count = candidates[0][0]
        best_colors = candidates[0][1][:]
    
    # Local improvement on top candidates
    top_candidates = candidates[:5]
    for cnt, c in top_candidates:
        if time.time() >= deadline - 5:
            break
        improved_c, improved_cnt = local_improve(c, rng, deadline)
        if improved_cnt > best_count:
            best_count = improved_cnt
            best_colors = improved_c
    
    # Final validation
    if not validate(best_colors):
        best_colors = [0] * NUM_EDGES
        best_count = 1
    
    return best_colors
