"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time
from collections import Counter

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# EVOLVE-BLOCK-START

def _edge_index(i, j):
    """Map (i,j) with 0<=i<j<N to linear index in lexicographic order."""
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def _build_edge_list():
    """Return list of (i,j) in lexicographic order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

# Precompute 4-cycles for each edge
# A 4-cycle is a set of 4 edges.
# We need for each edge e, the list of 4-cycles containing e,
# and for each such cycle, the other 3 edges.

def _precompute_cycles():
    """
    Returns a list `cycles_for_edge` of length NUM_EDGES.
    cycles_for_edge[e] is a list of tuples (e1, e2, e3) where 
    {e, e1, e2, e3} is a 4-cycle.
    """
    edges = _build_edge_list()
    edge_to_idx = {e: i for i, e in enumerate(edges)}
    
    # Map from vertex to list of edges incident to it?
    # Or just iterate over all 4-vertex subsets.
    
    cycles_for_edge = [[] for _ in range(NUM_EDGES)]
    
    # Iterate over all 4-vertex subsets
    for a in range(N):
        for b in range(a + 1, N):
            for c in range(b + 1, N):
                for d in range(c + 1, N):
                    # 3 cycles on {a,b,c,d}
                    # Cycle 1: (a,b), (b,c), (c,d), (d,a)
                    # Cycle 2: (a,b), (b,d), (d,c), (c,a)
                    # Cycle 3: (a,c), (c,b), (b,d), (d,a)
                    
                    # We need to get the edge indices for these pairs.
                    # Helper to get index
                    def get_idx(u, v):
                        if u > v: u, v = v, u
                        return edge_to_idx[(u, v)]
                    
                    # Cycle 1
                    e1 = get_idx(a, b)
                    e2 = get_idx(b, c)
                    e3 = get_idx(c, d)
                    e4 = get_idx(d, a)
                    cyc1 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc1):
                        others = [cyc1[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 2
                    e1 = get_idx(a, b)
                    e2 = get_idx(b, d)
                    e3 = get_idx(d, c)
                    e4 = get_idx(c, a)
                    cyc2 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc2):
                        others = [cyc2[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                    
                    # Cycle 3
                    e1 = get_idx(a, c)
                    e2 = get_idx(c, b)
                    e3 = get_idx(b, d)
                    e4 = get_idx(d, a)
                    cyc3 = [e1, e2, e3, e4]
                    for i, e in enumerate(cyc3):
                        others = [cyc3[j] for j in range(4) if j != i]
                        cycles_for_edge[e].append(tuple(others))
                        
    return cycles_for_edge

CYCLES_FOR_EDGE = _precompute_cycles()

def _can_assign_new_color(e, colors):
    """
    Check if edge e can be assigned a new unique color.
    Condition: For all 4-cycles containing e, the other 3 edges do not have 3 distinct colors.
    """
    for others in CYCLES_FOR_EDGE[e]:
        c1 = colors[others[0]]
        c2 = colors[others[1]]
        c3 = colors[others[2]]
        if c1 != c2 and c2 != c3 and c1 != c3:
            return False
    return True

def _get_best_existing_color(e, colors):
    """
    If we cannot assign a new color, pick an existing color.
    Heuristic: Pick the most frequent color among edges in 4-cycles with e.
    """
    counter = Counter()
    for others in CYCLES_FOR_EDGE[e]:
        for oe in others:
            counter[colors[oe]] += 1
    if not counter:
        return 0
    return counter.most_common(1)[0][0]

def _greedy_max_colors(seed):
    random.seed(seed)
    colors = [0] * NUM_EDGES
    next_color = 1
    edge_order = list(range(NUM_EDGES))
    random.shuffle(edge_order)
    
    for e in edge_order:
        if _can_assign_new_color(e, colors):
            colors[e] = next_color
            next_color += 1
        else:
            colors[e] = _get_best_existing_color(e, colors)
            
    return colors

def _count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    best_colors = [0] * NUM_EDGES
    best_count = 1
    
    # Run multiple seeds
    # Each run is very fast, so we can do many.
    for seed in range(500):
        if time.time() - start_time > 55:
            break
        colors = _greedy_max_colors(seed)
        count = _count_colors(colors)
        if count > best_count:
            best_count = count
            best_colors = colors[:]
            
    return best_colors

# EVOLVE-BLOCK-END
