"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    """Map (i,j) with i<j to index 0..119 in lexicographic order."""
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

def build_edge_list():
    """Return list of (i, j) for i < j in lex order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def build_four_cycles():
    """
    For every 4-vertex subset {a,b,c,d} with a<b<c<d,
    there are 3 four-cycles (3 ways to pair into a cycle):
      (a,b,c,d), (a,b,d,c), (a,c,b,d)
    Each cycle has 4 edges.
    Return list of tuples of 4 edge indices.
    """
    cycles = []
    vertices = list(range(N))
    for quad in itertools.combinations(vertices, 4):
        a, b, c, d = quad
        # Three 4-cycles on {a,b,c,d}:
        # 1: a-b-c-d-a  edges: (a,b),(b,c),(c,d),(d,a)
        # 2: a-b-d-c-a  edges: (a,b),(b,d),(d,c),(c,a)
        # 3: a-c-b-d-a  edges: (a,c),(c,b),(b,d),(d,a)
        cycles.append((edge_index(a,b), edge_index(b,c), edge_index(c,d), edge_index(d,a)))
        cycles.append((edge_index(a,b), edge_index(b,d), edge_index(c,d), edge_index(a,c)))
        cycles.append((edge_index(a,c), edge_index(b,c), edge_index(b,d), edge_index(a,d)))
    return cycles

def count_colors(coloring):
    return len(set(coloring))

def is_valid(coloring, cycles):
    for c in cycles:
        e0, e1, e2, e3 = c
        if len({coloring[e0], coloring[e1], coloring[e2], coloring[e3]}) == 4:
            return False
    return True

def bipartite_construction():
    """
    Partition 16 vertices into two sets of 8.
    Color all cross-edges with color 0.
    Color all within-partition edges with distinct colors 1..56.
    Returns 57 distinct colors.
    """
    coloring = [0] * NUM_EDGES
    left = list(range(8))
    right = list(range(8, 16))
    color = 1
    # Within left
    for i in range(8):
        for j in range(i+1, 8):
            coloring[edge_index(i, j)] = color
            color += 1
    # Within right
    for i in range(8, 16):
        for j in range(i+1, 16):
            coloring[edge_index(i, j)] = color
            color += 1
    # Cross edges already 0
    return coloring

def local_search_improve(coloring, cycles, deadline, rng):
    """
    Try to increase the number of distinct colors while maintaining validity.
    Strategy: try to recolor an edge to a new color, or merge/split.
    We'll try: for each edge, try assigning a random new color and check validity.
    Also try: find a color used on exactly one edge, and try to "free" that edge
    by recoloring it to a new color if valid.
    """
    best_coloring = list(coloring)
    best_count = count_colors(best_coloring)
    
    while time.time() < deadline:
        # Try to add a new color by recoloring a random edge
        edge = rng.randint(0, NUM_EDGES - 1)
        old_color = best_coloring[edge]
        new_color = rng.randint(0, 119)
        if new_color == old_color:
            continue
        
        # Check if new_color is already used
        # We want to maximize distinct colors, so prefer unused colors
        used = set(best_coloring)
        if new_color not in used:
            # Great, we'd add a new color
            best_coloring[edge] = new_color
            if is_valid(best_coloring, cycles):
                best_count += 1
            else:
                best_coloring[edge] = old_color
        else:
            # Just try random recoloring to potentially enable future improvements
            best_coloring[edge] = new_color
            if is_valid(best_coloring, cycles):
                pass  # keep it
            else:
                best_coloring[edge] = old_color
        
        # Occasionally try targeted: recolor an edge that currently uses a color
        # shared with many other edges, to a new color
        if rng.random() < 0.1:
            edge = rng.randint(0, NUM_EDGES - 1)
            old_color = best_coloring[edge]
            used = set(best_coloring)
            unused = [c for c in range(120) if c not in used]
            if unused:
                new_color = rng.choice(unused)
                best_coloring[edge] = new_color
                if is_valid(best_coloring, cycles):
                    best_count += 1
                else:
                    best_coloring[edge] = old_color
    
    return best_coloring, best_count

# Precompute cycles once
_CYCLES = build_four_cycles()

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin
    
    # Start with bipartite construction (57 colors, provably valid)
    coloring = bipartite_construction()
    
    # Verify it's valid
    assert is_valid(coloring, _CYCLES), "Bipartite construction should be valid"
    
    base_count = count_colors(coloring)
    
    # Try local search to improve
    rng = random.Random(42)
    improved_coloring, improved_count = local_search_improve(
        coloring, _CYCLES, deadline, rng
    )
    
    if improved_count >= base_count:
        return improved_coloring
    else:
        return coloring

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
