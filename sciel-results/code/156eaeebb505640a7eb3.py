import time
import random
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    """Return the lexicographic index of edge (i,j) where 0 <= i < j < N."""
    if i > j:
        i, j = j, i
    # Row i has edges (i, i+1), (i, i+2), ..., (i, N-1)
    # Number of edges before row i: sum_{k=0}^{i-1} (N-1-k) = i*(N-1) - i*(i-1)//2
    # Position within row i: j - i - 1
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def build_cycle_data():
    """Precompute all 4-cycles and which edges they contain.
    Returns list of (edges_tuple, vertex_set) for all 3 cycles on each 4-vertex subset."""
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles on 4 vertices:
        # 1. a-b-c-d-a: edges (a,b), (b,c), (c,d), (d,a)
        # 2. a-b-d-c-a: edges (a,b), (b,d), (d,c), (c,a)
        # 3. a-c-b-d-a: edges (a,c), (c,b), (b,d), (d,a)
        cycles.append((edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(a,d)))
        cycles.append((edge_idx(a,b), edge_idx(b,d), edge_idx(c,d), edge_idx(a,c)))
        cycles.append((edge_idx(a,c), edge_idx(b,c), edge_idx(b,d), edge_idx(a,d)))
    return cycles

def check_validity(colors, cycles):
    """Check that no 4-cycle has 4 distinct colors. Returns True if valid."""
    for e1, e2, e3, e4 in cycles:
        if len({colors[e1], colors[e2], colors[e3], colors[e4]}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    """
    Construct a coloring of K_16 edges with no rainbow 4-cycle.
    Maximizes the number of distinct colors used.
    
    Strategy: Start from transitive coloring (15 colors), then greedily try to
    introduce new colors on edges while maintaining validity.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s margin before 90s kill
    
    cycles = build_cycle_data()
    
    # Precompute: for each edge, which cycles contain it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, (e1, e2, e3, e4) in enumerate(cycles):
        for e in (e1, e2, e3, e4):
            edge_to_cycles[e].append(ci)
    
    # Start with transitive coloring: color(i,j) = i for all i<j
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = edge_idx(i, j)
            colors[idx] = i
    
    # This gives 16 colors: 0,1,...,14,15 (edge (14,15) has color 14)
    # Wait, transitive gives colors 0..14, that's 15 colors.
    # Edge (14,15) gets color 14. So colors used are {0,1,...,14} = 15 colors.
    
    # Verify initial validity
    assert check_validity(colors, cycles), "Transitive coloring should be valid"
    
    best_colors = colors[:]
    best_count = count_colors(colors)
    
    # Try to improve by assigning new colors to edges
    # We'll do a greedy search: for each edge, try to assign a new color
    # (one not currently used) and check if the coloring remains valid.
    
    # Build a pool of candidate new colors (0..119)
    all_possible_colors = list(range(120))
    
    # Do multiple passes with random ordering
    random.seed(42)
    
    for attempt in range(50000):
        if time.time() > deadline:
            break
        
        # Determine which colors are currently used
        used_colors = set(best_colors)
        available_new_colors = [c for c in all_possible_colors if c not in used_colors]
        
        if not available_new_colors:
            break
        
        # Pick a random edge to try
        e = random.randint(0, NUM_EDGES - 1)
        # Pick a random new color
        new_color = random.choice(available_new_colors)
        
        old_color = best_colors[e]
        if new_color == old_color:
            continue
        
        # Try the change
        best_colors[e] = new_color
        
        # Check only cycles containing edge e
        valid = True
        for ci in edge_to_cycles[e]:
            e1, e2, e3, e4 = cycles[ci]
            if len({best_colors[e1], best_colors[e2], best_colors[e3], best_colors[e4]}) == 4:
                valid = False
                break
        
        if valid:
            new_count = count_colors(best_colors)
            if new_count > best_count:
                best_count = new_count
                # Keep this improvement
            else:
                # Revert if it doesn't improve (keep the coloring that maximizes colors)
                best_colors[e] = old_color
        else:
            # Revert
            best_colors[e] = old_color
    
    # Final check
    if not check_validity(best_colors, cycles):
        # Fallback to transitive
        for i in range(N):
            for j in range(i + 1, N):
                idx = edge_idx(i, j)
                best_colors[idx] = i
    
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
    # Quick validity check
    cycles = build_cycle_data()
    print(f"Valid: {check_validity(result, cycles)}")
