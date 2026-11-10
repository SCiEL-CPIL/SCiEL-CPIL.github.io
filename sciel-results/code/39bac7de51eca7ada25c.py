import time
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    """Get index of edge (i,j) in lexicographic ordering."""
    if i > j:
        i, j = j, i
    # Count edges before (i,j): edges (0,1)..(0,i-1), (1,2)..(1,i), ..., (i, i+1)..(i,j)
    # For vertex v < i: edges (v, v+1)..(v, i) -> (i - v) edges... 
    # Actually: for each v from 0 to i-1, edges (v, v+1), (v, v+2), ..., (v, i) -> i - v edges
    # Then edges (i, i+1), ..., (i, j) -> j - i edges
    idx = 0
    for v in range(i):
        idx += (i - v)
    idx += (j - i)
    return idx

# Precompute edge indices
def build_edge_map():
    edge_map = {}
    idx = 0
    for i in range(N):
        for j in range(i + 1, N):
            edge_map[(i, j)] = idx
            idx += 1
    return edge_map

EDGE_MAP = build_edge_map()

def get_4cycles_with_edge(i, j):
    """Yield all 4-cycles containing edge (i,j). Each cycle is (i,j,k,l) meaning edges (i,j),(j,k),(k,l),(l,i)."""
    others = [v for v in range(N) if v != i and v != j]
    for k in others:
        for l in others:
            if l == k:
                continue
            # The 4-cycle is i-j-k-l-i. We need k != l, and the cycle must be valid
            # (all four vertices distinct, which they are since i,j,k,l are all distinct)
            yield (i, j, k, l)

def count_violations(colors, i, j, new_color):
    """Check if assigning new_color to edge (i,j) creates any 4-cycle with 4 distinct colors."""
    others = [v for v in range(N) if v != i and v != j]
    for k in others:
        for l in others:
            if l == k:
                continue
            # 4-cycle: i-j-k-l-i
            # Edges: (i,j), (j,k), (k,l), (l,i)
            e1 = new_color  # (i,j)
            e2 = colors[EDGE_MAP.get((j, k), EDGE_MAP.get((k, j)))]
            e3 = colors[EDGE_MAP.get((k, l), EDGE_MAP.get((l, k)))]
            e4 = colors[EDGE_MAP.get((l, i), EDGE_MAP.get((i, l)))]
            if len({e1, e2, e3, e4}) == 4:
                return 1
    return 0

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    colors = [0] * NUM_EDGES
    used_colors = {0}
    next_color = 1
    
    # Order edges: process in a strategic order
    # Try edges in lexicographic order, but we could also try random orders
    edge_list = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # For each edge, try to assign the next available color
    for (i, j) in edge_list:
        if time.time() > deadline:
            break
        if next_color >= 120:
            break
        
        # Check if we can assign next_color to this edge
        idx = EDGE_MAP[(i, j)]
        old_color = colors[idx]
        
        if count_violations(colors, i, j, next_color) == 0:
            colors[idx] = next_color
            used_colors.add(next_color)
            next_color += 1
    
    return colors

# Precompute edge map at module level for speed
EDGE_MAP = build_edge_map()
