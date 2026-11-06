import time
import itertools
import random

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

def get_edge_vertices(idx):
    """Convert edge index back to (i, j)"""
    # Inverse of edge_idx
    # Find i such that cumulative edges up to row i-1 <= idx < cumulative up to row i
    # Row k has N-1-k edges
    # Cumulative before row i: i*(N-1) - i*(i-1)//2
    # We need to find i where cum(i) <= idx < cum(i+1)
    # cum(i) = i*(N-1) - i*(i-1)//2
    # Let's just iterate
    for i in range(N):
        start = i * (N - 1) - i * (i - 1) // 2
        end = start + (N - 1 - i)
        if start <= idx < end:
            j = i + 1 + (idx - start)
            return (i, j)
    return (0, 1)  # fallback

def build_cycle_data():
    """Precompute all 4-cycles (all three cycle structures on each 4-vertex subset)"""
    cycles = []
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Three 4-cycles:
        # 1. a-b-c-d-a
        cycles.append([(a,b), (b,c), (c,d), (d,a)])
        # 2. a-b-d-c-a
        cycles.append([(a,b), (b,d), (d,c), (c,a)])
        # 3. a-c-b-d-a
        cycles.append([(a,c), (c,b), (b,d), (d,a)])
    # Convert to edge indices
    cycle_edges = []
    for cycle in cycles:
        edge_idxs = [edge_idx(u, v) for (u, v) in cycle]
        cycle_edges.append(edge_idxs)
    return cycle_edges

def check_validity(colors, cycle_edges):
    """Check if no 4-cycle has 4 distinct colors"""
    for edges in cycle_edges:
        c0, c1, c2, c3 = colors[edges[0]], colors[edges[1]], colors[edges[2]], colors[edges[3]]
        if len({c0, c1, c2, c3}) == 4:
            return False
    return True

def count_colors(colors):
    return len(set(colors))

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline
    
    # Start with transitive coloring
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i + 1, N):
            idx = edge_idx(i, j)
            colors[idx] = i
    
    # This gives 16 colors: 0-15
    # Actually colors 0-14 from the rule, and we can add more
    
    # Precompute cycles
    cycle_edges = build_cycle_data()
    
    # Try to increase colors by recoloring edges
    # Strategy: for each edge, try assigning a new color (not currently used)
    # and check if validity is maintained
    
    # First, let's see what colors are currently used
    current_colors = set(colors)
    max_color_used = max(colors)
    
    # Try to add new colors by recoloring edges
    # For each edge, try to assign a color > max_color_used
    # But we need to be careful: recoloring might break validity
    
    # Better approach: try to recolor edges to new colors one at a time
    # Start with the 16-color solution
    
    best_colors = colors[:]
    best_count = count_colors(best_colors)
    
    # Try random recoloring: pick an edge, try a new color, check validity
    random.seed(42)
    
    # Precompute for each edge, which cycles it participates in
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, edges in enumerate(cycle_edges):
        for e in edges:
            edge_to_cycles[e].append(ci)
    
    # Try to improve
    for attempt in range(500000):
        if time.time() > deadline - 2:
            break
        
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Try to assign a new color
        # New color should be > max_color_used to increase count
        new_color = max_color_used + 1 + random.randint(0, 50)
        
        if new_color > 119:
            continue
        
        # Temporarily assign
        colors[e] = new_color
        
        # Check only affected cycles
        valid = True
        for ci in edge_to_cycles[e]:
            edges = cycle_edges[ci]
            c0, c1, c2, c3 = colors[edges[0]], colors[edges[1]], colors[edges[2]], colors[edges[3]]
            if len({c0, c1, c2, c3}) == 4:
                valid = False
                break
        
        if valid:
            new_count = count_colors(colors)
            if new_count > best_count:
                best_colors = colors[:]
                best_count = new_count
                max_color_used = max(colors)
        else:
            # Revert
            colors[e] = old_color
    
    return best_colors
