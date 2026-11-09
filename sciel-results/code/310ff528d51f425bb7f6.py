import random
import time
import itertools
import sys

N = 20
C = 20

# Precompute all 4-cycles in K_N
# A 4-cycle is a set of 4 vertices {u,v,w,x} with edges (u,v), (v,w), (w,x), (x,u)
# In a complete graph, any 4 vertices form 3 distinct 4-cycles (up to rotation/reflection).
# The cycles are:
# 1-2-3-4-1
# 1-2-4-3-1
# 1-3-2-4-1
# We represent a cycle by the ordered tuple of vertices (u,v,w,x)
# such that edges are (u,v), (v,w), (w,x), (x,u).

def generate_cycles(n):
    cycles = []
    for combo in itertools.combinations(range(n), 4):
        u, v, w, x = combo
        # 3 distinct cycles on these 4 vertices
        # 1. u-v-w-x-u
        cycles.append((u, v, w, x))
        # 2. u-v-x-w-u
        cycles.append((u, v, x, w))
        # 3. u-w-v-x-u
        cycles.append((u, w, v, x))
    return cycles

CYCLES = generate_cycles(N)
NUM_CYCLES = len(CYCLES)

def count_invalid(col):
    """
    col: 2D list or numpy array of shape (N, N). -1 means no edge.
    Returns number of 4-cycles where all 4 edges exist and colors are not distinct.
    """
    invalid = 0
    for u, v, w, x in CYCLES:
        c1 = col[u][v]
        if c1 == -1: continue
        c2 = col[v][w]
        if c2 == -1: continue
        c3 = col[w][x]
        if c3 == -1: continue
        c4 = col[x][u]
        if c4 == -1: continue
        
        # Check distinctness
        if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
            invalid += 1
    return invalid

def get_invalid_cycle(col):
    """Returns one invalid cycle (u,v,w,x) or None."""
    for u, v, w, x in CYCLES:
        c1 = col[u][v]
        if c1 == -1: continue
        c2 = col[v][w]
        if c2 == -1: continue
        c3 = col[w][x]
        if c3 == -1: continue
        c4 = col[x][u]
        if c4 == -1: continue
        
        if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
            return (u, v, w, x)
    return None

def _repair_and_grow(seed):
    rng = random.Random(seed)
    
    # Initialize with complete graph and random colors
    col = [[-1] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            c = rng.randint(0, C - 1)
            col[i][j] = c
            col[j][i] = c
            
    # Phase 1: Repair (remove edges or recolor to make valid)
    # We prioritize recoloring if it helps, otherwise remove.
    # To keep it simple and fast, we will just remove edges from invalid cycles.
    # This guarantees validity but might lose too many edges.
    # Let's try a few recoloring attempts before removing.
    
    max_repair_steps = 10000
    steps = 0
    while steps < max_repair_steps:
        invalid_cycle = get_invalid_cycle(col)
        if invalid_cycle is None:
            break
        
        u, v, w, x = invalid_cycle
        # Edges in cycle: (u,v), (v,w), (w,x), (x,u)
        edges_in_cycle = [(u,v), (v,w), (w,x), (x,u)]
        
        # Try to recolor one of the edges to a random color
        success = False
        for _ in range(5): # Try a few recolors
            e = rng.choice(edges_in_cycle)
            a, b = e
            old_c = col[a][b]
            new_c = rng.randint(0, C - 1)
            if new_c == old_c:
                continue
                
            col[a][b] = new_c
            col[b][a] = new_c
            
            # Check if this reduced the number of invalid cycles
            # We don't want to calculate full count every time, it's slow.
            # But we can check if this specific cycle is fixed?
            # Or just accept the move if it doesn't increase invalid count?
            # For speed, let's just remove an edge if recoloring is risky.
            # Actually, let's just remove an edge. It's deterministic and safe.
            # We can optimize by removing the edge that is in the most invalid cycles?
            # For now, random removal.
            break # Skip recoloring for now to ensure validity quickly
            
        # Remove a random edge from the invalid cycle
        a, b = rng.choice(edges_in_cycle)
        col[a][b] = -1
        col[b][a] = -1
        
        steps += 1
        
    if get_invalid_cycle(col) is not None:
        return [] # Failed to repair

    # Phase 2: Grow (add edges)
    # List of all possible edges not currently in graph
    missing_edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if col[i][j] == -1:
                missing_edges.append((i, j))
                
    rng.shuffle(missing_edges)
    
    # Try to add edges back
    for u, v in missing_edges:
        # Try random colors
        found_color = False
        colors = list(range(C))
        rng.shuffle(colors)
        for c in colors:
            col[u][v] = c
            col[v][u] = c
            
            if get_invalid_cycle(col) is None:
                found_color = True
                break
            else:
                col[u][v] = -1
                col[v][u] = -1
                
        if not found_color:
            # Could not add this edge with any color
            pass
            
    # Convert to output format
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if col[i][j] != -1:
                edges.append([i, j, col[i][j]])
                
    return edges

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    best_edges = []
    max_edges = 0
    
    seed = 42
    while time.time() < deadline:
        edges = _repair_and_grow(seed)
        if len(edges) > max_edges:
            max_edges = len(edges)
            best_edges = edges
        seed += 1
        
    return best_edges
