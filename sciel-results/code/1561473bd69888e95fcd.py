import time
import random
import sys

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

# Precompute edge index
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)

# Precompute all 4-cycles: for each 4-set {a,b,c,d}, the 3 cycles
# Cycle (a,b,c,d) uses edges (a,b),(b,c),(c,d),(d,a)
# Cycle (a,b,d,c) uses edges (a,b),(b,d),(d,c),(c,a)
# Cycle (a,c,b,d) uses edges (a,c),(c,b),(b,d),(d,a)
def precompute_cycles():
    cycles = []
    for a in range(N):
        for b in range(a+1, N):
            for c in range(b+1, N):
                for d in range(c+1, N):
                    # Cycle 1: a-b-c-d
                    e1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
                    cycles.append(e1)
                    # Cycle 2: a-b-d-c
                    e2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
                    cycles.append(e2)
                    # Cycle 3: a-c-b-d
                    e3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
                    cycles.append(e3)
    return cycles

# Precompute for each edge, which cycles it belongs to
def precompute_edge_cycles(cycles):
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cyc in enumerate(cycles):
        for e in cyc:
            edge_to_cycles[e].append(ci)
    return edge_to_cycles

# Initial structured coloring: use (i * 7 + j * 3) % 120 for diversity
def initial_coloring():
    colors = [0] * NUM_EDGES
    for i in range(N):
        for j in range(i+1, N):
            colors[edge_idx(i,j)] = (i * 7 + j * 3) % NUM_EDGES
    return colors

def count_distinct(colors):
    return len(set(colors))

def is_valid_cycle(cyc, colors):
    """Check if a 4-cycle has 4 distinct colors (which is INVALID)"""
    c0 = colors[cyc[0]]
    c1 = colors[cyc[1]]
    c2 = colors[cyc[2]]
    c3 = colors[cyc[3]]
    return len([c0, c1, c2, c3]) == 4

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s
    
    cycles = precompute_cycles()
    edge_to_cycles = precompute_edge_cycles(cycles)
    
    # Start with a structured coloring
    colors = initial_coloring()
    
    # Fix initial invalid cycles by recoloring
    # For each invalid cycle, change one edge's color to match another
    for _ in range(100):  # limited iterations
        if time.time() > deadline - 5.0:
            break
        found_fix = False
        for ci, cyc in enumerate(cycles):
            if is_valid_cycle(cyc, colors):  # has 4 distinct = invalid
                # Change edge cyc[3] to match cyc[0]'s color
                colors[cyc[3]] = colors[cyc[0]]
                found_fix = True
                break
        if not found_fix:
            break
    
    best_score = count_distinct(colors)
    best_colors = colors[:]
    
    # Local search: try to increase color count by swapping
    for iteration in range(500000):
        if time.time() > deadline - 3.0:
            break
        
        e1 = random.randint(0, NUM_EDGES - 1)
        e2 = random.randint(0, NUM_EDGES - 1)
        if e1 == e2:
            continue
        
        c1 = colors[e1]
        c2 = colors[e2]
        if c1 == c2:
            continue
        
        # Try swapping
        colors[e1] = c2
        colors[e2] = c1
        
        # Check if still valid (only need to check cycles involving e1 or e2)
        valid = True
        checked = set()
        for ci in edge_to_cycles[e1]:
            if ci not in checked:
                checked.add(ci)
                if is_valid_cycle(cycles[ci], colors):
                    valid = False
                    break
        if valid:
            for ci in edge_to_cycles[e2]:
                if ci not in checked:
                    checked.add(ci)
                    if is_valid_cycle(cycles[ci], colors):
                        valid = False
                        break
        
        if valid:
            new_score = count_distinct(colors)
            if new_score > best_score:
                best_score = new_score
                best_colors = colors[:]
        else:
            # Revert
            colors[e1] = c1
            colors[e2] = c2
    
    return best_colors

# Precompute at module level for speed
_CYCLES = precompute_cycles()
_EDGE_TO_CYCLES = precompute_edge_cycles(_CYCLES)

def construct():
    deadline = time.time() + 85.0
    
    # Start with a structured coloring
    colors = initial_coloring()
    
    # Fix initial invalid cycles
    for _ in range(200):
        if time.time() > deadline - 5.0:
            break
        found_fix = False
        for ci, cyc in enumerate(_CYCLES):
            if is_valid_cycle(cyc, colors):
                colors[cyc[3]] = colors[cyc[0]]
                found_fix = True
                break
        if not found_fix:
            break
    
    best_score = count_distinct(colors)
    best_colors = colors[:]
    
    # Local search with random restarts
    for restart in range(20):
        if time.time() > deadline - 3.0:
            break
        
        # Random perturbation
        for _ in range(5):
            e = random.randint(0, NUM_EDGES - 1)
            colors[e] = random.randint(0, NUM_EDGES - 1)
        
        # Fix any violations
        for _ in range(100):
            if time.time() > deadline - 3.0:
                break
            found_fix = False
            for ci, cyc in enumerate(_CYCLES):
                if is_valid_cycle(cyc, colors):
                    colors[cyc[3]] = colors[cyc[0]]
                    found_fix = True
                    break
            if not found_fix:
                break
        
        # Local search
        for iteration in range(100000):
            if time.time() > deadline - 3.0:
                break
            
            e1 = random.randint(0, NUM_EDGES - 1)
            e2 = random.randint(0, NUM_EDGES - 1)
            if e1 == e2:
                continue
            
            c1 = colors[e1]
            c2 = colors[e2]
            if c1 == c2:
                continue
            
            colors[e1] = c2
            colors[e2] = c1
            
            valid = True
            checked = set()
            for ci in _EDGE_TO_CYCLES[e1]:
                if ci not in checked:
                    checked.add(ci)
                    if is_valid_cycle(_CYCLES[ci], colors):
                        valid = False
                        break
            if valid:
                for ci in _EDGE_TO_CYCLES[e2]:
                    if ci not in checked:
                        checked.add(ci)
                        if is_valid_cycle(_CYCLES[ci], colors):
                            valid = False
                            break
            
            if valid:
                new_score = count_distinct(colors)
                if new_score > best_score:
                    best_score = new_score
                    best_colors = colors[:]
            else:
                colors[e1] = c1
                colors[e2] = c2
    
    return best_colors

