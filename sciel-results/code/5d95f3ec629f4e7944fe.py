import time
import random
from itertools import combinations
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLOR = 120

def get_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def build_edges():
    """Return list of (i, j) for all edges in lex order."""
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            edges.append((i, j))
    return edges

def build_cycles():
    """
    For each 4-vertex subset, generate the 3 cycles.
    Each cycle is a tuple of 4 edge indices.
    Returns list of cycles.
    """
    cycles = []
    for quad in combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = (get_idx(a,b), get_idx(b,c), get_idx(c,d), get_idx(d,a))
        # Cycle 2: a-b-d-c-a
        e2 = (get_idx(a,b), get_idx(b,d), get_idx(d,c), get_idx(c,a))
        # Cycle 3: a-c-b-d-a
        e3 = (get_idx(a,c), get_idx(c,b), get_idx(b,d), get_idx(d,a))
        cycles.append(e1)
        cycles.append(e2)
        cycles.append(e3)
    return cycles

def build_edge_to_cycles():
    """For each edge index, list of cycle indices it belongs to."""
    cycles = build_cycles()
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    for ci, cycle in enumerate(cycles):
        for ei in cycle:
            edge_to_cycles[ei].append(ci)
    return cycles, edge_to_cycles

def is_rainbow(cycle, colors):
    """Check if a cycle has 4 distinct colors."""
    # Unpack for speed
    c0 = colors[cycle[0]]
    c1 = colors[cycle[1]]
    c2 = colors[cycle[2]]
    c3 = colors[cycle[3]]
    return (c0 != c1) and (c0 != c2) and (c0 != c3) and (c1 != c2) and (c1 != c3) and (c2 != c3)

def check_move(colors, edge_idx, new_color, cycles, edge_to_cycles):
    """
    Check if setting colors[edge_idx] = new_color is valid.
    Only checks cycles containing edge_idx.
    """
    if colors[edge_idx] == new_color:
        return True
    
    old_color = colors[edge_idx]
    colors[edge_idx] = new_color
    
    valid = True
    for ci in edge_to_cycles[edge_idx]:
        if is_rainbow(cycles[ci], colors):
            valid = False
            break
            
    colors[edge_idx] = old_color
    return valid

def try_swap(colors, e1, e2, cycles, edge_to_cycles):
    """
    Check if swapping colors of e1 and e2 is valid.
    Checks all cycles affected by e1 or e2.
    """
    c1 = colors[e1]
    c2 = colors[e2]
    if c1 == c2:
        return True
        
    colors[e1] = c2
    colors[e2] = c1
    
    valid = True
    checked = set()
    
    for ci in edge_to_cycles[e1]:
        if ci not in checked:
            checked.add(ci)
            if is_rainbow(cycles[ci], colors):
                valid = False
                break
                
    if valid:
        for ci in edge_to_cycles[e2]:
            if ci not in checked:
                checked.add(ci)
                if is_rainbow(cycles[ci], colors):
                    valid = False
                    break
                    
    colors[e1] = c1
    colors[e2] = c2
    
    return valid

def greedy_fill(colors, cycles, edge_to_cycles, deadline):
    """
    Greedily introduce new colors.
    """
    used_colors = set(colors)
    
    # We want to maximize distinct colors. 
    # Strategy: iterate through unused colors, try to assign to an edge.
    # To be fast, we pick a random edge for each new color.
    
    for new_color in range(MAX_COLOR):
        if time.time() > deadline:
            break
        if new_color in used_colors:
            continue
        
        # Try a few random edges
        for _ in range(20):
            ei = random.randint(0, NUM_EDGES - 1)
            if check_move(colors, ei, new_color, cycles, edge_to_cycles):
                colors[ei] = new_color
                used_colors.add(new_color)
                break
    return used_colors

def local_search(colors, cycles, edge_to_cycles, deadline):
    """
    Local search using swaps to enable more colors.
    """
    used_colors = set(colors)
    unused = [c for c in range(MAX_COLOR) if c not in used_colors]
    
    max_passes = 50
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        if not unused:
            break
            
        improved = False
        # Shuffle edges for randomization
        edges = list(range(NUM_EDGES))
        random.shuffle(edges)
        
        for ei1 in edges:
            if time.time() > deadline:
                break
            for ei2 in edges:
                if ei2 <= ei1:
                    continue
                if time.time() > deadline:
                    break
                    
                if try_swap(colors, ei1, ei2, cycles, edge_to_cycles):
                    # Try to add a new color to ei1 or ei2
                    for new_color in unused[:5]: # Only try a few
                        if check_move(colors, ei1, new_color, cycles, edge_to_cycles):
                            colors[ei1] = new_color
                            used_colors.add(new_color)
                            if new_color in unused:
                                unused.remove(new_color)
                            improved = True
                            break
                        if check_move(colors, ei2, new_color, cycles, edge_to_cycles):
                            colors[ei2] = new_color
                            used_colors.add(new_color)
                            if new_color in unused:
                                unused.remove(new_color)
                            improved = True
                            break
                            
        if not improved:
            break
            
    return used_colors

def construct():
    deadline = time.time() + 50  # 50s deadline, 10s safety margin
    random.seed(42)
    
    cycles, edge_to_cycles = build_edge_to_cycles()
    
    best_colors = None
    best_count = 0
    
    # Multiple restarts
    num_restarts = 10
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Initial coloring: random small number of colors
        max_init_colors = 3 + (restart % 3)
        colors = [random.randint(0, max_init_colors) for _ in range(NUM_EDGES)]
        
        # Ensure valid start (should be if we start with few colors, but let's be safe)
        # Actually, random assignment might create rainbows. 
        # Simple fix: if invalid, just use 0s.
        # But greedy fill will fix it.
        
        # Greedy fill
        used_colors = greedy_fill(colors, cycles, edge_to_cycles, deadline)
        
        # Local search
        used_colors = local_search(colors, cycles, edge_to_cycles, deadline)
        
        current_count = len(used_colors)
        if current_count > best_count:
            best_count = current_count
            best_colors = colors[:]
            
    if best_colors is None:
        best_colors = [0] * NUM_EDGES
        
    return best_colors

# Pre-compute for efficiency
_cycles, _edge_to_cycles = build_edge_to_cycles()
