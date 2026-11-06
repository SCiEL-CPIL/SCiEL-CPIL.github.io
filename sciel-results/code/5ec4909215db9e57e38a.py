import time
import random
import math
from itertools import combinations

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLOR = 120

# Precompute edge index mapping
def edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * (N - 1) - i * (i - 1) // 2 + (j - i - 1)

# Precompute all 4-cycles (3 per 4-vertex subset)
CYCLES = []
for a, b, c, d in combinations(range(N), 4):
    # Three distinct 4-cycles on {a,b,c,d}
    CYCLES.append((edge_idx(a, b), edge_idx(b, c), edge_idx(c, d), edge_idx(d, a)))
    CYCLES.append((edge_idx(a, b), edge_idx(b, d), edge_idx(d, c), edge_idx(c, a)))
    CYCLES.append((edge_idx(a, c), edge_idx(c, b), edge_idx(b, d), edge_idx(d, a)))

NUM_CYCLES = len(CYCLES)

# Precompute for speed: for each edge, which cycles contain it
edge_to_cycles = [[] for _ in range(NUM_EDGES)]
for ci, cycle in enumerate(CYCLES):
    for edge in cycle:
        edge_to_cycles[edge].append(ci)

def check_cycle_rainbow(colors, cycle, edge_to_check, new_color):
    """
    Check if setting edge_to_check to new_color makes this cycle a rainbow.
    Returns True if it IS a rainbow (violation).
    """
    c0 = cycle[0]
    c1 = cycle[1]
    c2 = cycle[2]
    c3 = cycle[3]
    
    # Get colors of the other 3 edges
    # We need to handle the case where edge_to_check is one of the cycle edges
    # The cycle tuple is (e1, e2, e3, e4)
    
    colors_other = []
    for e in cycle:
        if e == edge_to_check:
            col = new_color
        else:
            col = colors[e]
        colors_other.append(col)
        
    # Check if all 4 are distinct
    return len(set(colors_other)) == 4

def is_valid_assignment(colors, edge, new_color):
    """
    Check if assigning new_color to edge is valid (no new rainbow 4-cycles).
    """
    for ci in edge_to_cycles[edge]:
        cycle = CYCLES[ci]
        if check_cycle_rainbow(colors, cycle, edge, new_color):
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    best_score = 0
    best_colors = [0] * NUM_EDGES
    
    # We will try multiple random restarts of the greedy process
    # Each restart:
    # 1. Start with all 0s.
    # 2. Greedily try to add new colors 1, 2, 3... to random edges.
    # 3. If stuck, try to "shake" the state by randomly re-coloring some edges to 0.
    
    rng = random.Random(42)
    
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        colors = [0] * NUM_EDGES
        used_colors = {0}
        next_color = 1
        
        # Greedy phase: try to add next_color
        # We iterate through edges in a random order
        edges = list(range(NUM_EDGES))
        rng.shuffle(edges)
        
        progress_made = True
        while progress_made and next_color < MAX_COLOR:
            progress_made = False
            # Try to place next_color
            placed = False
            for e in edges:
                if is_valid_assignment(colors, e, next_color):
                    colors[e] = next_color
                    used_colors.add(next_color)
                    next_color += 1
                    placed = True
                    progress_made = True
                    break
            
            if not placed:
                # We couldn't place the next color in this pass.
                # Try a few random edges specifically for this color?
                # Or just break and restart/shake.
                # Let's try a few random attempts before giving up on this color
                for _ in range(10):
                    e = rng.randrange(NUM_EDGES)
                    if is_valid_assignment(colors, e, next_color):
                        colors[e] = next_color
                        used_colors.add(next_color)
                        next_color += 1
                        progress_made = True
                        break
                
                if not progress_made:
                    # Stuck. Shake the state.
                    # Randomly reset 10% of edges to 0 to break symmetries
                    for _ in range(NUM_EDGES // 10):
                        e = rng.randrange(NUM_EDGES)
                        colors[e] = 0
                        # Note: This might remove a color from used_colors if it was the only one.
                        # But since we are restarting the "next_color" logic or just continuing,
                        # it's safer to just continue with the current next_color.
                        # However, if we removed the only instance of a color, the count of distinct colors drops.
                        # To keep it simple, we just accept the drop and try to add new colors again.
                        # Actually, we should track distinct colors properly.
                    break
        
        # Update best
        current_score = len(set(colors))
        if current_score > best_score:
            best_score = current_score
            best_colors = colors[:]
            
    return best_colors
