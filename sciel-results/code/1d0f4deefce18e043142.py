import random
import time

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def edge_index(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

# Precompute
EDGE_IDX = {}
for i in range(N):
    for j in range(i + 1, N):
        idx = edge_index(i, j)
        EDGE_IDX[(i, j)] = idx
        EDGE_IDX[(j, i)] = idx

# Precompute cycles per edge
edge_to_cycles = {}
for i in range(N):
    for j in range(i + 1, N):
        eidx = EDGE_IDX[(i, j)]
        cycles = []
        others = [x for x in range(N) if x != i and x != j]
        for x_idx in range(len(others)):
            for y_idx in range(x_idx + 1, len(others)):
                x = others[x_idx]
                y = others[y_idx]
                # Cycle: i-j-x-y-i
                e1 = EDGE_IDX[(i, j)]
                e2 = EDGE_IDX[(j, x)]
                e3 = EDGE_IDX[(x, y)]
                e4 = EDGE_IDX[(y, i)]
                cycles.append((e1, e2, e3, e4))
                # Cycle: i-j-y-x-i
                e2b = EDGE_IDX[(j, y)]
                e3b = EDGE_IDX[(y, x)]
                e4b = EDGE_IDX[(x, i)]
                cycles.append((e1, e2b, e3b, e4b))
        edge_to_cycles[eidx] = cycles

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    random.seed(12345)
    
    # Start with a valid coloring using a few colors
    # Use color = i % 4 for edge (i,j) - let's check validity
    # Actually, let's just start with all 0s and build up
    colors = [0] * NUM_EDGES
    color_count = {0: NUM_EDGES}  # count of each color
    distinct_count = 1
    
    best_colors = colors[:]
    best_distinct = 1
    
    def try_change(e, new_color):
        nonlocal colors, color_count, distinct_count
        old_color = colors[e]
        if old_color == new_color:
            return True  # no change needed
        
        # Check all cycles containing e
        for cycle in edge_to_cycles[e]:
            e1, e2, e3, e4 = cycle
            # Get colors with proposed change
            c1 = new_color if e1 == e else colors[e1]
            c2 = new_color if e2 == e else colors[e2]
            c3 = new_color if e3 == e else colors[e3]
            c4 = new_color if e4 == e else colors[e4]
            if len({c1, c2, c3, c4}) == 4:
                return False
        
        # Valid, apply change
        color_count[old_color] -= 1
        if color_count[old_color] == 0:
            del color_count[old_color]
            distinct_count -= 1
        if new_color not in color_count:
            color_count[new_color] = 0
            distinct_count += 1
        color_count[new_color] += 1
        colors[e] = new_color
        return True
    
    # Local search
    while time.time() < deadline:
        e = random.randint(0, NUM_EDGES - 1)
        new_color = random.randint(0, 119)
        
        if try_change(e, new_color):
            if distinct_count > best_distinct:
                best_distinct = distinct_count
                best_colors = colors[:]
    
    return best_colors
# EVOLVE-BLOCK-END
