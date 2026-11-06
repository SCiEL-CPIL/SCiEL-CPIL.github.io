import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 50.0

    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute cycles and their edge indices
    cycles = []
    # Map edge index to list of cycle indices containing it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][c]
        e3 = edge_idx[c][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][d]
        e3 = edge_idx[d][c]
        e4 = edge_idx[c][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_idx[a][c]
        e2 = edge_idx[c][b]
        e3 = edge_idx[b][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

    def is_bad_cycle(colors, c_idx):
        e1, e2, e3, e4 = c_idx
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    # Initialize with recursive bisection
    colors = [0] * NUM_EDGES
    color_id = 0
    def process_partition(vertices):
        nonlocal color_id
        if len(vertices) <= 1:
            return
        mid = len(vertices) // 2
        left = vertices[:mid]
        right = vertices[mid:]
        for u in left:
            for v in right:
                idx = edge_idx[u][v]
                colors[idx] = color_id
        color_id += 1
        process_partition(left)
        process_partition(right)
    process_partition(list(range(N)))
    
    # Initial state
    freq = [0] * MAX_COLORS
    for c in colors:
        freq[c] += 1
        
    # Calculate initial bad cycles (should be 0)
    bad_count = 0
    for c_idx in cycles:
        if is_bad_cycle(colors, c_idx):
            bad_count += 1
            
    best_score = len(set(colors))
    best_colors = colors.copy()
    
    random.seed(42)
    
    # Local Search
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = colors[e]
        
        # Heuristic: If old_color has high frequency, try to move it to a new/rare color
        # Pick a new color
        # Bias towards unused colors if possible
        # But random is often better for escape.
        # Let's try: 50% random, 50% specific strategy
        if random.random() < 0.5:
            new_color = random.randint(0, MAX_COLORS - 1)
        else:
            # Try to pick a color with low freq or unused
            # Simple: pick random unused if available
            unused = [i for i in range(MAX_COLORS) if freq[i] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)

        if new_color == old_color:
            continue

        # Calculate delta in bad_count
        # Only check cycles containing edge e
        delta_bad = 0
        for cid in edge_to_cycles[e]:
            c_idx = cycles[cid]
            # Check if cycle was bad before
            was_bad = is_bad_cycle(colors, c_idx)
            # Temporarily change color
            colors[e] = new_color
            is_bad_now = is_bad_cycle(colors, c_idx)
            # Revert
            colors[e] = old_color
            
            if was_bad and not is_bad_now:
                delta_bad -= 1
            elif not was_bad and is_bad_now:
                delta_bad += 1
                
        # Accept if no new bad cycles
        if delta_bad <= 0:
            colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            bad_count += delta_bad
            
            current_score = len(set(colors))
            if current_score > best_score:
                best_score = current_score
                best_colors = colors.copy()
                
    return best_colors

if __name__ == "__main__":
    result = construct()
    print(f"Colors used: {len(set(result))}")
