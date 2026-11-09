import time
import itertools
import random
import numpy as np

N = 16
NUM_EDGES = N * (N - 1) // 2  # 120

def construct():
    """
    Constructs a coloring of K_16 maximizing distinct colors with no rainbow C4.
    Uses initial Gallai construction + local search with efficient incremental cycle checking.
    """
    deadline = time.time() + 55
    
    # Precompute edge indices
    def edge_idx(i, j):
        if i > j:
            i, j = j, i
        return i * (N - 1) - i * (i + 1) // 2 + (j - i - 1)
    
    # Precompute all 4-cycles and which edges they contain
    all_cycles = []
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    cycle_id = 0
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        cyc1 = [edge_idx(a,b), edge_idx(b,c), edge_idx(c,d), edge_idx(d,a)]
        all_cycles.append(cyc1)
        for e in cyc1:
            edge_to_cycles[e].append(cycle_id)
        cycle_id += 1
        cyc2 = [edge_idx(a,b), edge_idx(b,d), edge_idx(d,c), edge_idx(c,a)]
        all_cycles.append(cyc2)
        for e in cyc2:
            edge_to_cycles[e].append(cycle_id)
        cycle_id += 1
        cyc3 = [edge_idx(a,c), edge_idx(c,b), edge_idx(b,d), edge_idx(d,a)]
        all_cycles.append(cyc3)
        for e in cyc3:
            edge_to_cycles[e].append(cycle_id)
        cycle_id += 1
        
    # Precompute for each edge, the other 3 edges in each cycle it belongs to
    # edge_cycle_info[e] = list of (other_e1, other_e2, other_e3)
    edge_cycle_info = [[] for _ in range(NUM_EDGES)]
    for e in range(NUM_EDGES):
        for cid in edge_to_cycles[e]:
            cyc = all_cycles[cid]
            others = [x for x in cyc if x != e]
            edge_cycle_info[e].append(tuple(others))

    # Build initial Gallai coloring
    colors = [0] * NUM_EDGES
    next_color = [0]
    
    SPLIT = {4:3, 5:3, 6:3, 7:3, 8:4, 9:3, 10:4, 11:5, 12:6, 13:6, 14:7, 15:6, 16:7}
    
    def assign_gallai(vertices):
        nonlocal colors
        n = len(vertices)
        if n <= 1:
            return
        if n == 2:
            u, v = vertices
            e = edge_idx(u, v)
            colors[e] = next_color[0]
            next_color[0] += 1
            return
        if n == 3:
            u, v, w = vertices
            e1 = edge_idx(u, v)
            e2 = edge_idx(v, w)
            e3 = edge_idx(u, w)
            colors[e1] = next_color[0]
            colors[e2] = next_color[0] + 1
            colors[e3] = next_color[0] + 2
            next_color[0] += 3
            return
        k = SPLIT[n]
        v1 = vertices[:k]
        v2 = vertices[k:]
        cross_color = next_color[0]
        for u in v1:
            for v in v2:
                e = edge_idx(u, v)
                colors[e] = cross_color
        next_color[0] += 1
        assign_gallai(v1)
        assign_gallai(v2)
        
    assign_gallai(list(range(N)))
    
    # Initialize state
    colors_np = np.array(colors, dtype=np.int32)
    color_count = np.zeros(120, dtype=np.int32)
    for c in colors_np:
        color_count[c] += 1
    num_colors = np.sum(color_count > 0)
    best_num_colors = num_colors
    best_colors = colors_np.copy()
    
    # Calculate initial rainbow count
    rainbow_count = 0
    for cyc in all_cycles:
        if len(set(colors_np[c] for c in cyc)) == 4:
            rainbow_count += 1
            
    random.seed(42)
    
    while time.time() < deadline:
        # Determine candidate colors
        # Prefer unused colors to increase num_colors
        unused_mask = color_count == 0
        num_unused = np.sum(unused_mask)
        
        if num_unused > 0:
            # Pick a random unused color
            unused_indices = np.where(unused_mask)[0]
            new_c = int(random.choice(unused_indices))
        else:
            # Pick a random color (will likely be used)
            new_c = random.randint(0, 119)
            
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_c = int(colors_np[e])
        
        if new_c == old_c:
            continue
            
        # Check validity and color count change
        delta_rainbow = 0
        for o1, o2, o3 in edge_cycle_info[e]:
            c1, c2, c3 = colors_np[o1], colors_np[o2], colors_np[o3]
            # Old check
            if len(set([old_c, c1, c2, c3])) == 4:
                delta_rainbow -= 1
            # New check
            if len(set([new_c, c1, c2, c3])) == 4:
                delta_rainbow += 1
                
        if rainbow_count + delta_rainbow > 0:
            continue # Invalid
            
        # Check color count improvement
        # Simulate count change
        old_count = color_count[old_c]
        new_count = color_count[new_c]
        
        delta_colors = 0
        if old_count == 1:
            delta_colors -= 1
        if new_count == 0:
            delta_colors += 1
            
        if num_colors + delta_colors > best_num_colors:
            # Accept
            colors_np[e] = new_c
            color_count[old_c] -= 1
            color_count[new_c] += 1
            num_colors += delta_colors
            rainbow_count += delta_rainbow
            
            best_colors = colors_np.copy()
            best_num_colors = num_colors
            
        elif num_colors + delta_colors == best_num_colors and random.random() < 0.1:
            # Occasionally accept neutral moves to escape local minima
            colors_np[e] = new_c
            color_count[old_c] -= 1
            color_count[new_c] += 1
            num_colors += delta_colors
            rainbow_count += delta_rainbow
            
    return best_colors.tolist()

if __name__ == "__main__":
    res = construct()
    print(f"Number of colors used: {len(set(res))}")
    print(f"First 10 colors: {res[:10]}")
