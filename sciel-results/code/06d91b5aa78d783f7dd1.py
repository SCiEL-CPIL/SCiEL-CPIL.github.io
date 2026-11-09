import time
import random
import numpy as np

N = 16
N_EDGES = N * (N - 1) // 2

def _edge_idx(i, j):
    if i > j:
        i, j = j, i
    return i * N - i * (i + 1) // 2 + (j - i - 1)

def construct():
    deadline = time.time() + 55
    n_edges = N_EDGES
    
    edges_list = []
    for i in range(N):
        for j in range(i + 1, N):
            edges_list.append((i, j))
    
    # Build 4-cycle data: for each edge, list of (e1, e2, e3) triples
    # A 4-cycle involving edge e=(u,v) and others w,x is (u,w)-(w,x)-(x,v) or (u,x)-(x,w)-(w,v)
    # The condition is that the 4 edges (u,v), (u,w), (w,x), (x,v) are not all distinct.
    # So if we change (u,v) to c, we must ensure that for all such cycles, c is not distinct from the other 3.
    # i.e., c must match at least one of the other 3 colors.
    edge_cycles = [[] for _ in range(n_edges)]
    for idx, (u, v) in enumerate(edges_list):
        others = [x for x in range(N) if x != u and x != v]
        for w_idx in range(len(others)):
            for x_idx in range(w_idx + 1, len(others)):
                w = others[w_idx]
                x = others[x_idx]
                e1 = _edge_idx(u, w)
                e2 = _edge_idx(w, x)
                e3 = _edge_idx(x, v)
                edge_cycles[idx].append((e1, e2, e3))
                e4 = _edge_idx(u, x)
                e5 = _edge_idx(x, w)
                e6 = _edge_idx(w, v)
                edge_cycles[idx].append((e4, e5, e6))
    
    # Convert to numpy arrays
    cycle_data = []
    for idx in range(n_edges):
        cycles = edge_cycles[idx]
        if cycles:
            arr = np.array(cycles, dtype=np.int32)
        else:
            arr = np.empty((0, 3), dtype=np.int32)
        cycle_data.append(arr)
    
    # Initialize with Gallai coloring (15 colors)
    colors = np.zeros(n_edges, dtype=np.int32)
    color_counter = 0
    def build_gallai(vertices):
        nonlocal color_counter
        if len(vertices) <= 1:
            return
        mid = len(vertices) // 2
        v1 = vertices[:mid]
        v2 = vertices[mid:]
        c = color_counter
        color_counter += 1
        for u in v1:
            for v in v2:
                colors[_edge_idx(u, v)] = c
        build_gallai(v1)
        build_gallai(v2)
    build_gallai(list(range(N)))
    
    best_colors = colors.copy()
    best_distinct = len(set(colors.tolist()))
    
    color_counts = np.zeros(120, dtype=np.int32)
    for c in colors:
        color_counts[c] += 1
    
    unused_colors = set(range(120)) - set(colors.tolist())
    random.seed(42)
    
    def is_valid_change(edge_idx, new_color, cur_colors):
        cycles = cycle_data[edge_idx]
        if len(cycles) == 0:
            return True
        c1 = cur_colors[cycles[:, 0]]
        c2 = cur_colors[cycles[:, 1]]
        c3 = cur_colors[cycles[:, 2]]
        all_distinct = (c1 != c2) & (c1 != c3) & (c2 != c3)
        new_distinct = (new_color != c1) & (new_color != c2) & (new_color != c3)
        invalid = all_distinct & new_distinct
        return not np.any(invalid)

    # 2-move validity check: change e1 to c1, e2 to c2
    def is_valid_2_change(e1_idx, c1, e2_idx, c2, cur_colors):
        # Temporarily apply changes? No, just check constraints for e1 and e2.
        # Constraints for e1: for each cycle (a,b,c) in e1, if a,b,c are distinct, c1 must match one of them.
        # BUT, if a or b or c is e2, its color is c2.
        # So we need to check with the modified colors.
        
        # This is complex to vectorize without modifying the array.
        # Given the small size, we can just create a copy or handle it carefully.
        # For speed, we'll stick to 1-move for now but optimize the loop.
        pass

    while time.time() < deadline:
        e = random.randint(0, n_edges - 1)
        old_c = int(colors[e])
        
        if len(unused_colors) > 0:
            if random.random() < 0.8:
                c = random.choice(list(unused_colors))
            else:
                c = random.randint(0, 119)
        else:
            c = random.randint(0, 119)
        
        if c == old_c:
            continue
        
        if is_valid_change(e, c, colors):
            colors[e] = c
            color_counts[old_c] -= 1
            color_counts[c] += 1
            if color_counts[old_c] == 0:
                unused_colors.add(old_c)
            if c in unused_colors:
                unused_colors.discard(c)
            
            current_distinct = len(set(colors.tolist()))
            if current_distinct > best_distinct:
                best_distinct = current_distinct
                best_colors = colors.copy()
                
            # Occasional 2-move attempt to escape local minima
            if random.random() < 0.1:
                e2 = random.randint(0, n_edges - 1)
                if e2 != e:
                    old_c2 = int(colors[e2])
                    if len(unused_colors) > 0:
                        c2 = random.choice(list(unused_colors))
                    else:
                        c2 = random.randint(0, 119)
                    
                    if c2 != old_c2:
                        # Check if (e, c) and (e2, c2) is valid
                        # We need to check cycles for e and e2 with the potential new colors.
                        # Simple approach: apply, check, revert if invalid.
                        # Since is_valid_change uses current colors, we must be careful.
                        
                        # Let's just do a single step for now to keep it simple and fast.
                        # The 1-step search with good initialization often works.
                        pass

    return best_colors.tolist()
