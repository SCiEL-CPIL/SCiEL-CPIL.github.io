import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit
    
    n = 16
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            edges.append((i, j))
    
    edge_idx = {}
    for idx, (i, j) in enumerate(edges):
        edge_idx[(i, j)] = idx
        edge_idx[(j, i)] = idx
    
    # Precompute all K4s as lists of 6 edge indices
    k4s = []
    for combo in itertools.combinations(range(n), 4):
        a, b, c, d = combo
        e1 = edge_idx[(a, b)]
        e2 = edge_idx[(a, c)]
        e3 = edge_idx[(a, d)]
        e4 = edge_idx[(b, c)]
        e5 = edge_idx[(b, d)]
        e6 = edge_idx[(c, d)]
        k4s.append((e1, e2, e3, e4, e5, e6))
        
    k4_edges_np = np.array(k4s, dtype=np.int32) # shape (1820, 6)

    def count_violations(colours):
        """Return number of K4s with fewer than 5 distinct colors using NumPy."""
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np] # shape (1820, 6)
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((1820, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))

    def is_valid(colours):
        return count_violations(colours) == 0

    # Start with seed (i+j) % 17
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
    
    if not is_valid(current):
        for _ in range(100):
            if time.time() > deadline: break
            rand = [random.randint(0, 16) for _ in range(120)]
            if is_valid(rand):
                current = rand
                break

    for target_max_colors in range(16, 12, -1):
        if time.time() > deadline:
            break
            
        if len(set(current)) <= target_max_colors:
            continue
            
        current_set = set(current)
        colors_list = sorted(current_set)
        
        if len(colors_list) <= target_max_colors:
            continue
            
        merged = False
        for i in range(len(colors_list)):
            for j in range(i + 1, len(colors_list)):
                if time.time() > deadline:
                    break
                
                c1 = colors_list[i]
                c2 = colors_list[j]
                
                new_colours = [c2 if c == c1 else c for c in current]
                
                if is_valid(new_colours):
                    current = new_colours
                    merged = True
                    break
            if merged:
                break
                
        if not merged:
            K = target_max_colors
            mapping = {}
            next_id = 0
            init_colours = []
            for c in current:
                if c not in mapping:
                    if next_id >= K:
                        break
                    mapping[c] = next_id
                    next_id += 1
                init_colours.append(mapping[c])
            
            if len(init_colours) < 120:
                continue
                
            if is_valid(init_colours):
                current = init_colours
                continue
            
            best_colours = init_colours
            best_violations = count_violations(init_colours)
            
            for step in range(1000):
                if time.time() > deadline:
                    break
                
                if best_violations == 0:
                    break
                    
                eidx = random.randint(0, 119)
                old_color = best_colours[eidx]
                new_color = random.randint(0, K - 1)
                
                if new_color == old_color:
                    continue
                
                best_colours[eidx] = new_color
                new_violations = count_violations(best_colours)
                
                if new_violations <= best_violations:
                    best_violations = new_violations
                else:
                    best_colours[eidx] = old_color
                    
            if best_violations == 0:
                current = best_colours
            else:
                success = False
                for restart in range(20):
                    if time.time() > deadline:
                        break
                    random.seed(int(time.time() * 1000) % 100000)
                    rand_colours = [random.randint(0, K - 1) for _ in range(120)]
                    v = count_violations(rand_colours)
                    
                    for step in range(500):
                        if time.time() > deadline:
                            break
                        if v == 0:
                            success = True
                            break
                        eidx = random.randint(0, 119)
                        old_c = rand_colours[eidx]
                        new_c = random.randint(0, K - 1)
                        if new_c == old_c: continue
                        rand_colours[eidx] = new_c
                        new_v = count_violations(rand_colours)
                        if new_v <= v:
                            v = new_v
                        else:
                            rand_colours[eidx] = old_c
                            
                    if success:
                        current = rand_colours
                        break
                        
                if not success:
                    break

    if not is_valid(current):
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            current = seed
            
    return current
