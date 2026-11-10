import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
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

    k4_edges_np = np.array(k4s, dtype=np.int32)  # shape (1820, 6)
    num_k4s = len(k4s)

    # Precompute: for each edge, which K4s contain it
    edge_to_k4s = [[] for _ in range(120)]
    for k4_idx in range(num_k4s):
        for eidx in k4_edges_np[k4_idx]:
            edge_to_k4s[eidx].append(k4_idx)
    edge_to_k4s = [np.array(lst, dtype=np.int32) for lst in edge_to_k4s]

    def count_violations(colours):
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((num_k4s, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        counts = np.sum(unique_mask, axis=1)
        return int(np.sum(counts < 5))

    def is_valid(colours):
        return count_violations(colours) == 0

    def compute_k4_counts(colours):
        c_np = np.asarray(colours, dtype=np.int32)
        k4_colors = c_np[k4_edges_np]
        sorted_colors = np.sort(k4_colors, axis=1)
        unique_mask = np.ones((num_k4s, 6), dtype=bool)
        unique_mask[:, 1:] = (sorted_colors[:, 1:] != sorted_colors[:, :-1])
        return np.sum(unique_mask, axis=1)

    # Start with seed (i+j) % 17
    current = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]

    if not is_valid(current):
        for _ in range(100):
            if time.time() > deadline:
                break
            rand = [random.randint(0, 16) for _ in range(120)]
            if is_valid(rand):
                current = rand
                break

    def local_search(colours, K, max_steps=10000):
        """Local search with incremental K4 count updates."""
        colours = list(colours)
        k4_counts = compute_k4_counts(colours)
        v = int(np.sum(k4_counts < 5))
        
        if v == 0:
            return colours, 0
            
        for step in range(max_steps):
            if time.time() > deadline:
                break
            if v == 0:
                break
            
            violated = np.where(k4_counts < 5)[0]
            if len(violated) == 0:
                v = 0
                break
            
            k4_idx = int(violated[random.randint(0, len(violated) - 1)])
            k4_edge_idxs = k4_edges_np[k4_idx]
            
            best_new_v = v
            best_edge = -1
            best_color = -1
            best_affected = None
            
            for eidx in k4_edge_idxs:
                old_c = colours[eidx]
                affected = edge_to_k4s[eidx]
                old_counts = k4_counts[affected].copy()
                
                for new_c in range(K):
                    if new_c == old_c:
                        continue
                    
                    colours[eidx] = new_c
                    for ak4 in affected:
                        ak4_e = k4_edges_np[ak4]
                        ak4_c = np.array([colours[ae] for ae in ak4_e], dtype=np.int32)
                        sorted_ak4 = np.sort(ak4_c)
                        um = np.ones(6, dtype=bool)
                        um[1:] = (sorted_ak4[1:] != sorted_ak4[:-1])
                        k4_counts[ak4] = np.sum(um)
                    
                    new_v = int(np.sum(k4_counts < 5))
                    if new_v < best_new_v:
                        best_new_v = new_v
                        best_edge = eidx
                        best_color = new_c
                        best_affected = affected
                    
                    k4_counts[affected] = old_counts
                    colours[eidx] = old_c
            
            if best_edge >= 0:
                colours[best_edge] = best_color
                for ak4 in best_affected:
                    ak4_e = k4_edges_np[ak4]
                    ak4_c = np.array([colours[ae] for ae in ak4_e], dtype=np.int32)
                    sorted_ak4 = np.sort(ak4_c)
                    um = np.ones(6, dtype=bool)
                    um[1:] = (sorted_ak4[1:] != sorted_ak4[:-1])
                    k4_counts[ak4] = np.sum(um)
                v = best_new_v
            else:
                eidx = random.randint(0, 119)
                colours[eidx] = random.randint(0, K - 1)
                affected = edge_to_k4s[eidx]
                for ak4 in affected:
                    ak4_e = k4_edges_np[ak4]
                    ak4_c = np.array([colours[ae] for ae in ak4_e], dtype=np.int32)
                    sorted_ak4 = np.sort(ak4_c)
                    um = np.ones(6, dtype=bool)
                    um[1:] = (sorted_ak4[1:] != sorted_ak4[:-1])
                    k4_counts[ak4] = np.sum(um)
                v = int(np.sum(k4_counts < 5))
        
        return colours, v

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

            # Local search on init_colours
            best_colours, best_violations = local_search(init_colours, K)
            if best_violations == 0:
                current = best_colours
                continue

            # Random restarts with local search
            success = False
            max_restarts = 50
            for restart in range(max_restarts):
                if time.time() > deadline:
                    break
                rand_colours = [random.randint(0, K - 1) for _ in range(120)]
                rand_colours, v = local_search(rand_colours, K)
                if v == 0:
                    current = rand_colours
                    success = True
                    break

            if not success:
                break

    if not is_valid(current):
        seed = [(i + j) % 17 for i in range(n) for j in range(i + 1, n)]
        if is_valid(seed):
            current = seed

    return current
