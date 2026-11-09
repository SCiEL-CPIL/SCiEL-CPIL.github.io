"""Dense 4-cycle rainbow edge-colouring on 20 vertices."""
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130.0
    n = 20
    num_colors = 20
    
    # Bipartite split
    u_nodes = list(range(10))
    v_nodes = list(range(10, 20))
    
    # Precompute bipartite 4-cycles in the 10x10 matrix
    # Cycle: (i1,j1), (i2,j1), (i2,j2), (i1,j2) for i1<i2, j1<j2
    cycle_list = []
    for i1 in range(10):
        for i2 in range(i1 + 1, 10):
            for j1 in range(10):
                for j2 in range(j1 + 1, 10):
                    cycle_list.append(((i1, j1), (i2, j1), (i2, j2), (i1, j2)))
    
    num_cycles = len(cycle_list)
    
    # Array of cycle edge positions: shape (num_cycles, 4, 2)
    cycle_edges_arr = np.zeros((num_cycles, 4, 2), dtype=np.int32)
    for idx, cyc in enumerate(cycle_list):
        for k, (r, c) in enumerate(cyc):
            cycle_edges_arr[idx, k, 0] = r
            cycle_edges_arr[idx, k, 1] = c
    
    # For each cell, which cycles contain it
    from collections import defaultdict
    cell_to_cycles = defaultdict(list)
    for idx, cyc in enumerate(cycle_list):
        for (r, c) in cyc:
            cell_to_cycles[(r, c)].append(idx)
    
    cell_to_cycles_arr = {}
    for (r, c), idxs in cell_to_cycles.items():
        cell_to_cycles_arr[(r, c)] = np.array(idxs)
    
    def count_bad_cycles(M):
        """Count non-rainbow 4-cycles where all edges present. M: 10x10, -1 = absent."""
        r1 = cycle_edges_arr[:, 0, 0]
        c1 = cycle_edges_arr[:, 0, 1]
        r2 = cycle_edges_arr[:, 1, 0]
        c2 = cycle_edges_arr[:, 1, 1]
        r3 = cycle_edges_arr[:, 2, 0]
        c3 = cycle_edges_arr[:, 2, 1]
        r4 = cycle_edges_arr[:, 3, 0]
        c4 = cycle_edges_arr[:, 3, 1]
        
        cols = np.column_stack((M[r1, c1], M[r2, c2], M[r3, c3], M[r4, c4]))
        all_present = (cols != -1).all(axis=1)
        
        c0, c1_, c2_, c3_ = cols[:, 0], cols[:, 1], cols[:, 2], cols[:, 3]
        distinct = (c0 != c1_) & (c0 != c2_) & (c0 != c3_) & \
                   (c1_ != c2_) & (c1_ != c3_) & (c2_ != c3_)
        
        bad_mask = all_present & ~distinct
        return np.sum(bad_mask), bad_mask
    
    def get_edge_count(M):
        return int(np.sum(M != -1))
    
    best_M = None
    best_edge_count = 0
    best_bad_count = float('inf')
    
    random.seed(42)
    
    # Phase 1: Greedy construction with multiple random restarts
    num_attempts = 25
    for attempt in range(num_attempts):
        if time.time() > deadline - 10.0:
            break
        
        M = np.full((10, 10), -1, dtype=np.int32)
        
        cells = [(i, j) for i in range(10) for j in range(10)]
        random.shuffle(cells)
        
        current_bad, _ = count_bad_cycles(M)
        
        for (i, j) in cells:
            if time.time() > deadline - 10.0:
                break
            
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            added = False
            for color in colors:
                M[i, j] = color
                new_bad, _ = count_bad_cycles(M)
                if new_bad == 0:
                    added = True
                    current_bad = 0
                    break
                else:
                    M[i, j] = -1
            
            if not added:
                pass
        
        edge_count = get_edge_count(M)
        bad_count, _ = count_bad_cycles(M)
        
        if bad_count == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_M = M.copy()
            best_bad_count = 0
    
    # Phase 2: Local search to add more edges
    if best_M is not None:
        M = best_M.copy()
        current_bad, _ = count_bad_cycles(M)
        current_edges = get_edge_count(M)
        
        missing_cells = [(i, j) for i in range(10) for j in range(10) if M[i, j] == -1]
        random.shuffle(missing_cells)
        
        for (i, j) in missing_cells:
            if time.time() > deadline - 5.0:
                break
            
            colors = list(range(num_colors))
            random.shuffle(colors)
            
            for color in colors:
                M[i, j] = color
                new_bad, _ = count_bad_cycles(M)
                if new_bad == 0:
                    current_edges += 1
                    break
                else:
                    M[i, j] = -1
        
        if current_edges > best_edge_count:
            best_edge_count = current_edges
            best_M = M.copy()
    
    # Phase 3: If we have fewer than 50 edges, try random full coloring + repair
    if best_edge_count < 50:
        for attempt in range(15):
            if time.time() > deadline - 5.0:
                break
            
            M = np.random.randint(0, num_colors, size=(10, 10), dtype=np.int32)
            current_bad, current_bad_mask = count_bad_cycles(M)
            local_best = M.copy()
            local_best_edges = get_edge_count(M)
            local_best_bad = current_bad
            
            for step in range(3000):
                if time.time() > deadline - 5.0:
                    break
                
                if current_bad == 0:
                    break
                
                bad_indices = np.where(current_bad_mask)[0]
                if len(bad_indices) == 0:
                    break
                target_idx = bad_indices[random.randint(0, len(bad_indices) - 1)]
                
                r1 = cycle_edges_arr[target_idx, 0, 0]
                c1 = cycle_edges_arr[target_idx, 0, 1]
                r2 = cycle_edges_arr[target_idx, 1, 0]
                c2 = cycle_edges_arr[target_idx, 1, 1]
                r3 = cycle_edges_arr[target_idx, 2, 0]
                c3 = cycle_edges_arr[target_idx, 2, 1]
                r4 = cycle_edges_arr[target_idx, 3, 0]
                c4 = cycle_edges_arr[target_idx, 3, 1]
                
                edges_in_cycle = [(r1, c1), (r2, c2), (r3, c3), (r4, c4)]
                r, c = edges_in_cycle[random.randint(0, 3)]
                old_color = M[r, c]
                
                improved = False
                for _ in range(10):
                    new_color = random.randint(0, num_colors - 1)
                    if new_color == old_color:
                        continue
                    M[r, c] = new_color
                    new_bad, new_bad_mask = count_bad_cycles(M)
                    
                    if new_bad < current_bad:
                        current_bad = new_bad
                        current_bad_mask = new_bad_mask
                        if current_bad < local_best_bad:
                            local_best_bad = current_bad
                            local_best = M.copy()
                            local_best_edges = get_edge_count(M)
                        improved = True
                        break
                    else:
                        M[r, c] = old_color
                
                if not improved:
                    rr = random.randint(0, 9)
                    cc = random.randint(0, 9)
                    M[rr, cc] = random.randint(0, num_colors - 1)
                    current_bad, current_bad_mask = count_bad_cycles(M)
                    if current_bad < local_best_bad:
                        local_best_bad = current_bad
                        local_best = M.copy()
                        local_best_edges = get_edge_count(M)
            
            final_bad, _ = count_bad_cycles(local_best)
            if final_bad == 0:
                edges = get_edge_count(local_best)
                if edges > best_edge_count:
                    best_edge_count = edges
                    best_M = local_best.copy()
    
    # Phase 4: Fallback sparse construction
    if best_M is None or best_edge_count < 30:
        M = np.full((10, 10), -1, dtype=np.int32)
        
        for i in range(10):
            for j in range(10):
                if time.time() > deadline - 2.0:
                    break
                colors = list(range(num_colors))
                random.shuffle(colors)
                for color in colors:
                    M[i, j] = color
                    new_bad, _ = count_bad_cycles(M)
                    if new_bad == 0:
                        break
                    else:
                        M[i, j] = -1
        
        edge_count = get_edge_count(M)
        bad_count, _ = count_bad_cycles(M)
        if bad_count == 0 and edge_count > best_edge_count:
            best_edge_count = edge_count
            best_M = M.copy()
    
    # Fallback: star graph
    if best_M is None:
        result = []
        for v in range(1, n):
            result.append([0, v, v % num_colors])
        return result
    
    # Convert to edge list
    result = []
    for i in range(10):
        for j in range(10):
            if best_M[i, j] != -1:
                u = u_nodes[i]
                v = v_nodes[j]
                c = int(best_M[i, j])
                result.append([u, v, c])
    
    return result

# EVOLVE-BLOCK-END
