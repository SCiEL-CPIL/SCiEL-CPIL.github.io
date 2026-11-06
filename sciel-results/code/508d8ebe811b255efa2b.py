import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    # Fallback seed for n=9 (Valid 4-coloring)
    # The seed in the prompt was: [min(j - i, n_seed - j + i) - 1 ...]
    # Let's verify or use a known valid small one.
    # Actually, the prompt says "Keep the function names...".
    # I will use the seed from the prompt as fallback.
    n_seed = 9
    colors_seed = [min(j - i, n_seed - j + i) - 1 for i in range(n_seed) for j in range(i + 1, n_seed)]
    best_result = {'n': n_seed, 'colors': colors_seed}
    best_n = n_seed
    
    # Try n=17, 16, 15
    # We allocate most time to 17.
    target_ns = [17, 16, 15]
    
    # Time allocation
    # 17: 90s, 16: 20s, 15: 5s
    time_alloc = {
        17: 90,
        16: 20,
        15: 5
    }
    
    for n in target_ns:
        if time.time() > deadline - 2:
            break
            
        t_limit = time.time() + time_alloc.get(n, 5)
        if t_limit > deadline - 1:
            t_limit = deadline - 1
            
        solution = local_search_numpy(n, t_limit)
        
        if solution is not None:
            return {'n': n, 'colors': solution}
            
    return best_result

def local_search_numpy(n, deadline):
    """
    Finds a valid 4-coloring of K_n with no monochromatic C6.
    Returns list of colors or None.
    """
    num_edges = n * (n - 1) // 2
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    edge_index = {e: idx for idx, e in enumerate(edges)}
    
    # Precompute cycles
    # For each edge, store the other 5 edges in every C6 containing it.
    # Structure: list of numpy arrays. edge_others[i] is array of shape (num_cycles_i, 5)
    edge_others = [[] for _ in range(num_edges)]
    
    # Generate all C6 cycles
    # To speed up, we can iterate combinations of 6 vertices
    vertices = list(range(n))
    
    # Optimization: If n is large, this might be slow.
    # For n=17, ~12k combos. Fast.
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        # Generate 60 unique cycles
        # Fix v0. Permute v1..v5. Keep perm[0] < perm[-1].
        perms = list(itertools.permutations([v1, v2, v3, v4, v5]))
        for perm in perms:
            if perm[0] > perm[-1]:
                continue
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_indices.append(idx)
            
            # Update edge_others
            for i in range(6):
                e_idx = edge_indices[i]
                others = [edge_indices[j] for j in range(6) if j != i]
                edge_others[e_idx].append(others)
                
    if time.time() > deadline:
        return None
        
    # Convert to numpy arrays for fast delta calculation
    # We need to handle variable lengths.
    # We can just keep them as lists of lists?
    # np.all on a list of lists is slower than on a 2D array.
    # Let's pad them? No, just convert each to a 2D array.
    edge_others_np = []
    for e in range(num_edges):
        if edge_others[e]:
            arr = np.array(edge_others[e], dtype=np.int32)
            edge_others_np.append(arr)
        else:
            edge_others_np.append(np.empty((0, 5), dtype=np.int32))
            
    # Initialize colors
    colors = np.random.randint(0, 4, size=num_edges, dtype=np.int32)
    
    # Calculate initial cost
    # We need to count mono cycles.
    # We can do this by iterating all cycles again? Or using the precomputed data?
    # Easier: Just iterate all cycles generated? We didn't store them globally.
    # Let's store all_cycles in the precomputation step.
    
    # Re-do precomputation to store all_cycles?
    # Or just calculate cost by checking all possible C6s?
    # We can reuse the generation loop.
    
    all_cycles = []
    # We'll just regenerate or store in first pass.
    # Let's modify the first pass to store all_cycles.
    
    # To avoid code duplication, I'll just re-run the generation if needed, 
    # but it's better to do it in one pass.
    
    # Let's restart precomputation to store both.
    edge_others = [[] for _ in range(num_edges)]
    all_cycles = []
    
    for comb in itertools.combinations(vertices, 6):
        v0, v1, v2, v3, v4, v5 = comb
        perms = list(itertools.permutations([v1, v2, v3, v4, v5]))
        for perm in perms:
            if perm[0] > perm[-1]:
                continue
            cycle_verts = [v0] + list(perm)
            edge_indices = []
            for k in range(6):
                u = cycle_verts[k]
                v = cycle_verts[(k + 1) % 6]
                if u > v:
                    u, v = v, u
                idx = edge_index[(u, v)]
                edge_indices.append(idx)
            
            all_cycles.append(edge_indices)
            for i in range(6):
                e_idx = edge_indices[i]
                others = [edge_indices[j] for j in range(6) if j != i]
                edge_others[e_idx].append(others)
                
    if time.time() > deadline:
        return None
        
    edge_others_np = []
    for e in range(num_edges):
        if edge_others[e]:
            arr = np.array(edge_others[e], dtype=np.int32)
            edge_others_np.append(arr)
        else:
            edge_others_np.append(np.empty((0, 5), dtype=np.int32))
            
    def count_mono_cycles(col):
        count = 0
        for cycle in all_cycles:
            c0 = col[cycle[0]]
            if col[cycle[1]] == c0 and col[cycle[2]] == c0 and col[cycle[3]] == c0 and col[cycle[4]] == c0 and col[cycle[5]] == c0:
                count += 1
        return count
        
    current_cost = count_mono_cycles(colors)
    if current_cost == 0:
        return colors.tolist()
        
    best_colors = colors.copy()
    best_cost = current_cost
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.9999
    
    while time.time() < deadline - 0.5:
        e_idx = random.randint(0, num_edges - 1)
        c_old = colors[e_idx]
        c_new = random.randint(0, 3)
        if c_new == c_old:
            continue
            
        # Calculate delta using numpy
        others = edge_others_np[e_idx]
        if len(others) > 0:
            cols = colors[others]
            all_old = np.all(cols == c_old, axis=1)
            all_new = np.all(cols == c_new, axis=1)
            delta = np.sum(all_new) - np.sum(all_old)
        else:
            delta = 0
            
        new_cost = current_cost + delta
        
        if new_cost < current_cost:
            colors[e_idx] = c_new
            current_cost = new_cost
            
            if current_cost == 0:
                return colors.tolist()
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
        else:
            if delta > 0:
                import math
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    colors[e_idx] = c_new
                    current_cost = new_cost
            else:
                # delta == 0
                if random.random() < 0.5:
                    colors[e_idx] = c_new
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return None
