import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # 1. Compute Forbidden Differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build adjacency lists and bitmasks
    adj_mask = [0] * m
    adj_list = [[] for _ in range(m)]
    for i in range(m):
        for j in range(i + 1, m):
            d = (i - j) % m
            if d in forbidden:
                adj_mask[i] |= (1 << j)
                adj_mask[j] |= (1 << i)
                adj_list[i].append(j)
                adj_list[j].append(i)
                
    full_mask = (1 << m) - 1
    
    def get_size(mask):
        return bin(mask).count("1")
        
    def greedy_init(shuffle_seed, strategy="random"):
        rng = random.Random(shuffle_seed)
        candidates = list(range(m))
        rng.shuffle(candidates)
        selected_mask = 0
        if strategy == "random":
            for v in candidates:
                if (adj_mask[v] & selected_mask) == 0:
                    selected_mask |= (1 << v)
        else:  # min_degree
            remaining = set(candidates)
            while remaining:
                best_v = None
                best_count = float('inf')
                for v in remaining:
                    count = bin(adj_mask[v] & selected_mask).count("1")
                    if count < best_count:
                        best_count = count
                        best_v = v
                    if count == 0:
                        break
                if best_v is not None and best_count == 0:
                    selected_mask |= (1 << best_v)
                    remaining.discard(best_v)
                else:
                    break
        return selected_mask

    def local_improve(mask, rng, use_swaps=True):
        # 1. Greedy Add
        changed = True
        while changed:
            changed = False
            addable = []
            not_selected = full_mask ^ mask
            for v in range(m):
                if (not_selected >> v) & 1:
                    if (adj_mask[v] & mask) == 0:
                        addable.append(v)
                        
            if not addable:
                break
                
            # Pick the one with fewest neighbors in remaining (min degree heuristic)
            if len(addable) > 1:
                best_v = None
                best_deg = float('inf')
                for v in addable:
                    deg = bin(adj_mask[v] & not_selected).count("1")
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                v = best_v
            else:
                v = addable[0]
            mask |= (1 << v)
            changed = True
            
        # 2. Swap-based improvement if enabled
        if use_swaps:
            improved = True
            while improved:
                improved = False
                selected = [v for v in range(m) if (mask >> v) & 1]
                not_sel = [v for v in range(m) if not (mask >> v) & 1]
                
                for u in selected:
                    mask_without_u = mask & ~(1 << u)
                    addable = []
                    for v in not_sel:
                        if (adj_mask[v] & mask_without_u) == 0:
                            addable.append(v)
                    
                    if len(addable) >= 2:
                        new_mask = mask_without_u
                        for v in addable:
                            if (adj_mask[v] & new_mask) == 0:
                                new_mask |= (1 << v)
                        if get_size(new_mask) > get_size(mask):
                            mask = new_mask
                            improved = True
                            break
                    elif len(addable) == 1:
                        v = addable[0]
                        new_mask = mask_without_u | (1 << v)
                        further_addable = []
                        for w in not_sel:
                            if w != v and (adj_mask[w] & new_mask) == 0:
                                further_addable.append(w)
                        if further_addable:
                            mask = new_mask
                            improved = True
                            break
                            
        return mask

    def perturb(mask, rng, k_remove):
        vertices = [v for v in range(m) if (mask >> v) & 1]
        if len(vertices) < k_remove:
            return 0
        to_remove = rng.sample(vertices, k_remove)
        new_mask = mask
        for v in to_remove:
            new_mask &= ~(1 << v)
        return new_mask

    start_time = time.time()
    max_time = 620  # Safety margin before 630s kill
    
    best_mask = 0
    best_size = 0
    
    # Initial Greedy Runs with different strategies
    for i in range(100):
        strategy = "random" if i % 2 == 0 else "min_degree"
        m_init = greedy_init(i, strategy)
        m_init = local_improve(m_init, random.Random(i), use_swaps=True)
        sz = get_size(m_init)
        if sz > best_size:
            best_size = sz
            best_mask = m_init
        if time.time() - start_time > 30:
            break

    # ILS Loop with multiple perturbation strategies
    rng = random.Random(42)
    curr_mask = best_mask
    curr_size = best_size
    temp = 50.0
    min_temp = 1.0
    cooling = 0.995
    
    iteration = 0
    while time.time() - start_time < max_time:
        iteration += 1
        
        if iteration % 10 == 0:
            k_remove = rng.randint(8, 15)
        else:
            k_remove = rng.randint(2, 8)
        
        pert_mask = perturb(curr_mask, rng, k_remove)
        new_mask = local_improve(pert_mask, rng, use_swaps=True)
        new_size = get_size(new_mask)
        
        if new_size >= curr_size:
            curr_mask = new_mask
            curr_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
        else:
            delta = new_size - curr_size
            if rng.random() < pow(2.71828, delta / temp):
                curr_mask = new_mask
                curr_size = new_size
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        
        if iteration % 50 == 0 and curr_size < best_size:
            curr_mask = best_mask
            curr_size = best_size
            temp = 30.0
    
    # Final Polish
    final_mask = best_mask
    for _ in range(10):
        if time.time() - start_time > max_time - 10:
            break
        polished = local_improve(final_mask, rng, use_swaps=True)
        if get_size(polished) > get_size(final_mask):
            final_mask = polished
    
    result = []
    for v in range(m):
        if (final_mask >> v) & 1:
            result.append(v)
            
    return np.array(result, dtype=np.int64)
