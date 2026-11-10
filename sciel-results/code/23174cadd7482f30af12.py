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
    
    # Precompute static degrees
    static_degrees = [bin(adj_mask[i]).count("1") for i in range(m)]
    
    def get_size(mask):
        return bin(mask).count("1")
        
    def greedy_init(shuffle_seed, strategy="random"):
        rng = random.Random(shuffle_seed)
        candidates = list(range(m))
        
        if strategy == "random":
            rng.shuffle(candidates)
        elif strategy == "min_degree":
            # Sort by degree, with random tie-breaking
            scored = [(static_degrees[v], rng.random(), v) for v in candidates]
            scored.sort()
            candidates = [v for _, _, v in scored]
        else:  # weighted random
            # Lower degree -> higher priority
            weights = [1.0 / (static_degrees[v] + 1) for v in candidates]
            # Sample without replacement based on weights
            selected = []
            pool = list(range(m))
            while pool:
                total_w = sum(weights[i] for i in pool)
                r = rng.random() * total_w
                cum = 0
                idx = 0
                for i, v in enumerate(pool):
                    cum += weights[v]
                    if r <= cum:
                        idx = i
                        break
                selected.append(pool[idx])
                pool.pop(idx)
            candidates = selected

        selected_mask = 0
        for v in candidates:
            if (adj_mask[v] & selected_mask) == 0:
                selected_mask |= (1 << v)
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
            # This is a strong heuristic for independent sets
            if len(addable) > 1:
                # Score by dynamic degree in the remaining graph
                # But computing dynamic degree is O(n^2). 
                # Approximation: use static degree or count conflicts with *unselected*?
                # Actually, "fewest neighbors in remaining" means we want to add a vertex 
                # that doesn't block too many other potential additions.
                # A vertex v with low degree in the *induced subgraph of unselected* is better?
                # No, if we add v, we remove its neighbors from unselected.
                # So we want v that has few neighbors in (unselected \ {v}).
                # Let's compute this cost.
                best_v = None
                best_cost = float('inf')
                for v in addable:
                    # Count neighbors of v that are in not_selected
                    cost = bin(adj_mask[v] & not_selected).count("1")
                    if cost < best_cost:
                        best_cost = cost
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
                
                # Try 1-for-1 swaps
                # For each u in selected, see if removing it allows adding 2+ vertices
                # This is expensive, so we limit or optimize.
                # Heuristic: Only try swaps for vertices with high "regret" or just random sample?
                # For m=205, iterating all selected is fine if the inner check is fast.
                
                for u in selected:
                    mask_without_u = mask & ~(1 << u)
                    addable = []
                    for v in not_sel:
                        if (adj_mask[v] & mask_without_u) == 0:
                            addable.append(v)
                    
                    if len(addable) >= 2:
                        # Try to add as many as possible from addable
                        new_mask = mask_without_u
                        for v in addable:
                            if (adj_mask[v] & new_mask) == 0:
                                new_mask |= (1 << v)
                        if get_size(new_mask) > get_size(mask):
                            mask = new_mask
                            improved = True
                            break
                    elif len(addable) == 1:
                        # Sometimes a 1-for-1 swap helps unlock a chain of additions
                        v = addable[0]
                        new_mask = mask_without_u | (1 << v)
                        # Check if this allows further additions
                        further_addable = []
                        for w in not_sel:
                            if w != v and (adj_mask[w] & new_mask) == 0:
                                further_addable.append(w)
                        if further_addable:
                            # Greedily add further
                            for w in further_addable:
                                if (adj_mask[w] & new_mask) == 0:
                                    new_mask |= (1 << w)
                            if get_size(new_mask) > get_size(mask):
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
    max_time = 600  # Safety margin before 630s kill
    
    best_mask = 0
    best_size = 0
    
    # Initial Greedy Runs with different strategies
    # Run a diverse set of initializations
    strategies = ["random", "min_degree", "weighted"]
    for i in range(150):
        strategy = strategies[i % 3]
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
        
        # Adaptive perturbation strength
        # Larger perturbations when stuck, smaller when improving
        if iteration % 20 == 0:
            k_remove = rng.randint(8, 15)
        elif iteration % 5 == 0:
            k_remove = rng.randint(4, 8)
        else:
            k_remove = rng.randint(2, 5)
        
        pert_mask = perturb(curr_mask, rng, k_remove)
        new_mask = local_improve(pert_mask, rng, use_swaps=True)
        new_size = get_size(new_mask)
        
        if new_size >= curr_size:
            curr_mask = new_mask
            curr_size = new_size
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
                # Reset temperature on new best? Or just let it cool.
                # Resetting can help escape local minima around the new best.
                # temp = 30.0 
        else:
            delta = new_size - curr_size
            if rng.random() < pow(2.71828, delta / temp):
                curr_mask = new_mask
                curr_size = new_size
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp
        
        # Restart mechanism: if stuck for a while, reset to best
        if iteration % 100 == 0 and curr_size < best_size:
            curr_mask = best_mask
            curr_size = best_size
            temp = 40.0

    # Final Polish
    final_mask = best_mask
    # Try a few more local improvements from the best mask
    for _ in range(20):
        if time.time() - start_time > max_time - 5:
            break
        polished = local_improve(final_mask, rng, use_swaps=True)
        if get_size(polished) > get_size(final_mask):
            final_mask = polished
    
    # One last check: try adding any single vertex
    for v in range(m):
        if (final_mask >> v) & 1:
            continue
        if (adj_mask[v] & final_mask) == 0:
            final_mask |= (1 << v)

    result = []
    for v in range(m):
        if (final_mask >> v) & 1:
            result.append(v)
            
    return np.array(result, dtype=np.int64)
