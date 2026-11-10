import numpy as np
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    
    # 1. Compute forbidden differences (nonzero quadratic residues)
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # 2. Build conflict masks
    # conflict_mask[u] has bit v set if (u - v) % m is in forbidden
    # Since the graph is undirected, (u-v) in forbidden iff (v-u) in forbidden?
    # Check: if d is a square, is -d a square?
    # x^2 = d. (-x)^2 = x^2 = d.
    # So if d is a square, -d is a square (with root -x).
    # So the graph is undirected.
    conflict_mask = [0] * m
    for u in range(m):
        mask = 0
        for v in range(m):
            if u == v:
                continue
            d = (u - v) % m
            if d in forbidden:
                mask |= (1 << v)
        conflict_mask[u] = mask

    best_size = 0
    best_mask = 0
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    # Helper to get list of nodes from mask
    def mask_to_list(mask):
        return [i for i in range(m) if (mask >> i) & 1]

    # Helper to get size
    def popcount(mask):
        return bin(mask).count('1')

    # Local search function
    def local_search(S_mask):
        S_size = popcount(S_mask)
        # We allow swaps to escape local optima.
        # To prevent infinite loops, we limit the number of consecutive swaps without an add.
        max_consecutive_swaps = 50
        consecutive_swaps = 0
        
        while True:
            # Check time
            if time.time() > deadline:
                break
            
            improved = False
            
            # 1. Try to add a vertex
            for v in range(m):
                if not (S_mask >> v) & 1:
                    if (conflict_mask[v] & S_mask) == 0:
                        S_mask |= (1 << v)
                        S_size += 1
                        improved = True
                        consecutive_swaps = 0
                        break # Restart add scan
            
            if improved:
                continue
                
            # 2. If no add, try a swap
            if consecutive_swaps >= max_consecutive_swaps:
                break
            
            # Find a valid swap: u in S, v not in S, S' = S - u + v is independent
            # We pick a random u to avoid bias
            S_list = mask_to_list(S_mask)
            if not S_list:
                break
                
            # Try a few random swaps or all?
            # With m=205, all swaps is 205*205 ~ 42k ops. Fast enough.
            # But we want to be smart.
            # Let's try all u in S and for each, find a valid v.
            swap_found = False
            # Shuffle S_list to randomize
            np.random.shuffle(S_list)
            
            for u in S_list:
                S_without_u = S_mask ^ (1 << u)
                # Find v not in S such that conflict_mask[v] & S_without_u == 0
                # v must not be u (it's not, u is in S)
                for v in range(m):
                    if not (S_mask >> v) & 1:
                        if (conflict_mask[v] & S_without_u) == 0:
                            # Valid swap
                            S_mask = S_without_u | (1 << v)
                            # Size remains same
                            swap_found = True
                            consecutive_swaps += 1
                            break
                if swap_found:
                    break
                    
            if not swap_found:
                # No valid swap found, stuck
                break

        return S_mask, S_size

    # Main search loop: Iterated Local Search
    while time.time() < deadline:
        # 1. Generate initial solution via random greedy
        S_mask = 0
        nodes = list(range(m))
        np.random.shuffle(nodes)
        for v in nodes:
            if (conflict_mask[v] & S_mask) == 0:
                S_mask |= (1 << v)
        
        # 2. Local search
        S_mask, S_size = local_search(S_mask)
        
        # 3. Update best
        if S_size > best_size:
            best_size = S_size
            best_mask = S_mask
            
        # 4. Perturbation for next iteration
        # If we found a good solution, we might want to perturb it slightly
        # Or just restart from scratch.
        # Given the small size, restarting from scratch with different random seeds is robust.
        # But re-using the best solution and perturbing it can be faster to find even better ones.
        # Let's mix: 50% restart, 50% perturb best.
        if best_mask and np.random.rand() < 0.5:
            # Perturb best: remove 1-2 random nodes
            best_list = mask_to_list(best_mask)
            if best_list:
                num_remove = min(2, len(best_list))
                to_remove = np.random.choice(best_list, num_remove, replace=False)
                perturbed_mask = best_mask
                for r in to_remove:
                    perturbed_mask ^= (1 << r)
                # Run local search on perturbed
                S_mask, S_size = local_search(perturbed_mask)
                if S_size > best_size:
                    best_size = S_size
                    best_mask = S_mask
        # Else, loop continues to generate new random greedy

    # Convert best_mask to array
    result = mask_to_list(best_mask)
    return np.array(result, dtype=np.int64)
