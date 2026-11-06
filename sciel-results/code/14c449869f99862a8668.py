import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85
    N = 512
    
    # Precompute AP counts for sorting
    ap_counts = [0] * (N + 1)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # AP: a, a+d, a+2d, a+3d
            for x in (a, a+d, a+2*d, a+3*d):
                ap_counts[x] += 1
                
    # Elements with fewer APs are "safer"
    # Sort indices 1..N by ap_counts
    sorted_elements = sorted(range(1, N + 1), key=lambda x: ap_counts[x])
    
    def is_valid_add(mask, x):
        idx = x - 1
        # Case 1: x is 4th
        for d in range(1, (x - 1) // 3 + 1):
            a1, a2, a3 = x - 3*d, x - 2*d, x - d
            if a1 >= 1:
                if (mask >> (a1 - 1)) & 1 and (mask >> (a2 - 1)) & 1 and (mask >> (a3 - 1)) & 1:
                    return False
            else:
                break
        # Case 2: x is 3rd
        for d in range(1, (x - 1) // 2 + 1):
            a1, a2, a3 = x - 2*d, x - d, x + d
            if a1 >= 1 and a3 <= N:
                if (mask >> (a1 - 1)) & 1 and (mask >> (a2 - 1)) & 1 and (mask >> (a3 - 1)) & 1:
                    return False
            elif a1 < 1:
                break
        # Case 3: x is 2nd
        for d in range(1, (N - x) // 2 + 1):
            a1, a2, a3 = x - d, x + d, x + 2*d
            if a1 >= 1 and a3 <= N:
                if (mask >> (a1 - 1)) & 1 and (mask >> (a2 - 1)) & 1 and (mask >> (a3 - 1)) & 1:
                    return False
            elif a3 > N:
                break
        # Case 4: x is 1st
        for d in range(1, (N - x) // 3 + 1):
            a1, a2, a3 = x + d, x + 2*d, x + 3*d
            if a3 <= N:
                if (mask >> (a1 - 1)) & 1 and (mask >> (a2 - 1)) & 1 and (mask >> (a3 - 1)) & 1:
                    return False
            else:
                break
        return True

    def greedy_randomize(order):
        mask = 0
        for x in order:
            if is_valid_add(mask, x):
                mask |= (1 << (x - 1))
        return mask

    def local_search_add(mask):
        improved = True
        while improved:
            improved = False
            for x in range(1, N + 1):
                if not ((mask >> (x - 1)) & 1):
                    if is_valid_add(mask, x):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
        return mask

    def local_search_swap(mask):
        current_mask = mask
        in_set = [i for i in range(1, N+1) if (current_mask >> (i-1)) & 1]
        not_in_set = [i for i in range(1, N+1) if not ((current_mask >> (i-1)) & 1)]
        
        # Limit swaps to avoid timeout
        swaps_tried = 0
        max_swaps = 200
        
        for y in in_set:
            if time.time() > deadline:
                return current_mask
            if swaps_tried > max_swaps:
                return current_mask
            for x in not_in_set:
                if time.time() > deadline:
                    return current_mask
                if swaps_tried > max_swaps:
                    return current_mask
                
                mask_wo_y = current_mask & ~(1 << (y - 1))
                if is_valid_add(mask_wo_y, x):
                    new_mask = mask_wo_y | (1 << (x - 1))
                    # Check if this swap enables an addition
                    # Try to add any z to new_mask
                    for z in range(1, N+1):
                        if not ((new_mask >> (z - 1)) & 1):
                            if is_valid_add(new_mask, z):
                                return new_mask | (1 << (z - 1))
                    # If no addition, we could still accept the swap to escape local opt,
                    # but it risks getting stuck. Let's only accept if it leads to immediate gain.
                    # However, sometimes a swap is needed to enable a gain later.
                    # Given the tight budget, we'll stick to immediate gain or random restarts.
                swaps_tried += 1
        return current_mask

    best_mask = 0
    best_size = 0
    
    # Run multiple greedy instances with different orderings
    while time.time() < deadline - 15:
        # Create a perturbed order
        order = sorted_elements[:]
        # Randomly shuffle a portion to create diversity
        for i in range(len(order)):
            if random.random() < 0.2:
                j = random.randint(0, len(order)-1)
                order[i], order[j] = order[j], order[i]
        
        mask = greedy_randomize(order)
        mask = local_search_add(mask)
        
        size = bin(mask).count('1')
        if size > best_size:
            best_size = size
            best_mask = mask
            # Try local search swap on the new best
            if time.time() < deadline - 5:
                improved_mask = local_search_swap(best_mask)
                improved_size = bin(improved_mask).count('1')
                if improved_size > best_size:
                    best_size = improved_size
                    best_mask = improved_mask
                    best_mask = local_search_add(best_mask)
                    best_size = bin(best_mask).count('1')

    # Final polish
    if time.time() < deadline:
        best_mask = local_search_add(best_mask)
        best_size = bin(best_mask).count('1')
        
        # Try a few more swaps
        for _ in range(10):
            if time.time() > deadline:
                break
            new_mask = local_search_swap(best_mask)
            new_size = bin(new_mask).count('1')
            if new_size > best_size:
                best_size = new_size
                best_mask = new_mask
                best_mask = local_search_add(best_mask)
                best_size = bin(best_mask).count('1')

    result = [i + 1 for i in range(N) if (best_mask >> i) & 1]
    return sorted(result)
