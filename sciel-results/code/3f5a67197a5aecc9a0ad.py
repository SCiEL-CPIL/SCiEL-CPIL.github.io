# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    N = 256
    candidates = list(range(1, N + 1))
    n_cand = N
    
    # Precompute conflicts as bitsets (Python ints) for fast operations
    # conflicts[i] is a bitmask of j such that gcd(candidates[i], candidates[j]) > 1
    conflicts = [0] * n_cand
    for i in range(n_cand):
        for j in range(i + 1, n_cand):
            if gcd(candidates[i], candidates[j]) > 1:
                conflicts[i] |= (1 << j)
                conflicts[j] |= (1 << i)
    
    # Sort indices by value descending for B&B ordering
    order = sorted(range(n_cand), key=lambda x: candidates[x], reverse=True)
    
    # Greedy Initialization
    def greedy_init():
        selected = set()
        blocked = 0
        for idx in order:
            if not (blocked >> idx) & 1:
                selected.add(idx)
                blocked |= (1 << idx)
                blocked |= conflicts[idx]
        return selected

    best_set = greedy_init()
    best_sum = sum(candidates[i] for i in best_set)
    
    # Local Search with 1-for-1 swaps
    def local_search(initial_set, max_iter=200):
        current_set = set(initial_set)
        current_sum = sum(candidates[i] for i in current_set)
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            # Try adding a single item
            for idx in range(n_cand):
                if idx in current_set:
                    continue
                # Check if idx is compatible with all in current_set
                # Using bitsets: (conflicts[idx] & current_mask) == 0
                current_mask = 0
                for i in current_set:
                    current_mask |= (1 << i)
                if (conflicts[idx] & current_mask) == 0:
                    current_set.add(idx)
                    current_sum += candidates[idx]
                    improved = True
            # Try 1-for-1 swaps: remove one, add one
            for idx_out in list(current_set):
                # Compute mask without idx_out
                mask_without_out = 0
                for i in current_set:
                    if i != idx_out:
                        mask_without_out |= (1 << i)
                for idx_in in range(n_cand):
                    if idx_in in current_set or idx_in == idx_out:
                        continue
                    # Check if idx_in is compatible with current_set - {idx_out}
                    if (conflicts[idx_in] & mask_without_out) == 0:
                        if candidates[idx_in] > candidates[idx_out]:
                            current_set.remove(idx_out)
                            current_set.add(idx_in)
                            current_sum += candidates[idx_in] - candidates[idx_out]
                            improved = True
        return current_set, current_sum

    best_set, best_sum = local_search(best_set)
    
    # Branch and Bound with bitsets
    # State: available_mask, current_sum, selected_set
    def dfs(available_mask, current_sum, selected):
        nonlocal best_sum, best_set
        
        if time.time() > deadline:
            return
        
        # Bound: sum of all values in available_mask
        # Precompute value sums for bitmasks? Too big.
        # Calculate on the fly but optimize
        avail_sum = 0
        mask = available_mask
        while mask:
            # Get lowest set bit
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            avail_sum += candidates[idx]
            mask ^= lsb
        
        if current_sum + avail_sum <= best_sum:
            return
        
        if available_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected)
            return
        
        # Find the next candidate in 'order' that is in available_mask
        next_idx = -1
        for idx in order:
            if (available_mask >> idx) & 1:
                next_idx = idx
                break
        
        if next_idx == -1:
            return

        # Option 1: Exclude next_idx
        new_avail_excl = available_mask & ~(1 << next_idx)
        dfs(new_avail_excl, current_sum, selected)
        
        # Option 2: Include next_idx
        if time.time() > deadline:
            return
        
        new_avail_incl = new_avail_excl & ~conflicts[next_idx]
        selected.append(next_idx)
        dfs(new_avail_incl, current_sum + candidates[next_idx], selected)
        selected.pop()
    
    # Initial call
    initial_mask = (1 << n_cand) - 1
    dfs(initial_mask, 0, [])
    
    # Convert indices back to values
    result = [candidates[i] for i in best_set]
    
    # Ensure it's a list of ints
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
