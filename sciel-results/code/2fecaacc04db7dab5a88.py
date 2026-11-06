import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill

    N = 1000
    MAX_SUM = 2 * N
    K = 2

    all_nums = np.arange(1, N + 1, dtype=np.int64)

    def _can_add_vec(x, chosen_arr, counts):
        """Check if x can be added without violating the B2[2] constraint."""
        if chosen_arr.size == 0:
            return counts[2 * x] < K
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
        if counts[2 * x] >= K:
            return False
        return True

    def _add_vec(x, chosen_arr, counts):
        """Add x to the set and update counts. Returns new array and counts."""
        if chosen_arr.size > 0:
            s = chosen_arr + x
            counts[s] += 1
        counts[2 * x] += 1
        return np.concatenate([chosen_arr, np.array([x], dtype=np.int64)]), counts

    def _remove_vec(x, chosen_arr, counts):
        """Remove x from the set and update counts. Returns new array and counts."""
        mask = chosen_arr != x
        remaining = chosen_arr[mask]
        if remaining.size > 0:
            s = remaining + x
            counts[s] -= 1
        counts[2 * x] -= 1
        return remaining, counts

    def _saturating_greedy(order, counts):
        """Greedy that adds any valid element in the given order."""
        chosen = np.empty(0, dtype=np.int64)
        for x in order:
            if _can_add_vec(int(x), chosen, counts):
                chosen, counts = _add_vec(int(x), chosen, counts)
        return chosen, counts

    def _can_add_list(x, chosen_list, counts):
        """Check if x can be added using a Python list of chosen elements."""
        if not chosen_list:
            return counts[2 * x] < K
        chosen_arr = np.array(chosen_list, dtype=np.int64)
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
        if counts[2 * x] >= K:
            return False
        return True

    def _local_search_v2(chosen, counts, rng, deadline, max_iter=500000):
        """
        Optimized local search using lists to avoid np.concatenate overhead.
        """
        n = len(chosen)
        chosen_list = [int(v) for v in chosen]
        chosen_mask = np.zeros(N + 1, dtype=bool)
        for v in chosen_list:
            chosen_mask[v] = True
            
        best_size = n
        best_chosen = chosen.copy()
        best_counts = counts.copy()
        
        consecutive_fails = 0
        
        for it in range(max_iter):
            if (it & 127) == 0:
                if time.time() > deadline:
                    break
            
            # Try to find any insertable element
            found_insert = False
            # Sample up to 10 random candidates
            for _ in range(10):
                if time.time() > deadline:
                    break
                x_add = rng.randint(1, N)
                if chosen_mask[x_add]:
                    continue
                if _can_add_list(x_add, chosen_list, counts):
                    # Add
                    chosen_list.append(x_add)
                    chosen_arr = np.array(chosen_list, dtype=np.int64)
                    s = chosen_arr + x_add
                    counts[s] += 1
                    counts[2 * x_add] += 1
                    chosen_mask[x_add] = True
                    found_insert = True
                    if len(chosen_list) > best_size:
                        best_size = len(chosen_list)
                        best_chosen = np.array(chosen_list, dtype=np.int64)
                        best_counts = counts.copy()
                    break
            
            if found_insert:
                consecutive_fails = 0
                continue
            
            consecutive_fails += 1
            
            # Adaptive swap threshold
            swap_threshold = max(5, 20 - consecutive_fails // 5)
            
            if consecutive_fails >= swap_threshold:
                # Try swaps: remove one, add another
                swap_success = False
                for _ in range(8):
                    if time.time() > deadline:
                        break
                    if not chosen_list:
                        break
                    
                    rm_idx = rng.randrange(len(chosen_list))
                    x_rm = chosen_list[rm_idx]
                    
                    # Remove x_rm
                    chosen_list.pop(rm_idx)
                    if chosen_list:
                        chosen_arr = np.array(chosen_list, dtype=np.int64)
                        s = chosen_arr + x_rm
                        counts[s] -= 1
                    counts[2 * x_rm] -= 1
                    chosen_mask[x_rm] = False
                    
                    # Try to add a random available element
                    x_try = rng.randint(1, N)
                    if chosen_mask[x_try]:
                        # Restore
                        chosen_list.append(x_rm)
                        if chosen_list:
                            chosen_arr = np.array(chosen_list, dtype=np.int64)
                            s = chosen_arr + x_rm
                            counts[s] += 1
                        counts[2 * x_rm] += 1
                        chosen_mask[x_rm] = True
                        continue
                    
                    if _can_add_list(x_try, chosen_list, counts):
                        # Swap successful
                        chosen_list.append(x_try)
                        chosen_arr = np.array(chosen_list, dtype=np.int64)
                        s = chosen_arr + x_try
                        counts[s] += 1
                        counts[2 * x_try] += 1
                        chosen_mask[x_try] = True
                        swap_success = True
                        consecutive_fails = 0
                        break
                    else:
                        # Restore
                        chosen_list.append(x_rm)
                        if chosen_list:
                            chosen_arr = np.array(chosen_list, dtype=np.int64)
                            s = chosen_arr + x_rm
                            counts[s] += 1
                        counts[2 * x_rm] += 1
                        chosen_mask[x_rm] = True
                
                if not swap_success:
                    pass

        return best_chosen, best_counts

    # --- Phase 1: Multiple Greedy Seeds ---
    best_chosen = np.empty(0, dtype=np.int64)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    best_size = 0

    # Deterministic orders
    orders = [
        all_nums,
        all_nums[::-1],
    ]
    
    # Add modular arithmetic orderings for diversity
    for mod in [2, 3, 4, 5, 7, 11, 13]:
        o = np.array(sorted(all_nums, key=lambda x: x % mod), dtype=np.int64)
        orders.append(o)
    
    # Add prime-based ordering
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for p in primes[:6]:
        o = np.array(sorted(all_nums, key=lambda x: (x // p, x % p)), dtype=np.int64)
        orders.append(o)
    
    # Randomized orders with more seeds
    for seed in (123, 7, 999, 2024, 555, 111, 42, 1337, 4242, 8675, 3141, 2718):
        o = all_nums.copy()
        r = random.Random(seed)
        r.shuffle(o)
        orders.append(o)

    # Time-budgeted greedy: stop early if we're close to deadline
    greedy_deadline = min(deadline, start_time + 30)
    for o in orders:
        if time.time() > greedy_deadline:
            break
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct

    # --- Phase 2: Local Search from Best Seed ---
    rng = random.Random(42)
    # Use improved local search v2
    ls_deadline = min(deadline, start_time + 110)
    ch, ct = _local_search_v2(best_chosen, best_counts, rng, ls_deadline, max_iter=500000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        best_counts = ct

    # --- Phase 3: Try other good seeds with Local Search if time remains ---
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        num_trials = min(6, int(remaining_time // 10))
        for trial in range(num_trials):
            if time.time() > deadline:
                break
            
            # Generate a new seed
            seed = 10000 + trial * 101
            o = all_nums.copy()
            rt = random.Random(seed)
            rt.shuffle(o)
            
            ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
            
            # Only proceed if this seed is competitive
            if len(ch2) < best_size - 2:
                continue
                
            # Run local search with a tighter time budget for this trial
            current_deadline = min(deadline, time.time() + max(5, int(remaining_time / (num_trials - trial))))
            if current_deadline <= time.time():
                break
                
            ch2, ct2 = _local_search_v2(ch2, ct2, rng, current_deadline, max_iter=500000)
            
            if len(ch2) > best_size:
                best_size = len(ch2)
                best_chosen = ch2
                best_counts = ct2

    # Final verification and cleanup
    # Verify the solution is valid
    final_set = set(int(v) for v in best_chosen)
    assert len(final_set) == len(best_chosen), "Duplicate elements found"
    assert all(1 <= v <= N for v in final_set), "Out of range elements found"
    
    # Verify B2[2] constraint
    count_arr = np.zeros(MAX_SUM + 1, dtype=np.int64)
    sorted_chosen = sorted(final_set)
    for i, a in enumerate(sorted_chosen):
        for j in range(i, len(sorted_chosen)):
            b = sorted_chosen[j]
            s = a + b
            count_arr[s] += 1
    assert np.all(count_arr <= K), "B2[2] constraint violated"
    
    return [int(v) for v in best_chosen]

# EVOLVE-BLOCK-END
