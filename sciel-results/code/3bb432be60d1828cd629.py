import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N          # 2000
K = 2                    # max representations allowed


def _sum_counts_vec(chosen):
    """Vectorized count of representations a<=b for each sum."""
    a = np.asarray(chosen, dtype=np.int64)
    if a.size == 0:
        return np.zeros(MAX_SUM + 1, dtype=np.int64)
    iu, ju = np.triu_indices(a.size)
    sums = a[iu] + a[ju]
    counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    np.add.at(counts, sums, 1)
    return counts


def _validity_vec(counts):
    """max count and the set of 'full' (count==K) sums."""
    mx = int(counts.max())
    full = np.flatnonzero(counts >= K)
    return mx, full


def _can_add_vec(x, chosen_arr, counts):
    """True if adding x keeps every sum <= K. Vectorized."""
    if chosen_arr.size > 0:
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
    s2 = 2 * x
    if counts[s2] >= K:
        return False
    return True


def _add_vec(x, chosen_arr, counts):
    """Return new chosen array (appended) and updated counts after adding x."""
    if chosen_arr.size > 0:
        s = chosen_arr + x
        counts[s] += 1
    counts[2 * x] += 1
    return np.concatenate([chosen_arr, np.array([x], dtype=np.int64)]), counts


def _remove_vec(x, chosen_arr, counts):
    """Return new chosen array and counts after removing x (assumed present)."""
    mask = chosen_arr != x
    if chosen_arr.size > 1:
        s = chosen_arr[mask] + x
        counts[s] -= 1
    counts[2 * x] -= 1
    return chosen_arr[mask], counts


def _saturating_greedy(order, counts):
    """Greedy that adds ANY valid element (order decides which first)."""
    chosen = np.empty(0, dtype=np.int64)
    for x in order:
        if _can_add_vec(x, chosen, counts):
            chosen, counts = _add_vec(x, chosen, counts)
    return chosen, counts


def _local_search(chosen, counts, rng, deadline, max_iter=500000):
    """Grow/swap local search with fast vectorized validity checks.
    
    Strategy:
      - Try to insert any available element (growth)
      - If can't insert, try swaps that reduce the number of full sums
      - After each swap, retry all available elements for insertion
    """
    n = len(chosen)
    chosen_set = set(int(v) for v in chosen)
    best_size = n
    best_chosen = chosen.copy()
    best_counts = counts.copy()

    all_nums = np.arange(1, N + 1, dtype=np.int64)
    avail_list = [v for v in range(1, N + 1) if v not in chosen_set]
    rng.shuffle(avail_list)
    
    # Track which elements are "full" (count == K) to measure progress
    def count_full(counts_arr):
        return int(np.sum(counts_arr == K))
    
    start = time.time()
    last_growth = start
    no_growth_iters = 0
    
    for it in range(max_iter):
        if (it & 255) == 0:
            if time.time() > deadline:
                break
        
        # Phase A: Try to insert any available element
        inserted = False
        for idx in range(len(avail_list)):
            x_add = avail_list[idx]
            if x_add in chosen_set:
                continue
            if _can_add_vec(x_add, chosen, counts):
                # INSERT: pure growth
                chosen, counts = _add_vec(x_add, chosen, counts)
                chosen_set.add(x_add)
                avail_list.pop(idx)
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = chosen.copy()
                    best_counts = counts.copy()
                    last_growth = time.time()
                    no_growth_iters = 0
                else:
                    no_growth_iters += 1
                inserted = True
                break
        
        if inserted:
            continue
        
        # Phase B: No insertion possible, try a swap
        if len(chosen) < 2:
            break
            
        current_full = count_full(counts)
        
        # Try a few random swaps
        swap_done = False
        for _ in range(10):
            idx_rm = rng.randrange(len(chosen))
            x_rm = int(chosen[idx_rm])
            
            # Find a candidate to add
            if len(avail_list) == 0:
                break
            idx_add = rng.randrange(len(avail_list))
            x_add = avail_list[idx_add]
            if x_add in chosen_set:
                continue
            
            # Compute counts after removing x_rm
            c2 = counts.copy()
            mask = chosen != x_rm
            if chosen.size > 1:
                s_rm = chosen[mask] + x_rm
                c2[s_rm] -= 1
            c2[2 * x_rm] -= 1
            
            # Check if we can add x_add under c2
            if chosen.size > 1:
                s_add = chosen[mask] + x_add
                if np.any(c2[s_add] >= K) or c2[2 * x_add] >= K:
                    continue
            else:
                if c2[2 * x_add] >= K:
                    continue
            
            # Count full sums after swap
            new_full = count_full(c2)
            
            # Accept swap if it reduces full sums or is neutral
            if new_full <= current_full:
                # Apply swap
                chosen_new = chosen[mask]
                chosen, counts = _add_vec(x_add, chosen_new, c2)
                chosen_set.discard(x_rm)
                chosen_set.add(x_add)
                avail_list.pop(idx_add)
                avail_list.append(x_rm)
                swap_done = True
                break
        
        if not swap_done:
            no_growth_iters += 1
            if no_growth_iters > 5000:
                break

    return best_chosen, best_counts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0   # safety margin vs 150s kill

    rng = random.Random(42)

    best_chosen = np.empty(0, dtype=np.int64)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    best_size = 0

    # --- Phase 1: multiple greedy seeds (cheap, fast) ---
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
    ]
    for seed in (123, 7, 999, 2024, 31415, 271828):
        o = list(range(1, N + 1))
        r = random.Random(seed)
        r.shuffle(o)
        orders.append(o)

    for o in orders:
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct

    # --- Phase 2: local search from the best seed (growth + swap) ---
    time_remaining = deadline - time.time()
    if time_remaining > 5:
        ch, ct = _local_search(best_chosen, best_counts, rng, deadline - 5.0, max_iter=500000)
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct

    # --- Phase 3: more local search rounds with different seeds ---
    for trial in range(6):
        if time.time() > deadline - 10:
            break
        o = list(range(1, N + 1))
        rt = random.Random(5000 + trial * 13)
        rt.shuffle(o)
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch2) >= best_size - 5:  # allow slightly smaller starting points
            time_rem = deadline - 10.0 - time.time()
            if time_rem > 2:
                ch2, ct2 = _local_search(ch2, ct2, rng, deadline - 10.0, max_iter=200000)
                if len(ch2) > best_size:
                    best_size = len(ch2)
                    best_chosen = ch2
                    best_counts = ct2

    # --- Phase 4: Final polish - try to insert any remaining elements ---
    if time.time() < deadline - 2:
        chosen_set = set(int(v) for v in best_chosen)
        for x in range(1, N + 1):
            if x in chosen_set:
                continue
            if _can_add_vec(x, best_chosen, best_counts):
                best_chosen, best_counts = _add_vec(x, best_chosen, best_counts)
                chosen_set.add(x)
                if len(best_chosen) > best_size:
                    best_size = len(best_chosen)

    return [int(v) for v in best_chosen]


# EVOLVE-BLOCK-END
