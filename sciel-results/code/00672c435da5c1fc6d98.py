import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    N = 1000
    MAX_SUM = 2 * N
    K = 2

    all_nums = np.arange(1, N + 1, dtype=np.int64)

    def _can_add_vec(x, chosen_arr, counts):
        if chosen_arr.size == 0:
            return counts[2 * x] < K
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
        if counts[2 * x] >= K:
            return False
        return True

    def _add_vec(x, chosen_arr, counts):
        if chosen_arr.size > 0:
            s = chosen_arr + x
            counts[s] += 1
        counts[2 * x] += 1
        return np.concatenate([chosen_arr, np.array([x], dtype=np.int64)]), counts

    def _remove_vec(x, chosen_arr, counts):
        mask = chosen_arr != x
        remaining = chosen_arr[mask]
        if remaining.size > 0:
            s = remaining + x
            counts[s] -= 1
        counts[2 * x] -= 1
        return remaining, counts

    def _saturating_greedy(order, counts):
        chosen = np.empty(0, dtype=np.int64)
        for x in order:
            if _can_add_vec(int(x), chosen, counts):
                chosen, counts = _add_vec(int(x), chosen, counts)
        return chosen, counts

    def _local_search(chosen, counts, rng, deadline, max_iter=500000):
        n = len(chosen)
        chosen_set = set(int(v) for v in chosen)
        best_size = n
        best_chosen = chosen.copy()
        best_counts = counts.copy()

        avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
        rng.shuffle(avail)

        fail_count = 0
        no_improve = 0

        for it in range(max_iter):
            if (it & 255) == 0:
                if time.time() > deadline:
                    break
            if avail.size == 0:
                break

            idx = rng.randrange(avail.size)
            x_add = int(avail[idx])
            if x_add in chosen_set:
                continue

            if _can_add_vec(x_add, chosen, counts):
                chosen, counts = _add_vec(x_add, chosen, counts)
                chosen_set.add(x_add)
                avail = np.delete(avail, idx)
                fail_count = 0
                no_improve = 0
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = chosen.copy()
                    best_counts = counts.copy()
                continue

            # Try swap
            if fail_count > 5 or (it % 50 == 0 and fail_count > 0):
                for _ in range(8):
                    if time.time() > deadline:
                        break
                    if chosen.size == 0:
                        break
                    rm_idx = rng.randrange(chosen.size)
                    x_rm = int(chosen[rm_idx])
                    temp_chosen, temp_counts = _remove_vec(x_rm, chosen, counts)
                    if avail.size == 0:
                        chosen, counts = _add_vec(x_rm, temp_chosen, temp_counts)
                        break
                    add_idx = rng.randrange(avail.size)
                    x_try = int(avail[add_idx])
                    if _can_add_vec(x_try, temp_chosen, temp_counts):
                        final_chosen, final_counts = _add_vec(x_try, temp_chosen, temp_counts)
                        chosen = final_chosen
                        counts = final_counts
                        chosen_set.discard(x_rm)
                        chosen_set.add(x_try)
                        avail = np.delete(avail, add_idx)
                        avail = np.concatenate([avail, np.array([x_rm], dtype=np.int64)])
                        fail_count = 0
                        no_improve = 0
                        break
                    else:
                        chosen, counts = _add_vec(x_rm, temp_chosen, temp_counts)
                fail_count += 1
            else:
                fail_count += 1

            # Periodic no-improve reset: randomize avail order
            if no_improve > 50:
                no_improve = 0
                rng.shuffle(avail)

        return best_chosen, best_counts

    # --- Phase 1: Multiple Greedy Seeds ---
    best_chosen = np.empty(0, dtype=np.int64)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    best_size = 0

    # Deterministic orders
    orders = [all_nums, all_nums[::-1]]

    # Stride-based orders
    for step in (2, 3, 5, 7, 11):
        o = all_nums[::step]
        orders.append(o)
        o2 = all_nums[::-step]
        orders.append(o2)

    # Randomized orders
    for seed in (123, 7, 999, 2024, 555, 111, 42, 777):
        o = all_nums.copy()
        r = random.Random(seed)
        r.shuffle(o)
        orders.append(o)

    # Weighted orders (favor smaller numbers first)
    for w in (1.5, 2.0, 0.5):
        weights = np.array([1.0 / (v ** w) for v in range(1, N + 1)])
        weights /= weights.sum()
        o = np.random.choice(all_nums, size=N, replace=False, p=weights)
        orders.append(o)

    for o in orders:
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch
            best_counts = ct

    # --- Phase 2: Local Search from Best Seed ---
    rng = random.Random(42)
    ch, ct = _local_search(best_chosen, best_counts, rng, deadline, max_iter=2000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        best_counts = ct

    # --- Phase 3: Try other good seeds with Local Search if time remains ---
    for trial in range(8):
        if time.time() > deadline:
            break
        seed = 10000 + trial * 101
        o = all_nums.copy()
        rt = random.Random(seed)
        rt.shuffle(o)
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch2) < best_size - 2:
            continue
        current_deadline = min(deadline, time.time() + 8)
        if current_deadline <= time.time():
            break
        ch2, ct2 = _local_search(ch2, ct2, rng, current_deadline, max_iter=300000)
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = ch2
            best_counts = ct2

    return [int(v) for v in best_chosen]

# EVOLVE-BLOCK-END
