import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N  # 2000
K = 2  # max representations allowed


def _can_add_vec(x, chosen_arr, counts):
    """True if adding x keeps every sum <= K. Vectorized."""
    if chosen_arr.size:
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
    s2 = 2 * x
    if counts[s2] >= K:
        return False
    return True


def _add_vec(x, chosen_arr, counts):
    """Return new chosen array (appended) and updated counts after adding x."""
    if chosen_arr.size:
        s = chosen_arr + x
        counts[s] += 1
    counts[2 * x] += 1
    return np.concatenate([chosen_arr, np.array([x], dtype=np.int64)]), counts


def _remove_vec(x, chosen_arr, counts):
    """Return new chosen array and counts after removing x (assumed present)."""
    mask = chosen_arr != x
    if chosen_arr.size:
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


def _spread_order(rng):
    """Create an ordering that spreads elements to reduce sum collisions.
    Start with widely spaced numbers, then fill in gaps."""
    order = []
    # First pass: pick every ~k-th number for large k, decreasing k
    for k in [50, 30, 20, 15, 10, 7, 5, 3, 2, 1]:
        for i in range(1, N + 1, k):
            if i not in order:
                order.append(i)
    # Shuffle within groups of similar spacing to add diversity
    rng.shuffle(order)
    return order


def _local_search(chosen, counts, rng, deadline, max_iter=2000000):
    """Grow/swap local search with fast vectorized validity checks.

    Moves:
      (A) INSERT a new element (grows size) -- primary objective.
      (B) SWAP one out for one in (keeps size, may enable later growth).
      (C) Try to find the 'best' insert candidate (one that creates fewest conflicts).
    """
    n = len(chosen)
    chosen_set = set(int(v) for v in chosen)
    best_size = n
    best_chosen = chosen.copy()
    best_counts = counts.copy()

    all_nums = np.arange(1, N + 1, dtype=np.int64)
    avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
    rng.shuffle(avail)

    start = time.time()
    stagnant = 0
    for it in range(max_iter):
        if (it & 511) == 0:
            if time.time() > deadline:
                break
        if avail.size == 0:
            break

        # Try a batch of random candidates for insertion
        batch_size = min(20, avail.size)
        found_insert = False
        for _ in range(batch_size):
            x_add = int(avail[rng.randrange(avail.size)])
            if x_add in chosen_set:
                continue

            if _can_add_vec(x_add, chosen, counts):
                # INSERT: pure growth
                chosen, counts = _add_vec(x_add, chosen, counts)
                chosen_set.add(x_add)
                avail = avail[avail != x_add]
                if len(chosen) > best_size:
                    best_size = len(chosen)
                    best_chosen = chosen.copy()
                    best_counts = counts.copy()
                    stagnant = 0
                found_insert = True
                break

        if found_insert:
            continue

        # Cannot insert directly -> try a SWAP to free capacity
        # Prioritize removing elements that are 'bottlenecks'
        idx = rng.randrange(len(chosen))
        x_rm = int(chosen[idx])

        # counts after removing x_rm
        c2 = counts.copy()
        s_rm = chosen[chosen != x_rm] + x_rm
        c2[s_rm] -= 1
        c2[2 * x_rm] -= 1

        # Try a few random available candidates for the swap
        swap_found = False
        for _ in range(10):
            x_add = int(avail[rng.randrange(avail.size)])
            if x_add in chosen_set:
                continue

            # can we add x_add under c2?
            s_add = chosen[chosen != x_rm] + x_add
            if np.any(c2[s_add] >= K) or c2[2 * x_add] >= K:
                continue

            # apply swap
            chosen, counts = _add_vec(x_add, chosen[chosen != x_rm], c2)
            chosen_set.discard(x_rm)
            chosen_set.add(x_add)
            if x_add in avail:
                avail = avail[avail != x_add]
            if x_rm not in avail:
                avail = np.concatenate([avail, np.array([x_rm], dtype=np.int64)])
            swap_found = True
            break

        if not swap_found:
            stagnant += 1
            # If very stagnant, try a more aggressive restart of the local search
            if stagnant > 5000:
                # Reset to best and try a different strategy
                chosen = best_chosen.copy()
                counts = best_counts.copy()
                chosen_set = set(int(v) for v in chosen)
                avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
                rng.shuffle(avail)
                stagnant = 0

    return best_chosen, best_counts


def _polish_pass(chosen, counts, deadline):
    """Final pass: try to insert any remaining available elements systematically."""
    chosen_set = set(int(v) for v in chosen)
    avail = [v for v in range(1, N + 1) if v not in chosen_set]
    
    # Sort by a heuristic: prefer elements that are 'far' from existing ones
    # to minimize new sum collisions
    def conflict_score(x):
        if not chosen.size:
            return 0
        s = chosen + x
        conflicts = np.sum(counts[s] >= K)
        if counts[2 * x] >= K:
            conflicts += 1
        return conflicts
    
    # Try in order of lowest conflict score
    avail.sort(key=lambda x: conflict_score(x))
    
    for x in avail:
        if time.time() > deadline:
            break
        if _can_add_vec(x, chosen, counts):
            chosen, counts = _add_vec(x, chosen, counts)
            chosen_set.add(x)
    
    return chosen, counts


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin vs 150s kill

    rng = random.Random(42)

    best_chosen = np.empty(0, dtype=np.int64)
    best_counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    best_size = 0

    # --- Phase 1: multiple greedy seeds (cheap, fast) ---
    orders = []
    
    # Natural order
    orders.append(list(range(1, N + 1)))
    # Reverse order
    orders.append(list(range(N, 0, -1)))
    # Spread orders with different seeds
    for seed in [123, 7, 999, 2024, 555, 888]:
        r = random.Random(seed)
        orders.append(_spread_order(r))
    # Random shuffles
    for seed in [42, 1337, 2024, 9999]:
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
    ch, ct = _local_search(best_chosen, best_counts, rng, deadline, max_iter=2000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        best_counts = ct

    # --- Phase 3: polish pass ---
    ch, ct = _polish_pass(best_chosen, best_counts, deadline)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch
        best_counts = ct

    # --- Phase 4: if time remains, try a couple more seeds + search ---
    for trial in range(4):
        if time.time() > deadline:
            break
        r = random.Random(5000 + trial * 13)
        o = _spread_order(r)
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch2) <= best_size:
            continue
        ch2, ct2 = _local_search(ch2, ct2, rng, deadline, max_iter=200000)
        ch2, ct2 = _polish_pass(ch2, ct2, deadline)
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = ch2
            best_counts = ct2

    return [int(v) for v in best_chosen]
