import time
import random
import numpy as np

N = 1000
MAX_SUM = 2 * N  # 2000
K = 2            # max representations allowed


def _can_add_vec(x, chosen_arr, counts):
    """True if adding x keeps every sum <= K. Vectorized."""
    if chosen_arr.size:
        s = chosen_arr + x
        if np.any(counts[s] >= K):
            return False
    if counts[2 * x] >= K:
        return False
    return True


def _add_vec(x, chosen_arr, counts):
    """Update counts in place after adding x."""
    if chosen_arr.size:
        s = chosen_arr + x
        counts[s] += 1
    counts[2 * x] += 1


def _remove_vec(x, chosen_arr, counts):
    """Update counts in place after removing x (x assumed present)."""
    if chosen_arr.size:
        s = chosen_arr + x
        counts[s] -= 1
    counts[2 * x] -= 1


def _saturating_greedy(order, counts, noise=0.0, rng=None):
    """Greedy that adds ANY valid element (order decides which first)."""
    chosen = np.empty(0, dtype=np.int64)
    for x in order:
        if _can_add_vec(x, chosen, counts):
            if noise > 0.0 and rng is not None and rng.random() < noise:
                continue
            _add_vec(x, chosen, counts)
            chosen = np.concatenate([chosen, np.array([x], dtype=np.int64)])
    return chosen, counts


def _local_search(chosen, counts, rng, deadline, max_iter=500000):
    """Local search with INSERT, SWAP, and 2-OPT moves."""
    n = len(chosen)
    chosen_set = set(int(v) for v in chosen)
    best_size = n
    best_chosen = chosen.copy()

    # Precompute available candidates
    avail = np.array([v for v in range(1, N + 1) if v not in chosen_set], dtype=np.int64)
    rng.shuffle(avail)

    start = time.time()
    for it in range(max_iter):
        if (it & 255) == 0:
            if time.time() > deadline:
                break
        if avail.size == 0:
            break
        if n == 0:
            break

        # Pick a random available candidate
        idx_add = rng.randrange(avail.size)
        x_add = int(avail[idx_add])

        # ---- (A) direct INSERT (pure growth) ----
        if _can_add_vec(x_add, chosen, counts):
            _add_vec(x_add, chosen, counts)
            chosen = np.concatenate([chosen, np.array([x_add], dtype=np.int64)])
            chosen_set.add(x_add)
            # Remove from avail
            avail = np.delete(avail, idx_add)
            n += 1
            if n > best_size:
                best_size = n
                best_chosen = chosen.copy()
            continue

        # ---- (B) SWAP : remove one, add x_add (same size) ----
        # Try a random swap
        idx_rm = rng.randrange(n)
        x_rm = int(chosen[idx_rm])

        # Check if swap is valid
        # Temporarily remove x_rm
        c2 = counts.copy()
        if chosen.size:
            c2[chosen + x_rm] -= 1
        c2[2 * x_rm] -= 1
        
        # Check if x_add can be added to (chosen - x_rm)
        valid_swap = True
        if chosen.size > 1:
            s = (chosen[chosen != x_rm] + x_add)
            if np.any(c2[s] >= K):
                valid_swap = False
        if c2[2 * x_add] >= K:
            valid_swap = False

        if valid_swap:
            # Apply swap
            _add_vec(x_add, chosen[chosen != x_rm], c2)
            chosen = np.concatenate([chosen[chosen != x_rm], np.array([x_add], dtype=np.int64)])
            chosen_set.discard(x_rm)
            chosen_set.add(x_add)
            
            # Update avail
            # x_add is removed from avail
            avail = np.delete(avail, idx_add)
            # x_rm is added to avail
            avail = np.concatenate([avail, np.array([x_rm], dtype=np.int64)])
            rng.shuffle(avail)
            # n unchanged
            
        # ---- (C) 2-OPT : remove two, add two (growth potential) ----
        # This is complex and slow. We'll rely on INSERT and SWAP primarily.
        # If we have time, we might try specific 2-opts, but for now, 
        # frequent SWAPs can break deadlocks.

    return best_chosen


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    rng = random.Random(42)

    best_chosen = np.empty(0, dtype=np.int64)
    best_size = 0

    # --- Phase 1: Diverse Greedy Seeds ---
    seed_orders = []
    
    # Deterministic ascending
    seed_orders.append(list(range(1, N + 1)))
    
    # Deterministic descending
    seed_orders.append(list(range(N, 0, -1)))
    
    # Random shuffles
    for seed in [123, 7, 999, 2024, 555, 31337, 888, 222, 111, 9999]:
        o = list(range(1, N + 1))
        r = random.Random(seed)
        r.shuffle(o)
        seed_orders.append(o)

    for o in seed_orders:
        ch, ct = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64))
        if len(ch) > best_size:
            best_size = len(ch)
            best_chosen = ch

    # --- Phase 2: Local Search from Best Seed ---
    counts = np.zeros(MAX_SUM + 1, dtype=np.int64)
    a = best_chosen
    if a.size:
        iu, ju = np.triu_indices(a.size)
        np.add.at(counts, a[iu] + a[ju], 1)
    
    # Run local search
    ch = _local_search(best_chosen, counts, rng, deadline, max_iter=1000000)
    if len(ch) > best_size:
        best_size = len(ch)
        best_chosen = ch

    # --- Phase 3: Restart Loop ---
    # If we have time, try a few more restarts with local search
    for trial in range(20):
        if time.time() > deadline:
            break
        o = list(range(1, N + 1))
        rt = random.Random(10000 + trial * 17)
        rt.shuffle(o)
        # Add some noise to greedy to diversify
        noise = 0.1 if trial % 2 == 0 else 0.0
        ch2, ct2 = _saturating_greedy(o, np.zeros(MAX_SUM + 1, dtype=np.int64), noise=noise, rng=rt if noise else None)
        
        if len(ch2) <= best_size:
            # Still might be worth local searching if it's close?
            # For now, only search if it's already better or very close
            if len(ch2) < best_size - 2:
                continue
        
        # Run a shorter local search
        ch2 = _local_search(ch2, ct2, rt, deadline, max_iter=200000)
        if len(ch2) > best_size:
            best_size = len(ch2)
            best_chosen = ch2

    return [int(v) for v in best_chosen]
