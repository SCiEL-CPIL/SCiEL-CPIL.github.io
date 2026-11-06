import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_VAL = 256

    def get_consecutive_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            total = 0
            for j in range(i, n):
                total += seq[j]
                sums.add(total)
        return sums

    def is_valid(seq):
        if not seq:
            return True
        if seq[0] < 1 or seq[-1] > MAX_VAL:
            return False
        for i in range(len(seq) - 1):
            if seq[i] >= seq[i + 1]:
                return False
        sums = get_consecutive_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def check_valid(seq):
        if not seq:
            return True
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # ---- Fast greedy: build sequence in a given order, check tail sums ----
    def fast_greedy(order, p_accept=1.0, rng=None):
        values = []
        sums = set()
        for x in order:
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                if p_accept >= 1.0 or (rng is not None and rng.random() < p_accept):
                    values.append(x)
                    sums.update(tails)
        values.sort()
        return values

    # ---- Local search: try to add elements, then swap to escape local optima ----
    def local_search(seq, time_limit, rng):
        current = list(seq)
        current_set = set(current)
        best_local = list(current)
        best_local_len = len(current)

        while time.time() < time_limit - 2.0:
            # Try to add any element
            candidates = [x for x in range(1, MAX_VAL + 1) if x not in current_set]
            if not candidates:
                break

            rng.shuffle(candidates)
            added = False
            for x in candidates:
                if time.time() > time_limit - 2.0:
                    break
                new_seq = sorted(current + [x])
                if check_valid(new_seq):
                    current = new_seq
                    current_set.add(x)
                    if len(current) > best_local_len:
                        best_local_len = len(current)
                        best_local = list(current)
                    added = True
                    break

            if added:
                continue

            # Try swaps to escape local optimum
            swap_made = False
            for _ in range(80):
                if time.time() > time_limit - 2.0:
                    break
                if not current:
                    break
                i_val = rng.choice(current)
                j_val = rng.choice(candidates)
                new_set = current_set - {i_val} | {j_val}
                new_seq = sorted(new_set)
                if check_valid(new_seq):
                    current = new_seq
                    current_set = new_set
                    swap_made = True
                    break

            if not swap_made:
                break

        return best_local

    # ---- Branch-and-bound DFS with suffix-sum collision tracking ----
    def bb_dfs(seq, sums, tot, lo_val, time_limit, rng, best_ref):
        """
        best_ref is a list [best_seq, best_len] modified in place.
        """
        if time.time() > time_limit - 1.0:
            return
        n = len(seq)
        if n > best_ref[1]:
            best_ref[1] = n
            best_ref[0] = list(seq)

        if n > 0:
            max_add = MAX_VAL - seq[-1]
        else:
            max_add = MAX_VAL
        if n + max_add <= best_ref[1]:
            return

        # Try candidate values: mix of high-to-low and random
        vals = list(range(lo_val, MAX_VAL + 1))
        if rng.random() < 0.4:
            vals.reverse()
        else:
            rng.shuffle(vals)

        for x in vals:
            if time.time() > time_limit - 1.0:
                return
            # New sums when appending x
            new_sums = set()
            s = x
            new_sums.add(s)
            suffix = 0
            for a in reversed(seq):
                suffix += a
                s = x + suffix
                new_sums.add(s)

            if new_sums & sums:
                continue

            new_seq = seq + [x]
            new_sums_set = sums | new_sums
            new_total = tot + x

            bb_dfs(new_seq, new_sums_set, new_total, x + 1, time_limit, rng, best_ref)

    def bb_search(seq, time_limit, rng):
        current = list(seq)
        sums_set = get_consecutive_sums(current)
        total = sum(current)
        if not current:
            lo = 1
        else:
            lo = current[-1] + 1
        if lo > MAX_VAL:
            return current
        best_ref = [list(current), len(current)]
        bb_dfs(current, sums_set, total, lo, time_limit, rng, best_ref)
        return best_ref[0]

    # ---- Build initial candidate pool ----
    rng = random.Random(42)
    candidates = []

    # Deterministic greeds
    candidates.append(fast_greedy(range(MAX_VAL, 0, -1)))
    candidates.append(fast_greedy(range(1, MAX_VAL + 1)))
    for step in (2, 3, 4):
        candidates.append(fast_greedy(range(MAX_VAL, 0, -step)))
        candidates.append(fast_greedy(range(1, MAX_VAL + 1, step)))

    # Randomized greedy restarts (fast)
    for _ in range(30):
        if time.time() > deadline - 25.0:
            break
        order = list(range(1, MAX_VAL + 1))
        rng.shuffle(order)
        p = rng.uniform(0.7, 1.0)
        candidates.append(fast_greedy(order, p, rng))

    # Keep top candidates
    candidates = sorted(candidates, key=len, reverse=True)
    best_seq = candidates[0]
    best_len = len(best_seq)

    # ---- Phase 2: Local search on top candidates ----
    top_n = min(6, len(candidates))
    time_per_ls = (deadline - 15.0) / max(1, top_n)

    for seq in candidates[:top_n]:
        if time.time() > deadline - 10.0:
            break
        if len(seq) < best_len - 2:
            continue
        t_end = min(time.time() + time_per_ls, deadline - 8.0)
        improved = local_search(list(seq), t_end, rng)
        if len(improved) > best_len:
            best_seq, best_len = improved, len(improved)

    # ---- Phase 3: Branch-and-bound on the best sequence ----
    if time.time() < deadline - 5.0:
        ext_seq = bb_search(list(best_seq), deadline - 2.0, rng)
        if len(ext_seq) > best_len:
            best_seq, best_len = ext_seq, len(ext_seq)

    # ---- Phase 4: Final local search on the best ----
    if time.time() < deadline - 3.0:
        improved = local_search(list(best_seq), deadline - 1.0, rng)
        if len(improved) > best_len:
            best_seq, best_len = improved, len(improved)

    return best_seq
