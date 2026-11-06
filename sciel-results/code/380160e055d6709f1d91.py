import time
import random

def construct():
    start_time = time.time()
    time_limit = 80.0  # Safety margin for 90s kill

    # ---------- validation ----------
    def validate_seq(seq):
        n = len(seq)
        if n == 0:
            return True
        for i in range(n):
            if seq[i] < 1 or seq[i] > 256:
                return False
            if i > 0 and seq[i] <= seq[i-1]:
                return False
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        sums = set()
        for i in range(n):
            for j in range(i+1, n+1):
                s = prefix[j] - prefix[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def build_state(seq):
        """Return (prefix_sums, used_sums) for a validated increasing seq."""
        n = len(seq)
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + seq[i]
        used = set()
        for i in range(n):
            for j in range(i+1, n+1):
                used.add(prefix[j] - prefix[i])
        return prefix, used

    def can_add_fast(prefix_sums, used_sums, x, last_val):
        """O(n) incremental check. Returns (new_prefix, new_sums) or None."""
        if x <= last_val or x > 256:
            return None
        new_prefix = prefix_sums[-1] + x
        new_sums = []
        for p in prefix_sums:
            s = new_prefix - p
            if s in used_sums:
                return None
            new_sums.append(s)
        seen = set()
        for s in new_sums:
            if s in seen:
                return None
            seen.add(s)
        return (new_prefix, new_sums)

    # ---------- greedy ----------
    def greedy_smallest(start_val=1):
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        for _ in range(256):
            found = False
            for x in range(start_val, 257):
                res = can_add_fast(prefix_sums, used_sums, x, last_val)
                if res is not None:
                    new_prefix, new_sums = res
                    seq.append(x)
                    prefix_sums.append(new_prefix)
                    used_sums.update(new_sums)
                    last_val = x
                    start_val = x + 1
                    found = True
                    break
            if not found:
                break
        return seq

    def greedy_randomized(start_val=1, seed_offset=0, bias_power=1.0):
        seq = []
        prefix_sums = [0]
        used_sums = set()
        last_val = 0
        rng = random.Random(seed_offset)
        for _ in range(256):
            valid = []
            for x in range(start_val, 257):
                res = can_add_fast(prefix_sums, used_sums, x, last_val)
                if res is not None:
                    valid.append((x, res))
            if not valid:
                break
            if len(valid) <= 3:
                x, (np_, ns) = valid[0]
            else:
                weights = [(1.0 / (i + 1)) ** bias_power for i in range(len(valid))]
                total_w = sum(weights)
                r = rng.random() * total_w
                cum = 0
                chosen = 0
                for i, w in enumerate(weights):
                    cum += w
                    if r <= cum:
                        chosen = i
                        break
                x, (np_, ns) = valid[chosen]
            seq.append(x)
            prefix_sums.append(np_)
            used_sums.update(ns)
            last_val = x
            start_val = x + 1
        return seq

    # ---------- backtracking with lookahead ----------
    def backtrack_search(prefix_sums, used_sums, seq, target_len,
                         max_time, order='smallest', lookahead=0, seed=0):
        """Try to build a sequence of length target_len from current state.
        order: 'smallest' or 'largest' ordering of candidate values.
        lookahead: prune a branch if it cannot be extended by `lookahead` more.
        Returns a full valid list of length target_len, or None."""
        deadline = start_time + max_time
        rng = random.Random(seed)
        n = len(seq)

        # candidate ordering function
        def candidate_range(last_val):
            if order == 'smallest':
                return range(last_val + 1, 257)
            else:
                return range(256, last_val, -1)

        def quick_extend(prefix_sums, used_sums, last_val, k):
            """Greedily try to extend by k (smallest) to check feasibility."""
            ps = list(prefix_sums)
            us = set(used_sums)
            lv = last_val
            for _ in range(k):
                found = False
                for x in range(lv + 1, 257):
                    res = can_add_fast(ps, us, x, lv)
                    if res is not None:
                        np_, ns = res
                        ps.append(np_)
                        us.update(ns)
                        lv = x
                        found = True
                        break
                if not found:
                    return False
            return True

        def dfs(last_val, depth):
            if depth == target_len:
                return True
            if time.time() > deadline:
                return False
            for x in candidate_range(last_val):
                if time.time() > deadline:
                    return False
                res = can_add_fast(prefix_sums, used_sums, x, last_val)
                if res is None:
                    continue
                np_, ns = res
                # lookahead prune
                if lookahead > 0 and depth + 1 < target_len:
                    if not quick_extend(prefix_sums + [np_],
                                         used_sums | set(ns), x, lookahead):
                        continue
                seq.append(x)
                prefix_sums.append(np_)
                used_sums.update(ns)
                if dfs(x, depth + 1):
                    return True
                seq.pop()
                prefix_sums.pop()
                for s in ns:
                    used_sums.discard(s)
            return False

        if dfs(seq[-1] if seq else 0, len(seq)):
            return list(seq)
        return None

    def iterative_deepen_from(base_seq, max_time, order='smallest', lookahead=0, seed=0):
        """Given a valid base_seq, try to extend it to longer lengths via backtracking."""
        base = list(base_seq)
        prefix, used = build_state(base)
        best = list(base)
        # try extending by 1, 2, ... up to a cap
        for extra in range(1, 8):
            target = len(base) + extra
            if target > 256:
                break
            if time.time() > start_time + max_time:
                break
            res = backtrack_search(prefix, used, list(base), target,
                                   max_time=min(max_time, 3.0),
                                   order=order, lookahead=lookahead, seed=seed)
            if res is not None and len(res) > len(best):
                best = res
            # stop early if we couldn't even extend by 1
            if extra == 1 and res is None:
                break
        return best

    # ---------- local search: insert into gaps ----------
    def local_insert(seq, time_budget=4.0):
        """Try to insert an extra element into a gap of seq to make it longer."""
        start_ls = time.time()
        best = list(seq)
        best_len = len(best)
        rng = random.Random(hash(tuple(seq)) % 100000)

        # Build a list of candidate insert positions with their value ranges
        def try_insert(pos, x):
            cand = best[:pos] + [x] + best[pos:]
            return validate_seq(cand)

        # First pass: systematic scan of every gap for any insertable value
        for _ in range(200):
            if time.time() - start_ls > time_budget:
                break
            if best_len >= 256:
                break
            n = len(best)
            # candidate gaps
            gaps = list(range(n + 1))
            rng.shuffle(gaps)
            improved = False
            for pos in gaps:
                if time.time() - start_ls > time_budget:
                    break
                if pos == 0:
                    lo, hi = 1, (best[0] - 1 if n > 0 else 256)
                elif pos == n:
                    lo, hi = best[-1] + 1, 256
                else:
                    lo, hi = best[pos-1] + 1, best[pos] - 1
                if lo > hi:
                    continue
                # try a handful of values in this gap
                span = hi - lo + 1
                if span <= 12:
                    vals = list(range(lo, hi + 1))
                else:
                    vals = [rng.randint(lo, hi) for _ in range(12)]
                for x in vals:
                    if try_insert(pos, x):
                        best = best[:pos] + [x] + best[pos:]
                        best_len += 1
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    # ---------- orchestration ----------
    # 1. Greedy baseline
    greedy_result = greedy_smallest()
    best_seq = list(greedy_result)
    best_len = len(best_seq)

    # 2. Greedy from varied starts
    for s in range(1, 30):
        if time.time() - start_time > time_limit - 40:
            break
        seq = greedy_smallest(s)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # 3. Randomized greedy
    for attempt in range(120):
        if time.time() - start_time > time_limit - 35:
            break
        bias = 0.5 + (attempt % 5) * 0.3
        sv = 1 + (attempt % 10)
        seq = greedy_randomized(start_val=sv, seed_offset=attempt * 7 + 1, bias_power=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # 4. Backtracking extension (smallest-first) from best greedy
    if time.time() - start_time < time_limit - 30:
        ext = iterative_deepen_from(best_seq, max_time=10.0, order='smallest', lookahead=2, seed=1)
        if len(ext) > best_len:
            best_len = len(ext)
            best_seq = ext

    # 5. Backtracking extension (largest-first) from best
    if time.time() - start_time < time_limit - 25:
        ext = iterative_deepen_from(best_seq, max_time=8.0, order='largest', lookahead=1, seed=2)
        if len(ext) > best_len:
            best_len = len(ext)
            best_seq = ext

    # 6. Local insert polish
    if time.time() - start_time < time_limit - 15:
        improved = local_insert(best_seq, time_budget=6.0)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved

    # 7. More diverse randomized attempts with fresh seeds
    for attempt in range(150):
        if time.time() - start_time > time_limit - 12:
            break
        sv = random.randint(1, 40)
        bias = random.uniform(0.3, 2.5)
        seq = greedy_randomized(start_val=sv, seed_offset=10000 + attempt * 13, bias_power=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # 8. Final backtracking push + local insert
    if time.time() - start_time < time_limit - 6:
        ext = iterative_deepen_from(best_seq, max_time=4.0, order='smallest', lookahead=1, seed=3)
        if len(ext) > best_len:
            best_len = len(ext)
            best_seq = ext
        if time.time() - start_time < time_limit - 3:
            improved = local_insert(best_seq, time_budget=3.0)
            if len(improved) > best_len:
                best_len = len(improved)
                best_seq = improved

    # Final validation
    if not validate_seq(best_seq):
        best_seq = [1, 2, 4, 8, 16, 32, 64, 128]
        assert validate_seq(best_seq), "Fallback invalid"

    return best_seq
