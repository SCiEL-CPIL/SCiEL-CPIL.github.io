import time
import random


def construct():
    start_time = time.time()
    time_limit = 82.0  # safety margin for 90s kill

    # ---------- bitset helpers ----------
    # prefix sums are bounded by 256*256 = 65536; use a Python int as a bitset.
    def new_sums_mask(p_last, prefix_list, x):
        """Bitmask of the sums of all nonempty consecutive blocks ending at the
        newly appended element x.  O(n) bitwise ops."""
        m = 0
        s = x
        for p in prefix_list:          # p_last, p_last - a[n-1], ... , 0
            m |= (1 << s)
            s -= p
        return m

    def feasible(prefix_list, used_mask, x):
        """Return (ok, new_mask) for appending x (x must be > last)."""
        p_last = prefix_list[-1]
        new_prefix = p_last + x
        m = 0
        s = x
        for p in prefix_list:
            m |= (1 << s)
            s -= p
        m |= (1 << new_prefix)         # prefix sum P[n+1] must be unused
        if m & used_mask:
            return False, 0
        return True, m

    # ---------- validation (final, independent) ----------
    def validate_seq(seq):
        n = len(seq)
        if n == 0:
            return True
        for i in range(n):
            if seq[i] < 1 or seq[i] > 256:
                return False
            if i > 0 and seq[i] <= seq[i - 1]:
                return False
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + seq[i]
        if len(set(prefix)) != n + 1:
            return False
        return True

    # ---------- greedy ----------
    def greedy_smallest(start_val=1):
        seq = []
        prefix = [0]
        used = 1  # bit 0 (P[0])
        last = 0
        for _ in range(256):
            found = False
            for x in range(start_val, 257):
                ok, m = feasible(prefix, used, x)
                if ok:
                    seq.append(x)
                    prefix.append(prefix[-1] + x)
                    used |= m
                    last = x
                    start_val = x + 1
                    found = True
                    break
            if not found:
                break
        return seq

    def greedy_randomized(start_val=1, seed_offset=0, bias_power=1.0):
        seq = []
        prefix = [0]
        used = 1
        last = 0
        rng = random.Random(seed_offset)
        for _ in range(256):
            valid = []
            for x in range(start_val, 257):
                ok, m = feasible(prefix, used, x)
                if ok:
                    valid.append((x, m))
            if not valid:
                break
            if len(valid) <= 3:
                x, m = valid[0]
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
                x, m = valid[chosen]
            seq.append(x)
            prefix.append(prefix[-1] + x)
            used |= m
            last = x
            start_val = x + 1
        return seq

    # ---------- backtracking (iterative deepening) ----------
    def greedy_extend_len(prefix_list, used_mask, last_val, k):
        """Greedily extend (smallest) by k; return length achieved (may be < k)."""
        cnt = 0
        for _ in range(k):
            found = False
            for x in range(last_val + 1, 257):
                ok, m = feasible(prefix_list, used_mask, x)
                if ok:
                    prefix_list.append(prefix_list[-1] + x)
                    used_mask |= m
                    last_val = x
                    cnt += 1
                    found = True
                    break
            if not found:
                break
        return cnt

    def backtrack(prefix_list, used_mask, seq, target_len, max_time,
                  order='smallest', lookahead=0, seed=0):
        deadline = start_time + max_time
        rng = random.Random(seed)
        n = len(seq)

        def candidate_range(last_val):
            if order == 'smallest':
                return range(last_val + 1, 257)
            else:
                return range(256, last_val, -1)

        def dfs(last_val, depth):
            if depth == target_len:
                return True
            if time.time() > deadline:
                return False
            for x in candidate_range(last_val):
                if time.time() > deadline:
                    return False
                ok, m = feasible(prefix_list, used_mask, x)
                if not ok:
                    continue
                if lookahead > 0 and depth + 1 < target_len:
                    if greedy_extend_len(prefix_list, used_mask, x, lookahead) < lookahead:
                        continue
                prefix_list.append(prefix_list[-1] + x)
                used_mask |= m
                seq.append(x)
                if dfs(x, depth + 1):
                    return True
                seq.pop()
                prefix_list.pop()
                used_mask ^= m
            return False

        if dfs(seq[-1] if seq else 0, len(seq)):
            return list(seq)
        return None

    def iterative_deepen(max_time, order='smallest', lookahead=0, seed=0):
        """Iteratively deepen from empty to find the longest sequence."""
        best = []
        for target in range(1, 257):
            if time.time() > start_time + max_time:
                break
            res = backtrack([0], 1, [], target,
                            max_time=min(max_time, 1.5),
                            order=order, lookahead=lookahead, seed=seed)
            if res is None:
                break
            if len(res) > len(best):
                best = res
        return best

    # ---------- local insert polish (bitset) ----------
    def local_insert(seq, time_budget=4.0):
        start_ls = time.time()
        best = list(seq)
        rng = random.Random(hash(tuple(seq)) % 100000)

        def try_insert(pos, x):
            cand = best[:pos] + [x] + best[pos:]
            return validate_seq(cand)

        for _ in range(300):
            if time.time() - start_ls > time_budget:
                break
            n = len(best)
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
                    lo, hi = best[pos - 1] + 1, best[pos] - 1
                if lo > hi:
                    continue
                span = hi - lo + 1
                if span <= 12:
                    vals = list(range(lo, hi + 1))
                else:
                    vals = [rng.randint(lo, hi) for _ in range(12)]
                for x in vals:
                    if try_insert(pos, x):
                        best = best[:pos] + [x] + best[pos:]
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best

    # ---------- orchestration ----------
    best_seq = greedy_smallest()
    best_len = len(best_seq)

    # 1. greedy from varied starts
    for s in range(1, 30):
        if time.time() - start_time > time_limit - 45:
            break
        seq = greedy_smallest(s)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # 2. randomized greedy
    for attempt in range(140):
        if time.time() - start_time > time_limit - 40:
            break
        bias = 0.5 + (attempt % 5) * 0.3
        sv = 1 + (attempt % 10)
        seq = greedy_randomized(start_val=sv, seed_offset=attempt * 7 + 1, bias_power=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # 3. iterative-deepening backtracking (smallest-first) to find longest
    if time.time() - start_time < time_limit - 35:
        ext = iterative_deepen(max_time=12.0, order='smallest', lookahead=2, seed=1)
        if len(ext) > best_len:
            best_len = len(ext)
            best_seq = ext

    # 4. iterative-deepening backtracking (largest-first)
    if time.time() - start_time < time_limit - 28:
        ext = iterative_deepen(max_time=8.0, order='largest', lookahead=1, seed=2)
        if len(ext) > best_len:
            best_len = len(ext)
            best_seq = ext

    # 5. local insert polish
    if time.time() - start_time < time_limit - 18:
        improved = local_insert(best_seq, time_budget=6.0)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved

    # 6. more diverse randomized attempts
    for attempt in range(160):
        if time.time() - start_time > time_limit - 12:
            break
        sv = random.randint(1, 40)
        bias = random.uniform(0.3, 2.5)
        seq = greedy_randomized(start_val=sv, seed_offset=10000 + attempt * 13, bias_power=bias)
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = list(seq)

    # 7. final backtracking push + local insert
    if time.time() - start_time < time_limit - 6:
        ext = iterative_deepen(max_time=4.0, order='smallest', lookahead=1, seed=3)
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
