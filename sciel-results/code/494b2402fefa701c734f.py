import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 55.0  # safety margin before 60s kill
    MAX_VAL = 256

    # ---------- Core helper: incremental block-sum tracking ----------
    # For a prefix P, the set of all consecutive-block sums is maintained.
    # When we append x, the NEW sums are exactly the suffix-sums ending at x:
    #   x, x+P[-1], x+P[-1]+P[-2], ..., x+sum(P)
    # i.e. the running total as we walk the prefix from the end.
    # These must be distinct from each other and from every existing sum.

    def greedy_build(start_val, stop_val):
        """Greedily extend an empty sequence from start_val..stop_val."""
        seq = []
        sums = set()
        for x in range(start_val, stop_val + 1):
            total = x
            tails = [x]
            for a in reversed(seq):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                seq.append(x)
                sums.update(tails)
        return seq, sums

    # ---------- Baseline: greedy from 1 ----------
    best, best_sums = greedy_build(1, MAX_VAL)

    # ---------- Multi-start greedy: try different first elements ----------
    # Starting with a different first value explores a different sum-space,
    # sometimes yielding a longer sequence.
    for start in range(1, 25):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        seq, sums = greedy_build(start, MAX_VAL)
        if len(seq) > len(best):
            best, best_sums = seq, sums

    # ---------- DFS / backtracking refinement (from inspirations) ----------
    # The key insight: a consecutive-block sum of a prefix is always
    #   prefix[i] + ... + prefix[j] = (prefix[j+1]-1) - (prefix[i-1]-1)
    # using a "cumulative-with-offset" view.  Equivalently, if we store
    # the prefix values, a new element p creates new sums p - q for every
    # earlier element q (and p itself, treating q=0).  So we track the set
    # of all such differences and require new ones to be fresh.
    #
    # We use the difference formulation: with a virtual prefix[0]=0,
    # appending p gives new diffs {p - prefix[j] for j in range(len(prefix))}.
    # This is exactly the set of consecutive-block sums ending at p.

    def dfs(prefix, diff_set):
        nonlocal best, best_sums
        if time.time() - start_time > TIME_LIMIT - 3:
            return

        n = len(prefix) - 1  # exclude the virtual 0
        if n > len(best):
            best = list(prefix[1:])
            # rebuild sums for record-keeping
            s = set()
            for i in range(1, len(prefix)):
                acc = 0
                for j in range(i, len(prefix)):
                    acc += prefix[j]
                    s.add(acc)
            best_sums = s

        last = prefix[-1]
        # upper bound on how many more we can add
        if n + (MAX_VAL - last) <= len(best):
            return

        for d in range(1, MAX_VAL - last + 1):
            if time.time() - start_time > TIME_LIMIT - 3:
                return
            p = last + d
            # new diffs: p - prefix[j] for every j
            new_diffs = []
            ok = True
            for q in prefix:
                diff = p - q
                if diff in diff_set:
                    ok = False
                    break
                new_diffs.append(diff)
            if not ok:
                continue
            prefix.append(p)
            diff_set.update(new_diffs)
            dfs(prefix, diff_set)
            diff_set.difference_update(new_diffs)
            prefix.pop()

    dfs([0], set())

    # ---------- Time-bounded local search: swap-and-extend ----------
    # Try to replace a single element with a larger one, then greedily
    # extend; keep the result if longer.
    def try_replace(seq, idx, new_val):
        """Return (new_seq, new_sums) if replacing seq[idx] with new_val
        (keeping strictly increasing) and greedily extending is valid and
        longer, else None."""
        if idx == 0:
            if new_val <= 1:
                return None
        else:
            if new_val <= seq[idx - 1]:
                return None
        if idx < len(seq) - 1:
            if new_val >= seq[idx + 1]:
                return None

        cand = seq[:idx] + [new_val] + seq[idx + 1:]
        # rebuild sums and validate
        sums = set()
        ok = True
        n = len(cand)
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += cand[j]
                if acc in sums:
                    ok = False
                    break
                sums.add(acc)
            if not ok:
                break
        if not ok:
            return None
        # greedily extend
        for x in range(cand[-1] + 1, MAX_VAL + 1):
            total = x
            tails = [x]
            for a in reversed(cand):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                cand.append(x)
                sums.update(tails)
        if len(cand) > len(seq):
            return cand, sums
        return None

    def local_search(seq, time_budget):
        seq = list(seq)
        best_local = list(seq)
        improved = True
        while improved and time.time() - start_time < time_budget:
            improved = False
            n = len(seq)
            for idx in range(n):
                if time.time() - start_time > time_budget:
                    break
                lo = seq[idx - 1] + 1 if idx > 0 else 1
                hi = seq[idx + 1] - 1 if idx < n - 1 else MAX_VAL
                for new_val in range(lo, hi + 1):
                    if new_val == seq[idx]:
                        continue
                    res = try_replace(seq, idx, new_val)
                    if res is not None:
                        seq, _ = res
                        best_local = list(seq)
                        improved = True
                        break
                if improved:
                    break
        return best_local

    best = local_search(best, TIME_LIMIT - 2)

    # ---------- Final greedy extension ----------
    extended = list(best)
    sums = set()
    n = len(extended)
    for i in range(n):
        acc = 0
        for j in range(i, n):
            acc += extended[j]
            sums.add(acc)
    for x in range(extended[-1] + 1, MAX_VAL + 1) if extended else range(1, MAX_VAL + 1):
        total = x
        tails = [x]
        for a in reversed(extended):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            extended.append(x)
            sums.update(tails)
    if len(extended) > len(best):
        best = extended

    # Safety: ensure strictly increasing and in range
    assert len(set(best)) == len(best)
    assert all(1 <= v <= MAX_VAL for v in best)
    if len(best) > 1:
        assert all(best[i] < best[i + 1] for i in range(len(best) - 1))

    return best
