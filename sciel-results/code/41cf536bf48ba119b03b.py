import time
import numpy as np

def construct():
    start = time.time()
    DEADLINE = 143.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        vals = set()
        while len(vals) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                vals.add(x)
        return sorted(vals)

    import random
    random.seed(0)

    def is_sum_free(S):
        Sset = set(S)
        for a in S:
            for b in S:
                if a + b in Sset:
                    return False
        return True

    def conflict_counts(S):
        """c[i] = number of (a,b) in S x S with a+b == S[i]."""
        n = len(S)
        c = np.zeros(n, dtype=np.int64)
        if n == 0:
            return c
        A = np.array(S, dtype=np.int64)
        for i in range(n):
            s = A + A[i]
            idx = np.searchsorted(A, s)
            m = idx < n
            hit = m & (A[idx[m]] == s[idx[m]])
            c[i] += int(np.count_nonzero(hit))
        return c

    def try_swap(A, Sset, c, i, j, x):
        """Return (delta, newA, newSset, newc) if swapping S[i] -> x is valid, else None."""
        n = len(A)
        if x in Sset:
            return None
        # conflicts involving x as the sum target
        hits = np.searchsorted(A, x)
        m = hits < n
        hit = m & (A[hits[m]] == x[hits[m]])
        if np.any(hit):
            return None
        # conflicts x as addend with existing (both roles)
        if np.any(np.searchsorted(A, x + A) < n):
            return None
        if np.any(np.searchsorted(A, x - A) < n):
            return None
        if i == j:
            return 1, A, Sset, c
        # remove A[i], A[j]; add x
        newA = A.copy()
        newA[i] = x
        newA[j] = A[i]
        newA = np.sort(newA)
        newSset = set(Sset)
        newSset.discard(A[i])
        newSset.discard(A[j])
        newSset.add(x)
        newc = conflict_counts(newA)
        delta = int(newc.sum() - c.sum())
        return delta, newA, newSset, newc

    def solve_instance(V, time_limit, start_time):
        n = len(V)
        A = np.array(V, dtype=np.int64)
        Sset = set(V)
        c = conflict_counts(A)
        cur = c.sum()

        deadline = start_time + time_limit
        no_improve = 0
        while time.time() < deadline - 0.05:
            if cur == 0:
                break
            # random order of positions
            order = np.random.permutation(n)
            found = False
            for i in order:
                if time.time() > deadline - 0.05:
                    break
                if c[i] == 0:
                    continue
                # candidate x's to try: values that could reduce conflicts
                # try a few: V elements not in set, plus +/- combinations
                cands = []
                # random sample of V
                for _ in range(6):
                    k = np.random.randint(n)
                    if V[k] not in Sset:
                        cands.append(V[k])
                # targeted: values near A[i] +/- A[j]
                for _ in range(4):
                    j = np.random.randint(n)
                    cands.append(A[i] + A[j])
                    cands.append(A[i] - A[j])
                seen = set()
                for x in cands:
                    if x in seen:
                        continue
                    seen.add(x)
                    if x in Sset or x == A[i]:
                        continue
                    res = try_swap(A, Sset, c, i, i, x)
                    if res is not None:
                        delta, newA, newSset, newc = res
                        if delta < 0:
                            A, Sset, c = newA, newSset, newc
                            cur = c.sum()
                            no_improve = 0
                            found = True
                            break
                if found:
                    break
            if not found:
                no_improve += 1
                if no_improve > 60:
                    break

        result = [int(x) for x in A]
        # guarantee
        if not is_sum_free(result):
            # greedy removal
            R = set(result)
            while not is_sum_free(R):
                # find a violation, remove the element that appears most
                lst = list(R)
                best = None
                bestcnt = -1
                for a in lst:
                    cnt = 0
                    for b in lst:
                        if a + b in R:
                            cnt += 1
                    if cnt > bestcnt:
                        bestcnt = cnt
                        best = a
                R.discard(best)
            result = sorted(R)
        return result

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        V = instance(size, seed)
        elapsed = time.time() - start
        remaining = DEADLINE - elapsed
        if remaining < 5:
            # fallback: all odds
            results.append([x for x in V if x % 2 != 0])
            continue
        results.append(solve_instance(V, remaining - 3.0, start))

    return results
