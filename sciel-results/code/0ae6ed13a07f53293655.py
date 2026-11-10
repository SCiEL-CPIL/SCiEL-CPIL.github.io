import time
import random
import numpy as np

N = 1000
MAXSUM = 2000
DEADLINE = 145.0


def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE
    rng = np.random.RandomState(42)

    def build_counts(S):
        arr = np.zeros(N + 1, dtype=np.int32)
        for x in S:
            arr[x] = 1
        conv = np.convolve(arr, arr, mode='full')
        counts = conv[:MAXSUM + 1].astype(np.int32)
        # For sums where both summands are equal (x = s/2), the diagonal pair is
        # counted once in conv but represents only a<=b when 2x==s. We keep the
        # conv value but must ensure the diagonal (x,x) contributes correctly:
        # conv[s] counts ordered pairs (a,b) with a+b=s. For a<=b we want
        # each unordered pair once. conv counts (a,b) and (b,a) separately for
        # a!=b, so unordered count = (conv[s] + diag[s]) // 2 where diag is 1
        # if s even and s/2 in S.
        diag = np.zeros(MAXSUM + 1, dtype=np.int32)
        for x in S:
            diag[2 * x] = 1
        counts = (conv[:MAXSUM + 1] + diag) // 2
        return counts, arr

    def valid_insert(counts, x):
        # adding x changes representation counts:
        # new pairs (x, y) for y in S with y>=x, plus (x,x).
        # We need to check that no sum exceeds 2 after adding x.
        # Representations involving x: for each y in S, y<=x gives (y,x) -> sum y+x.
        # and (x,x) -> 2x.
        # We'll just simulate using the boolean array.
        pass

    # Fast membership + sum-count check for a candidate x given current counts
    # and membership array. We compute which sums would exceed 2.
    def can_insert(counts, mem, x):
        # sums that involve x and an existing element y (y in S): y + x.
        # For y <= x: sum = x + y (y ranges over S elements <= x)
        # For y > x: sum = x + y (y ranges over S elements > x)
        # Both are just x + y for every y in S. Plus the (x,x) diagonal = 2x.
        # So new representation count for sum s becomes counts[s] + (# of y in S
        # with x + y = s) + (1 if 2x == s else 0).
        # # of y in S with x + y = s is mem[s - x] (if s-x in [1,N]).
        # So for each s, extra = mem[s-x] + (1 if s==2x else 0).
        # We need counts[s] + extra <= 2 for all s.
        # Equivalent: for all y in S, counts[x+y] + 1 <= 2  => counts[x+y] <= 1
        # and counts[2x] + (1 + mem[2x - x]?) ... careful: (x,x) diagonal adds 1,
        # and if there is also y in S with x+y=2x i.e. y=x, that's the same pair.
        # Since y=x only if x in S, but x not in S (candidate), mem[x]=0.
        # So extra for s=2x is mem[2x - x]=mem[x]=0 plus diagonal 1 => extra=1.
        # For s=x+y with y!=x, extra=mem[s-x]=1.
        # Thus condition: for all y in S, counts[x+y] <= 1, and counts[2x] <= 1.
        # (2x check is same as y=x which is not in S, so handle separately.)
        # We need: max over y in S of counts[x+y] <= 1  AND  counts[2x] <= 1.
        # But x+y for y in S: the set of sums is {x+y : y in S}.
        # We can vectorize: for s in range(x+1, x+N+1): if mem[s-x] and counts[s]>1 -> fail.
        # Also check 2x.
        if counts[2 * x] > 1:
            return False
        # check all y in S
        # s = x + y, y in [1,N], y in S
        ys = np.nonzero(mem[1:])[0] + 1
        if ys.size == 0:
            return True
        sums = x + ys
        # sums range x+1 .. x+N
        if sums.min() < 0 or sums.max() > MAXSUM:
            return False
        c = counts[sums]
        return bool(c.max() <= 1)

    def insert(counts, mem, x):
        mem[x] = 1
        # update counts: for each y in S (including x now), new unordered pair.
        # We recompute contributions: for each existing y (before insert) plus x itself.
        # Easier: for y in S_old: counts[x+y]+=1. Then counts[2x]+=1.
        ys = np.nonzero(mem[:x])[0] + 1  # y < x in old set (x not yet set)
        # Actually we set mem[x]=1 already. Recompute old members as mem with x removed.
        pass

    # We'll maintain counts by full recompute on each accepted move? Too slow.
    # Instead incremental:
    def do_insert(counts, mem, x):
        # old members
        old = np.nonzero(mem)[0] + 1
        # pairs (x, y) for y in old: sum x+y, unordered, y can be < x or > x, all distinct sums
        for y in old:
            counts[x + y] += 1
        counts[2 * x] += 1
        mem[x] = 1

    def do_remove(counts, mem, x):
        old = np.nonzero(mem)[0] + 1
        for y in old:
            if y != x:
                counts[x + y] -= 1
        counts[2 * x] -= 1
        mem[x] = 0

    # ---- Initialization strategies ----
    def greedy_init(order):
        mem = np.zeros(N + 1, dtype=np.int32)
        counts = np.zeros(MAXSUM + 1, dtype=np.int32)
        for x in order:
            if can_insert(counts, mem, x):
                do_insert(counts, mem, x)
        return mem, counts

    def random_init(k):
        mem = np.zeros(N + 1, dtype=np.int32)
        counts = np.zeros(MAXSUM + 1, dtype=np.int32)
        pool = rng.permutation(np.arange(1, N + 1))
        added = 0
        for x in pool:
            if added >= k:
                break
            if can_insert(counts, mem, x):
                do_insert(counts, mem, x)
                added += 1
        return mem, counts

    def repair(mem, counts):
        # Remove elements that cause a sum to exceed 2, iteratively.
        # Find all sums > 2, and remove a contributing element.
        changed = True
        while changed:
            changed = False
            bad = np.nonzero(counts > 2)[0]
            if bad.size == 0:
                break
            s = int(bad[0])
            # find pairs (a,b) a<=b in S with a+b=s
            found = False
            for a in range(1, min(s // 2 + 1, N + 1)):
                b = s - a
                if b > N or b < a:
                    continue
                if mem[a] and mem[b]:
                    # remove the larger of a,b (heuristic: remove element that
                    # participates in most bad sums). Just remove b.
                    do_remove(counts, mem, b)
                    found = True
                    changed = True
                    break
            if not found:
                break
        return mem, counts

    def local_search(mem, counts, iters):
        mem = mem.copy()
        counts = counts.copy()
        size = int(mem.sum())
        best_size = size
        best_mem = mem.copy()
        all_vals = np.arange(1, N + 1)
        stall = 0
        for it in range(iters):
            if time.time() > deadline:
                break
            r = rng.random_sample()
            if r < 0.5:
                # insertion
                if size >= N:
                    continue
                # pick a random not-in-set element
                while True:
                    x = int(all_vals[rng.randint(0, N)])
                    if not mem[x]:
                        break
                if can_insert(counts, mem, x):
                    do_insert(counts, mem, x)
                    size += 1
                    if size > best_size:
                        best_size = size
                        best_mem = mem.copy()
                        stall = 0
                    else:
                        stall += 1
            else:
                # swap: remove one in-set, add one not-in-set
                if size == 0:
                    continue
                x = int(all_vals[rng.randint(0, N)])
                if not mem[x]:
                    # just try insertion
                    if can_insert(counts, mem, x):
                        do_insert(counts, mem, x)
                        size += 1
                        if size > best_size:
                            best_size = size
                            best_mem = mem.copy()
                        continue
                # x in set, remove it
                do_remove(counts, mem, x)
                size -= 1
                # try to add a different element
                improved = False
                for _ in range(20):
                    y = int(all_vals[rng.randint(0, N)])
                    if not mem[y] and can_insert(counts, mem, y):
                        do_insert(counts, mem, y)
                        size += 1
                        improved = True
                        break
                if not improved:
                    # put x back
                    do_insert(counts, mem, x)
                    size += 1
                if size > best_size:
                    best_size = size
                    best_mem = mem.copy()
                    stall = 0
                else:
                    stall += 1
            if stall > 400:
                # kick: random perturbation
                for _ in range(3):
                    z = int(all_vals[rng.randint(0, N)])
                    if mem[z]:
                        do_remove(counts, mem, z)
                        size -= 1
                stall = 0
        return best_mem, best_size

    # ---- Run multiple restarts with diverse initializations ----
    best_mem_overall = None
    best_size_overall = 0

    strategies = []
    # ascending greedy
    strategies.append(('greedy_asc', lambda: greedy_init(np.arange(1, N + 1))))
    # descending greedy
    strategies.append(('greedy_desc', lambda: greedy_init(np.arange(N, 0, -1))))
    # random greedy
    strategies.append(('greedy_rand', lambda: greedy_init(rng.permutation(np.arange(1, N + 1)))))
    # dense random init with repair
    for k in (500, 600, 700):
        strategies.append(('rand%d' % k, (lambda kk: random_init(kk))(k)))

    for name, init_fn in strategies:
        if time.time() > deadline - 5:
            break
        mem, counts = init_fn()
        mem, counts = repair(mem, counts)
        mem, size = local_search(mem, counts, iters=200000)
        if size > best_size_overall:
            best_size_overall = size
            best_mem_overall = mem.copy()

    # Final polish: try inserting every remaining element
    if best_mem_overall is not None and time.time() < deadline - 3:
        mem = best_mem_overall.copy()
        counts, _ = build_counts(np.nonzero(mem)[0] + 1)
        for x in range(1, N + 1):
            if time.time() > deadline - 1:
                break
            if not mem[x] and can_insert(counts, mem, x):
                do_insert(counts, mem, x)
        if int(mem.sum()) > best_size_overall:
            best_size_overall = int(mem.sum())
            best_mem_overall = mem

    result = [int(x) for x in np.nonzero(best_mem_overall)[0] + 1] if best_mem_overall is not None else []
    return result


if __name__ == '__main__':
    print(len(construct()))
