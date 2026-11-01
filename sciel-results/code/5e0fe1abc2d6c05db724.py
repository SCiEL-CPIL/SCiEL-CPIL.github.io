import time
import random
import numpy as np

def construct(q, n, radius):
    start = time.time()
    deadline = start + 146.0  # safety margin before 150s kill

    space = q ** n

    def i2w(i):
        w = [0] * n
        for j in range(n - 1, -1, -1):
            w[j] = i % q
            i //= q
        return w

    # Precompute neighbor index matrix (closed ball of radius `radius`) as numpy.
    # For radius 1 (the task case): each word covers itself + 1*n*(q-1) neighbors.
    if radius >= 1:
        # Build via base-q digit manipulation vectorized.
        # positions: for each position p, the multiplier q**(n-1-p)
        pos_mult = np.array([q ** (n - 1 - p) for p in range(n)], dtype=np.int64)
        idx = np.arange(space, dtype=np.int64)
        # For each position p and each new value nv != old value, the neighbor index.
        # old value at p = (idx // pos_mult[p]) % q
        old_vals = (idx[:, None] // pos_mult[None, :]) % q  # shape (space, n)
        # neighbors: for each p, for each nv in 0..q-1, nv != old
        # We'll build a list then stack.
        parts = [idx]  # self
        for p in range(n):
            mult = pos_mult[p]
            ov = old_vals[:, p]
            for nv in range(q):
                mask = (ov != nv)
                if not mask.any():
                    continue
                # neighbor index = idx - ov*mult + nv*mult
                nb = idx[mask] - ov[mask] * mult + nv * mult
                parts.append(nb)
        nbrs = np.concatenate(parts, axis=0)  # (space, ball_size)
        ball_size = nbrs.shape[1]
    else:
        nbrs = np.arange(space, dtype=np.int64).reshape(space, 1)
        ball_size = 1

    # For fast "which codewords cover word w": invert. For each word w, the set of
    # codewords c such that w is in nbrs[c]. Build via scatter.
    # cover_of[w] = boolean mask over codeword-space of size `space`.
    cover_of = np.zeros((space, space), dtype=bool)
    for w in range(space):
        # codewords c where w in ball(c) <=> c in ball(w) (symmetric)
        cover_of[w, nbrs[w]] = True

    def energy_of(pool):
        # pool: list of indices (may have dups). Returns uncovered count.
        if not pool:
            return space
        parr = np.asarray(pool, dtype=np.int64)
        cnt = np.zeros(space, dtype=np.int32)
        cnt[nbrs[parr]] += 1
        return int((cnt <= 0).sum())

    def full_coverage(pool):
        parr = np.asarray(pool, dtype=np.int64)
        cnt = np.zeros(space, dtype=np.int32)
        cnt[nbrs[parr]] += 1
        return bool((cnt > 0).all())

    # Greedy seed: start with a few random, then repeatedly add the word that covers
    # the most currently-uncovered words (sampled over candidates).
    def greedy_seed(rng, target_size=None):
        pool = []
        cnt = np.zeros(space, dtype=np.int32)
        uncovered = np.ones(space, dtype=bool)
        # seed with a couple of random points to break symmetry
        for _ in range(2):
            c = rng.randrange(space)
            if c not in pool:
                pool.append(c)
                cnt[nbrs[c]] += 1
        uncovered = cnt <= 0
        while uncovered.any():
            # candidate set: uncovered words (adding an uncovered word is always useful)
            cand = np.where(uncovered)[0]
            if len(cand) > 120:
                cand = np.array(rng.sample(cand.tolist(), 120))
            best_c = -1
            best_gain = -1
            for c in cand:
                gain = int(uncovered[nbrs[c]].sum())
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            pool.append(best_c)
            cnt[nbrs[best_c]] += 1
            uncovered = cnt <= 0
        return pool

    def greedy_delete(pool, rng):
        # Remove any codeword whose removal keeps full coverage.
        pset = list(pool)
        cnt = np.zeros(space, dtype=np.int32)
        for c in pset:
            cnt[nbrs[c]] += 1
        changed = True
        while changed:
            changed = False
            order = list(pset)
            rng.shuffle(order)
            for c in order:
                if c not in pset:
                    continue
                nb = nbrs[c]
                if (cnt[nb] <= 1).any():
                    continue
                pset.remove(c)
                cnt[nb] -= 1
                changed = True
        return pset

    def simulated_annealing(pool, rng, budget):
        # Fixed-size pool: we keep size constant (add+remove or swap), except we
        # allow removals when they keep coverage (shrinks size).
        pool = list(pool)
        size = len(pool)
        cnt = np.zeros(space, dtype=np.int32)
        for c in pool:
            cnt[nbrs[c]] += 1
        uncovered = int((cnt <= 0).sum())
        best_pool = list(pool)
        best_uncovered = uncovered
        best_size = size

        t = 1.0
        tmin = 0.02
        cool = 0.995
        cur = pool

        def do_add_remove(c_add, c_rem):
            # returns (new_uncovered_delta_ok) ; mutates cnt
            pass

        while time.time() < budget and t > tmin:
            step = rng.random()
            if step < 0.5:
                # swap: remove a random cur element, add a random candidate
                if not cur:
                    break
                ri = rng.randrange(len(cur))
                c_rem = cur[ri]
                # candidate: prefer words that are uncovered or cover uncovered
                if uncovered > 0 and rng.random() < 0.6:
                    uc = np.where(cnt <= 0)[0]
                    if len(uc) > 60:
                        uc = np.array(rng.sample(uc.tolist(), 60))
                    c_add = int(rng.choice(uc))
                else:
                    c_add = rng.randrange(space)
                if c_add == c_rem:
                    t *= cool
                    continue
                # apply
                cnt[nbrs[c_rem]] -= 1
                cnt[nbrs[c_add]] += 1
                new_unc = int((cnt <= 0).sum())
                delta = new_unc - uncovered
                if delta <= 0 or rng.random() < np.exp(-delta / t):
                    cur[ri] = c_add
                    uncovered = new_unc
                    if uncovered < best_uncovered or (uncovered == best_uncovered and len(cur) < best_size):
                        best_uncovered = uncovered
                        best_size = len(cur)
                        best_pool = list(cur)
                        if best_uncovered == 0:
                            # try to shrink
                            shr = greedy_delete(best_pool, rng)
                            if len(shr) < best_size:
                                best_size = len(shr)
                                best_pool = list(shr)
                else:
                    cnt[nbrs[c_rem]] += 1
                    cnt[nbrs[c_add]] -= 1
            elif step < 0.8:
                # removal (shrink) if safe
                if not cur:
                    break
                ri = rng.randrange(len(cur))
                c_rem = cur[ri]
                nb = nbrs[c_rem]
                if (cnt[nb] <= 1).any():
                    t *= cool
                    continue
                cnt[nb] -= 1
                cur.pop(ri)
                new_unc = int((cnt <= 0).sum())
                # removal should not increase uncovered
                if new_unc <= uncovered:
                    uncovered = new_unc
                    if uncovered < best_uncovered or (uncovered == best_uncovered and len(cur) < best_size):
                        best_uncovered = uncovered
                        best_size = len(cur)
                        best_pool = list(cur)
                else:
                    # revert
                    cur.insert(ri, c_rem)
                    cnt[nb] += 1
            else:
                # addition (grow) only if it reduces uncovered
                c_add = rng.randrange(space)
                nb = nbrs[c_add]
                gain = int((cnt <= 0)[nb].sum())
                if gain <= 0:
                    t *= cool
                    continue
                cnt[nb] += 1
                cur.append(c_add)
                new_unc = int((cnt <= 0).sum())
                if new_unc < uncovered:
                    uncovered = new_unc
                    if uncovered < best_uncovered:
                        best_uncovered = uncovered
                        best_pool = list(cur)
                        best_size = len(cur)
                else:
                    cur.pop()
                    cnt[nb] -= 1
            t *= cool
            # periodic greedy delete to keep pool lean
            if int((t * 10000)) % 40 == 0 and time.time() < budget:
                shr = greedy_delete(cur, rng)
                if len(shr) < len(cur):
                    cur = shr
                    cnt = np.zeros(space, dtype=np.int32)
                    for c in cur:
                        cnt[nbrs[c]] += 1
                    uncovered = int((cnt <= 0).sum())
                    if uncovered < best_uncovered or (uncovered == best_uncovered and len(cur) < best_size):
                        best_uncovered = uncovered
                        best_size = len(cur)
                        best_pool = list(cur)
        return best_pool, best_uncovered

    def targeted_fill(pool, rng, budget):
        # Move codewords to cover gaps: for each uncovered word, find a codeword to
        # relocate so that the gap is covered and no new gap is created.
        pool = list(pool)
        cnt = np.zeros(space, dtype=np.int32)
        for c in pool:
            cnt[nbrs[c]] += 1
        pset = set(pool)
        improved = True
        while improved and time.time() < budget:
            improved = False
            unc = np.where(cnt <= 0)[0]
            if len(unc) == 0:
                break
            rng.shuffle(unc.tolist())
            for w in unc[:40]:
                # we need a codeword c such that moving c to a new location x covers w
                # and keeps all covered. Simplest: find x = a neighbor of w (so x covers w)
                # that is not already a codeword, and a codeword c that is non-essential
                # (removing c keeps coverage of its exclusive words) and c != x.
                # Try x over neighbors of w.
                xs = nbrs[w].tolist()
                rng.shuffle(xs)
                done = False
                for x in xs[:12]:
                    if x in pset:
                        continue
                    # find a removable codeword c (non-essential) to replace with x
                    for c in list(pool):
                        if c == x:
                            continue
                        nbc = nbrs[c]
                        if (cnt[nbc] <= 1).any():
                            continue  # essential
                        # simulate: remove c, add x
                        cnt[nbc] -= 1
                        cnt[nbrs[x]] += 1
                        if not (cnt > 0).all():
                            cnt[nbc] += 1
                            cnt[nbrs[x]] -= 1
                            continue
                        # commit
                        pool[pool.index(c)] = x
                        pset.discard(c)
                        pset.add(x)
                        improved = True
                        done = True
                        break
                    if done:
                        break
                if done:
                    # recompute to be safe
                    cnt = np.zeros(space, dtype=np.int32)
                    for c in pool:
                        cnt[nbrs[c]] += 1
        return pool

    # ---- Main search ----
    rng0 = random.Random(12345)
    # Initial seed
    pool = greedy_seed(rng0)
    pool = greedy_delete(pool, rng0)
    best = list(pool)
    best_len = len(pool)

    # SA rounds with different temperatures/seeds, each within a sub-budget
    remaining = deadline - time.time()
    n_rounds = max(1, int(remaining // 12))
    for r in range(n_rounds):
        if time.time() > deadline - 8:
            break
        rng = random.Random(1000 + r)
        sub_budget = deadline - 10
        # start SA from current best
        cand, unc = simulated_annealing(list(best), rng, sub_budget)
        cand = greedy_delete(cand, rng)
        cand = targeted_fill(cand, rng, sub_budget)
        cand = greedy_delete(cand, rng)
        if full_coverage(cand) and len(cand) < best_len:
            best = list(cand)
            best_len = len(cand)
        elif full_coverage(cand) and len(cand) == best_len:
            # keep as alternative for next round diversity
            pass

    # Final polish
    if time.time() < deadline - 6:
        rng = random.Random(777)
        best = greedy_delete(best, rng)
        best = targeted_fill(best, rng, deadline - 6)
        best = greedy_delete(best, rng)
        if full_coverage(best):
            best_len = len(best)

    if not best:
        # fallback: full space (valid but huge)
        best = list(range(space))

    return [i2w(i) for i in best]
