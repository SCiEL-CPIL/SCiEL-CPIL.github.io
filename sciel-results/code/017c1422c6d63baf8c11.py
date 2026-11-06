"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    t0 = time.time()
    DEADLINE = t0 + 85.0  # safety margin before 90s kill

    N = 256
    ALL = (1 << N) - 1

    # Precompute conflict_mask[i]: bitmask of j such that (i+1,j+1) conflict
    conflict_mask = [0] * N
    for i in range(N):
        a = i + 1
        m = 0
        for j in range(N):
            if i == j:
                continue
            b = j + 1
            if (a * b) % (a + b) == 0:
                m |= (1 << j)
        conflict_mask[i] = m

    odds_indices = [i for i in range(N) if (i + 1) % 2 == 1]
    evens_indices = [i for i in range(N) if (i + 1) % 2 == 0]

    # base = all odds (mutually compatible: odd*odd odd, odd+odd even)
    base_mask = 0
    for i in odds_indices:
        base_mask |= (1 << i)

    def popcount(x):
        return x.bit_count()

    def greedy_add(start_mask, order, deadline):
        cur = start_mask
        for e in order:
            if time.time() > deadline:
                break
            if (cur >> e) & 1:
                continue
            if (conflict_mask[e] & cur) == 0:
                cur |= (1 << e)
        return cur

    def greedy_complete(mask, deadline, max_rounds=200):
        """Repeatedly scan all candidates and add any that fit, until no change."""
        cur = mask
        for _ in range(max_rounds):
            if time.time() > deadline:
                break
            changed = False
            not_in = ALL & ~cur
            while not_in:
                if time.time() > deadline:
                    break
                lsb = not_in & (-not_in)
                e = lsb.bit_length() - 1
                not_in ^= lsb
                if (conflict_mask[e] & cur) == 0:
                    cur |= lsb
                    changed = True
            if not changed:
                break
        return cur

    def local_search(start_mask, deadline, max_iters=200000):
        best_mask = start_mask
        best_size = popcount(start_mask)
        cur = start_mask
        cur_size = best_size
        rng = random.Random(12345)

        for _ in range(max_iters):
            if time.time() > deadline:
                break

            r = rng.random()
            if r < 0.6:
                # ADD a random compatible element
                not_in = ALL & ~cur
                if not_in == 0:
                    continue
                # pick a random set bit
                k = rng.randrange(cur_size, N)  # not directly usable; use random index
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                if (conflict_mask[e] & cur) == 0:
                    cur |= (1 << e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            elif r < 0.85:
                # 1-SWAP: remove a conflicting element of e, add e
                e = rng.randrange(N)
                if (cur >> e) & 1:
                    continue
                conflicts = conflict_mask[e] & cur
                if conflicts == 0:
                    cur |= (1 << e)
                    cur_size += 1
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
                    continue
                # pick a random conflicting element to remove
                k = rng.randrange(popcount(conflicts))
                c = None
                cc = conflicts
                for _ in range(k + 1):
                    lsb = cc & (-cc)
                    c = lsb.bit_length() - 1
                    cc ^= lsb
                new_mask = (cur & ~(1 << c)) | (1 << e)
                # validate: e must not conflict with new_mask minus e
                if (conflict_mask[e] & (new_mask & ~(1 << e))) == 0:
                    # also ensure removing c didn't break anything (it only helps)
                    cur = new_mask
                    # size unchanged; but try to complete
                    cur = greedy_complete(cur, deadline, max_rounds=3)
                    cur_size = popcount(cur)
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur
            else:
                # 2-REMOVE + complete: drop 1-2 random elements, then greedy complete
                drop = 1 if rng.random() < 0.5 else 2
                trial = cur
                for _ in range(drop):
                    if trial == 0:
                        break
                    e = rng.randrange(N)
                    if (trial >> e) & 1:
                        trial &= ~(1 << e)
                trial = greedy_complete(trial, deadline, max_rounds=5)
                ts = popcount(trial)
                if ts > cur_size:
                    cur = trial
                    cur_size = ts
                    if cur_size > best_size:
                        best_size = cur_size
                        best_mask = cur

        return best_mask

    # ---- Multi-start diverse greedy ----
    best_overall = base_mask
    best_size = popcount(base_mask)

    def consider(mask):
        nonlocal best_overall, best_size
        sz = popcount(mask)
        if sz > best_size:
            best_overall = mask
            best_size = sz

    # 1) natural order
    consider(greedy_complete(greedy_add(base_mask, evens_indices, DEADLINE), DEADLINE))
    # 2) reverse order
    consider(greedy_complete(greedy_add(base_mask, list(reversed(evens_indices)), DEADLINE), DEADLINE))
    # 3) ascending by conflict-degree with odds (low-degree first)
    deg = [(popcount(conflict_mask[e] & base_mask), e) for e in evens_indices]
    deg.sort()
    consider(greedy_complete(greedy_add(base_mask, [e for _, e in deg], DEADLINE), DEADLINE))
    # 4) descending degree
    deg.sort(key=lambda x: -x[0])
    consider(greedy_complete(greedy_add(base_mask, [e for _, e in deg], DEADLINE), DEADLINE))

    # 5) many random orderings
    for attempt in range(60):
        if time.time() > DEADLINE - 20:
            break
        order = evens_indices[:]
        random.shuffle(order)
        consider(greedy_complete(greedy_add(base_mask, order, DEADLINE), DEADLINE))

    # ---- Local search from best, with several restarts from diverse starts ----
    if time.time() < DEADLINE - 15:
        improved = local_search(best_overall, DEADLINE, max_iters=200000)
        consider(improved)

    # A couple of extra local-search restarts from other good starts if time allows
    extra_starts = []
    for attempt in range(5):
        if time.time() > DEADLINE - 10:
            break
        order = evens_indices[:]
        random.shuffle(order)
        m = greedy_complete(greedy_add(base_mask, order, DEADLINE), DEADLINE)
        if popcount(m) >= best_size - 2:
            extra_starts.append(m)

    for m in extra_starts:
        if time.time() > DEADLINE - 5:
            break
        improved = local_search(m, DEADLINE, max_iters=30000)
        consider(improved)

    result = sorted([i + 1 for i in range(N) if (best_overall >> i) & 1])

    # Final verification; fall back to odds if any conflict slips through
    for i in range(len(result)):
        for j in range(i + 1, len(result)):
            a, b = result[i], result[j]
            if (a * b) % (a + b) == 0:
                return sorted([i + 1 for i in odds_indices])

    return result
# EVOLVE-BLOCK-END
