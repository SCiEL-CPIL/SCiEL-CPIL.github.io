import time, random

def construct():
    t0 = time.time()
    DEADLINE = 110.0  # safety margin vs 120s (evaluator kills at 150s, but keep margin)

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def solve(S):
        n = len(S)
        val_to_idx = {v: i for i, v in enumerate(S)}
        # sums[k]: list of (i,j), i<=j, S[i]+S[j]==S[k]
        sums = [[] for _ in range(n)]
        # neighbors[i]: list of (j,k) with S[i]+S[j]==S[k]
        neighbors = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in val_to_idx:
                    k = val_to_idx[s]
                    sums[k].append((i, j))
                    # i as summand
                    if i != j:
                        neighbors[i].append((j, k))
                        neighbors[j].append((i, k))
                    else:
                        neighbors[i].append((j, k))  # j==i
        # state
        in_ = [False]*n
        cc = [0]*n   # conflict count as sum
        sc = [0]*n   # conflict count as summand
        count = 0
        total_conf = 0

        def add(i):
            nonlocal count, total_conf
            # as summand
            for (j, k) in neighbors[i]:
                if in_[j] and in_[k]:
                    cc[k] += 1
                    sc[j] += 1
                    total_conf += 1
            # as sum
            for (a, b) in sums[i]:
                if in_[a] and in_[b]:
                    cc[i] += 1
                    sc[a] += 1
                    sc[b] += 1
                    total_conf += 1
            in_[i] = True
            count += 1

        def remove(i):
            nonlocal count, total_conf
            for (j, k) in neighbors[i]:
                if in_[j] and in_[k]:
                    cc[k] -= 1
                    sc[j] -= 1
                    total_conf -= 1
            for (a, b) in sums[i]:
                if in_[a] and in_[b]:
                    cc[i] -= 1
                    sc[a] -= 1
                    sc[b] -= 1
                    total_conf -= 1
            in_[i] = False
            count -= 1

        def is_sum_free():
            return total_conf == 0

        def current_count():
            return count

        # start from odds
        for i in range(n):
            if S[i] % 2:
                add(i)

        # greedy add
        changed = True
        while changed:
            changed = False
            for i in range(n):
                if not in_[i]:
                    # check if adding i creates conflict
                    ok = True
                    for (j, k) in neighbors[i]:
                        if in_[j] and in_[k]:
                            ok = False; break
                    if ok:
                        for (a, b) in sums[i]:
                            if in_[a] and in_[b]:
                                ok = False; break
                    if ok:
                        add(i); changed = True

        best = list(in_)
        bestc = count

        # local search with swaps + re-adds
        rng = random.Random(12345)
        idx = list(range(n))
        steps = 0
        while time.time() - t0 < DEADLINE - 2.0:
            # try single add
            improved = False
            for i in idx:
                if not in_[i]:
                    ok = True
                    for (j, k) in neighbors[i]:
                        if in_[j] and in_[k]:
                            ok = False; break
                    if ok:
                        for (a, b) in sums[i]:
                            if in_[a] and in_[b]:
                                ok = False; break
                    if ok:
                        add(i); improved = True
            if improved:
                if count > bestc:
                    bestc = count; best = list(in_)
                continue
            # try swap: remove a, add b
            done = False
            for _ in range(2000):
                a = rng.choice(idx)
                b = rng.choice(idx)
                if in_[a] and not in_[b]:
                    # simulate
                    remove(a)
                    # try add b
                    ok = True
                    for (j, k) in neighbors[b]:
                        if in_[j] and in_[k]:
                            ok = False; break
                    if ok:
                        for (x, y) in sums[b]:
                            if in_[x] and in_[y]:
                                ok = False; break
                    if ok:
                        add(b)
                        # greedy re-add
                        for i in idx:
                            if not in_[i]:
                                o2 = True
                                for (j, k) in neighbors[i]:
                                    if in_[j] and in_[k]:
                                        o2 = False; break
                                if o2:
                                    for (x, y) in sums[i]:
                                        if in_[x] and in_[y]:
                                            o2 = False; break
                                if o2:
                                    add(i)
                        if count > bestc:
                            bestc = count; best = list(in_)
                        done = True
                    else:
                        add(a)  # revert
                    if done:
                        break
            if not done:
                # kick: remove a random element that's in a conflict, then re-add greedily
                # find an element in a conflict
                cands = [i for i in idx if in_[i] and (cc[i] > 0 or sc[i] > 0)]
                if not cands:
                    # stuck, break or random restart
                    break
                a = rng.choice(cands)
                remove(a)
                for i in idx:
                    if not in_[i]:
                        o2 = True
                        for (j, k) in neighbors[i]:
                            if in_[j] and in_[k]:
                                o2 = False; break
                        if o2:
                            for (x, y) in sums[i]:
                                if in_[x] and in_[y]:
                                    o2 = False; break
                        if o2:
                            add(i)
                if count > bestc:
                    bestc = count; best = list(in_)

        # rebuild from best
        return [S[i] for i in range(n) if best[i]]

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        results.append(solve(S))
    return results
