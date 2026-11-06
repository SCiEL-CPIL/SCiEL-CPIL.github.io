import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best = []
    best_len = 0

    def can_append(seq, sums, x):
        new_sums = set()
        s = x
        new_sums.add(s)
        for a in reversed(seq):
            s += a
            new_sums.add(s)
        if new_sums & sums:
            return None
        return new_sums

    def get_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return None
                sums.add(s)
        return sums

    def update_best(seq):
        nonlocal best, best_len
        if len(seq) > best_len:
            best = list(seq)
            best_len = len(seq)

    def dfs(seq, sums, min_val, max_val, deadline, depth=0, max_branch=3):
        if time.time() > deadline:
            return
        update_best(seq)
        if depth >= 15:
            return
        tried = 0
        for x in range(min_val, max_val + 1):
            if time.time() > deadline:
                return
            ns = can_append(seq, sums, x)
            if ns is not None:
                seq.append(x)
                sums.update(ns)
                dfs(seq, sums, x + 1, max_val, deadline, depth + 1, max_branch)
                seq.pop()
                for s in ns:
                    sums.discard(s)
                tried += 1
                if tried >= max_branch:
                    return

    def greedy_extend(seq, sums, min_val, max_val, deadline, direction='smallest'):
        nonlocal best, best_len
        while time.time() < deadline:
            added = False
            if direction == 'smallest':
                rng = range(min_val, max_val + 1)
            else:
                rng = range(max_val, min_val - 1, -1)
            for x in rng:
                ns = can_append(seq, sums, x)
                if ns is not None:
                    seq.append(x)
                    sums.update(ns)
                    update_best(seq)
                    if direction == 'smallest':
                        min_val = x + 1
                    else:
                        max_val = x - 1
                    added = True
                    break
            if not added:
                break

    base = [1, 2, 4, 8, 16, 32, 64, 128]
    base_sums = get_sums(base)

    # Phase 1: Greedy extend from power-of-2 base
    if base_sums:
        seq = list(base)
        update_best(seq)
        greedy_extend(seq, set(base_sums), 129, 256, deadline, 'smallest')

    # Phase 2: DFS from power-of-2 (branch=3)
    if base_sums and time.time() < deadline - 10:
        dfs(list(base), set(base_sums), 129, 256, deadline, max_branch=3)

    # Phase 3: DFS from power-of-2 (branch=5)
    if base_sums and time.time() < deadline - 10:
        dfs(list(base), set(base_sums), 129, 256, deadline, max_branch=5)

    # Phase 4: Other starting bases
    other_bases = [
        [1, 2, 4, 8, 16, 32, 64],
        [1, 2, 4, 8, 16, 32, 63],
        [1, 2, 4, 8, 16, 33, 64, 128],
        [1, 2, 4, 8, 16, 32, 65, 128],
    ]
    for b in other_bases:
        if time.time() > deadline - 5:
            break
        s = get_sums(b)
        if s is None:
            continue
        seq = list(b)
        update_best(seq)
        greedy_extend(seq, set(s), b[-1] + 1, 256, deadline, 'smallest')

    # Phase 5: Local search - trim and re-extend
    if best and time.time() < deadline - 5:
        for trim in range(1, 5):
            if time.time() > deadline - 2:
                break
            if trim >= len(best):
                continue
            trial = list(best[:-trim])
            s = get_sums(trial)
            if s is None:
                continue
            greedy_extend(trial, set(s), trial[-1] + 1, 256, deadline, 'smallest')

    # Phase 6: Randomized restarts
    for seed in range(30):
        if time.time() > deadline - 3:
            break
        random.seed(seed * 31 + 7)
        seq = []
        sums = set()
        min_val = 1
        for _ in range(200):
            if time.time() > deadline:
                break
            candidates = list(range(min_val, 257))
            random.shuffle(candidates)
            added = False
            for x in candidates:
                ns = can_append(seq, sums, x)
                if ns is not None:
                    seq.append(x)
                    sums.update(ns)
                    update_best(seq)
                    min_val = x + 1
                    added = True
                    break
            if not added:
                break

    return best
