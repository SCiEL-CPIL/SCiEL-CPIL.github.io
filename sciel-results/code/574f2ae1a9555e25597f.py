"""New design: randomized greedy + local search (add/remove/swap) with deadline guard."""
import time
import random

_DEADLINE = 85.0  # safety margin before 90s kill


def _valid(seq):
    """Check that all nonempty consecutive-block sums are distinct."""
    n = len(seq)
    if n == 0:
        return True
    sums = set()
    for i in range(n):
        s = 0
        for j in range(i, n):
            s += seq[j]
            if s in sums:
                return False
            sums.add(s)
    return True


def _greedy(values, used_set, deadline):
    """Continue building from existing list using greedy next-element selection."""
    while True:
        if time.time() >= deadline:
            break
        best_candidate = None
        best_score = -1
        for x in range(1, 257):
            if x in used_set:
                continue
            # Check if adding x creates a sum collision
            new_sums = set()
            ok = True
            for i in range(len(values)):
                s = 0
                for j in range(i, len(values) + 1):
                    if j < len(values):
                        s += values[j]
                    else:
                        s += x
                    if s in sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
                if not ok:
                    break
            if ok:
                # Prefer larger values to fill the range efficiently
                score = x
                if score > best_score:
                    best_score = score
                    best_candidate = x
        if best_candidate is None:
            break
        values.append(best_candidate)
        used_set.add(best_candidate)
        # Update sums
        for i in range(len(values)):
            s = 0
            for j in range(i, len(values)):
                s += values[j]
                sums.add(s)
    return values


def _greedy_largest_start(deadline):
    """Greedy starting from largest available values first."""
    values = []
    sums = set()
    used = set()
    for x in range(256, 0, -1):
        if time.time() >= deadline:
            break
        new_sums = set()
        ok = True
        for i in range(len(values)):
            s = 0
            for j in range(i, len(values) + 1):
                if j < len(values):
                    s += values[j]
                else:
                    s += x
                if s in sums or s in new_sums:
                    ok = False
                    break
                new_sums.add(s)
            if not ok:
                break
        if ok:
            values.append(x)
            used.add(x)
            sums.update(new_sums)
    # Sort to make strictly increasing
    values.sort()
    return values


def _local_search(seq, deadline):
    """Local search with add, remove, and swap moves."""
    best_seq = list(seq)
    best_len = len(best_seq)
    used = set(best_seq)
    sums = set()
    for i in range(len(best_seq)):
        s = 0
        for j in range(i, len(best_seq)):
            s += best_seq[j]
            sums.add(s)

    random.seed(hash(tuple(best_seq)) % (2**31))

    while time.time() < deadline:
        improved = False
        # Try adding a random unused element
        unused = [x for x in range(1, 257) if x not in used]
        if unused:
            for _ in range(min(len(unused), 50)):
                x = random.choice(unused)
                # Try inserting at each position
                for pos in range(len(best_seq) + 1):
                    new_seq = best_seq[:pos] + [x] + best_seq[pos:]
                    if new_seq != sorted(new_seq):
                        continue
                    if _valid(new_seq):
                        best_seq = new_seq
                        used.add(x)
                        improved = True
                        break
                if improved:
                    break

        if improved:
            # Recompute sums
            sums = set()
            for i in range(len(best_seq)):
                s = 0
                for j in range(i, len(best_seq)):
                    s += best_seq[j]
                    sums.add(s)
            if len(best_seq) > best_len:
                best_len = len(best_seq)
            continue

        # Try removing an element and re-adding a different one
        if len(best_seq) > 1:
            for _ in range(min(len(best_seq), 20)):
                idx = random.randrange(len(best_seq))
                removed = best_seq[idx]
                trial = best_seq[:idx] + best_seq[idx+1:]
                if _valid(trial):
                    # Try adding back a random unused value
                    unused2 = [x for x in range(1, 257) if x not in set(trial)]
                    if unused2:
                        for x in random.sample(unused2, min(len(unused2), 20)):
                            for pos in range(len(trial) + 1):
                                new_seq = trial[:pos] + [x] + trial[pos:]
                                if new_seq == sorted(new_seq) and _valid(new_seq):
                                    best_seq = new_seq
                                    improved = True
                                    break
                            if improved:
                                break
                    if improved:
                        break

        if improved:
            if len(best_seq) > best_len:
                best_len = len(best_seq)
            continue

        # Try swap
        if len(best_seq) > 1:
            for _ in range(min(len(best_seq), 10)):
                i = random.randrange(len(best_seq))
                j = random.randrange(len(best_seq))
                if i == j:
                    continue
                trial = list(best_seq)
                trial[i], trial[j] = trial[j], trial[i]
                if trial == sorted(trial) and _valid(trial):
                    best_seq = trial
                    improved = True
                    break

        if not improved:
            break

    return best_seq


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE

    best_seq = []
    best_len = 0

    # Phase 1: Multiple greedy initializations with different seeds/orders
    strategies = [
        lambda: _greedy_largest_start(deadline),
    ]

    # Standard greedy (smallest first)
    def greedy_smallest():
        values = []
        sums = set()
        for x in range(1, 257):
            if time.time() >= deadline:
                break
            new_sums = set()
            ok = True
            for i in range(len(values)):
                s = 0
                for j in range(i, len(values) + 1):
                    if j < len(values):
                        s += values[j]
                    else:
                        s += x
                    if s in sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
            if ok:
                values.append(x)
                sums.update(new_sums)
        return values

    strategies.append(greedy_smallest)

    # Randomized greedy
    def greedy_random(seed_val):
        rng = random.Random(seed_val)
        values = []
        sums = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        for x in candidates:
            if time.time() >= deadline:
                break
            new_sums = set()
            ok = True
            for i in range(len(values)):
                s = 0
                for j in range(i, len(values) + 1):
                    if j < len(values):
                        s += values[j]
                    else:
                        s += x
                    if s in sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
            if ok:
                values.append(x)
                sums.update(new_sums)
        values.sort()
        return values

    for s in range(10):
        strategies.append(lambda s=s: greedy_random(s * 1000 + 42))

    # Run all strategies, keep best
    for strat in strategies:
        if time.time() >= deadline:
            break
        seq = strat()
        if len(seq) > best_len:
            best_len = len(seq)
            best_seq = seq

    # Phase 2: Local search on best sequence and a few others
    candidates_for_ls = [best_seq]
    # Add a couple more greedy results
    for s in range(3):
        if time.time() >= deadline:
            break
        seq = greedy_random(9000 + s * 777)
        if len(seq) >= best_len - 1:
            candidates_for_ls.append(seq)

    for seq in candidates_for_ls:
        if time.time() >= deadline:
            break
        improved = _local_search(seq, deadline)
        if len(improved) > best_len:
            best_len = len(improved)
            best_seq = improved

    # Final greedy polish on best_seq
    if time.time() < deadline:
        used = set(best_seq)
        sums = set()
        for i in range(len(best_seq)):
            s = 0
            for j in range(i, len(best_seq)):
                s += best_seq[j]
                sums.add(s)
        for x in range(1, 257):
            if time.time() >= deadline:
                break
            if x in used:
                continue
            new_sums = set()
            ok = True
            for i in range(len(best_seq)):
                s = 0
                for j in range(i, len(best_seq) + 1):
                    if j < len(best_seq):
                        s += best_seq[j]
                    else:
                        s += x
                    if s in sums or s in new_sums:
                        ok = False
                        break
                    new_sums.add(s)
            if ok:
                best_seq.append(x)
                used.add(x)
                sums.update(new_sums)
                best_len += 1

    # Ensure strictly increasing
    best_seq.sort()

    # Final validity check
    if not _valid(best_seq):
        # Fallback: pure greedy smallest
        best_seq = greedy_smallest()

    return best_seq
