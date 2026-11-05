import random
import time

N = 1000
LIMIT = 140.0  # safety margin under 150s wall-clock

def _add_ok(c, x):
    # returns (feasible, touched, new_max) for adding x to current counts c
    # touches 2x and x+y for y in chosen (handled by caller passing list)
    raise NotImplementedError

def _greedy(order, counts, chosen, chosen_set):
    for x in order:
        if x in chosen_set:
            continue
        new = [x + y for y in chosen] + [2 * x]
        ok = True
        for s in new:
            if counts.get(s, 0) + 1 > 2:
                ok = False
                break
        if ok:
            for s in new:
                counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
            chosen_set.add(x)
    return chosen, counts

def _local_search(chosen, counts, chosen_set, rng, deadline):
    # Hill-climb: try to add a removed element, or swap
    # Represent as set; attempt single additions of previously-rejected elements
    rejected = [x for x in range(1, N + 1) if x not in chosen_set]
    # try adding each rejected element if it fits (shouldn't, by construction, but after swaps it might)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in rejected:
            new = [x + y for y in chosen] + [2 * x]
            ok = True
            for s in new:
                if counts.get(s, 0) + 1 > 2:
                    ok = False
                    break
            if ok:
                for s in new:
                    counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
                chosen_set.add(x)
                rejected.remove(x)
                improved = True
    return chosen, counts, chosen_set

def _remove(chosen, counts, chosen_set, x):
    new = [x + y for y in chosen if y != x] + [2 * x]
    for s in new:
        counts[s] -= 1
    chosen.remove(x)
    chosen_set.discard(x)

def construct():
    t0 = time.time()
    deadline = t0 + LIMIT
    rng = random.Random(12345)
    best_size = -1
    best_set = None

    # Deterministic seed greedy (ascending) as baseline
    counts = {}
    chosen = []
    chosen_set = set()
    for x in range(1, N + 1):
        new = [x + y for y in chosen] + [2 * x]
        trial_ok = True
        for s in new:
            if counts.get(s, 0) + 1 > 2:
                trial_ok = False
                break
        if trial_ok:
            for s in new:
                counts[s] = counts.get(s, 0) + 1
            chosen.append(x)
            chosen_set.add(x)
    best_size = len(chosen)
    best_set = set(chosen)

    # Randomized greedy with multiple restarts + local search
    # Use bitset-ish fast check via dict
    def run_randomized(order, deadline):
        counts = {}
        chosen = []
        chosen_set = set()
        for x in order:
            if time.time() > deadline:
                break
            new = [x + y for y in chosen] + [2 * x]
            ok = True
            for s in new:
                if counts.get(s, 0) + 1 > 2:
                    ok = False
                    break
            if ok:
                for s in new:
                    counts[s] = counts.get(s, 0) + 1
                chosen.append(x)
                chosen_set.add(x)
        # local search: try to add any not-chosen element
        notchosen = [x for x in range(1, N + 1) if x not in chosen_set]
        changed = True
        while changed and time.time() < deadline:
            changed = False
            for x in notchosen:
                new = [x + y for y in chosen] + [2 * x]
                ok = True
                for s in new:
                    if counts.get(s, 0) + 1 > 2:
                        ok = False
                        break
                if ok:
                    for s in new:
                        counts[s] = counts.get(s, 0) + 1
                    chosen.append(x)
                    chosen_set.add(x)
                    notchosen.remove(x)
                    changed = True
        # swap-based local search: remove one, try to add two
        if time.time() < deadline:
            for _ in range(3):
                if time.time() > deadline:
                    break
                # try a few random swaps
                cand = list(chosen)
                rng.shuffle(cand)
                for x in cand[:50]:
                    if time.time() > deadline:
                        break
                    # remove x
                    new = [x + y for y in chosen if y != x] + [2 * x]
                    for s in new:
                        counts[s] -= 1
                    chosen.remove(x)
                    chosen_set.discard(x)
                    # try to add up to 2 elements
                    added = 0
                    notchosen2 = [x for x in range(1, N + 1) if x not in chosen_set]
                    for y in notchosen2:
                        if added >= 2:
                            break
                        new2 = [y + z for z in chosen] + [2 * y]
                        ok = True
                        for s in new2:
                            if counts.get(s, 0) + 1 > 2:
                                ok = False
                                break
                        if ok:
                            for s in new2:
                                counts[s] = counts.get(s, 0) + 1
                            chosen.append(y)
                            chosen_set.add(y)
                            added += 1
                    if added < 1:
                        # revert: re-add x
                        new = [x + z for z in chosen] + [2 * x]
                        for s in new:
                            counts[s] = counts.get(s, 0) + 1
                        chosen.append(x)
                        chosen_set.add(x)
        return chosen, counts, chosen_set

    # Try several randomized orderings
    attempts = 0
    while time.time() < deadline - 5.0:
        attempts += 1
        order = list(range(1, N + 1))
        # mix: bias toward spread-out selection using random stride
        stride = rng.randint(2, 4)
        start = rng.randint(0, stride - 1)
        order = [x for x in range(1, N + 1) if (x - 1) % stride == start]
        order += [x for x in range(1, N + 1) if (x - 1) % stride != start]
        rng.shuffle(order)
        chosen, counts, chosen_set = run_randomized(order, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best_set = set(chosen)

    # Final: try to greedily add to best_set if possible
    if time.time() < deadline - 2.0:
        counts = {}
        for x in best_set:
            for y in best_set:
                if x <= y:
                    counts[x + y] = counts.get(x + y, 0) + 1
        # rebuild counts properly
        counts = {}
        for i, x in enumerate(sorted(best_set)):
            for y in sorted(best_set)[i:]:
                counts[x + y] = counts.get(x + y, 0) + 1
        notchosen = [x for x in range(1, N + 1) if x not in best_set]
        for x in notchosen:
            if time.time() > deadline - 2.0:
                break
            new = [x + y for y in best_set] + [2 * x]
            ok = True
            for s in new:
                if counts.get(s, 0) + 1 > 2:
                    ok = False
                    break
            if ok:
                for s in new:
                    counts[s] = counts.get(s, 0) + 1
                best_set.add(x)

    return sorted(best_set)
