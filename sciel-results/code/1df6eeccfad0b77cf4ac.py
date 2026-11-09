"""Combined: graph-based structural optimization + simulated annealing refinement for small maximal Sidon sets in [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 57  # 57s safety margin (evaluator kills at 60s)

# EVOLVE-BLOCK-START

def _build_sum_set(selected):
    """Build the set of all unordered pair sums (including diagonal) for a Sidon set."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _is_sidon(selected):
    """Check if selected is a Sidon set (all unordered sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            s = selected[i] + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected, sums):
    """Check inclusion maximality: no missing element can be added without breaking Sidon."""
    sset = set(selected)
    for x in range(1, N + 1):
        if x in sset:
            continue
        new_sums = [2 * x] + [x + a for a in selected]
        if any(s in sums for s in new_sums):
            continue
        if len(set(new_sums)) != len(new_sums):
            continue
        return False  # x can be added, so not maximal
    return True

def _greedy_randomize(perm):
    """Greedy: process elements in given order, add if keeps Sidon property."""
    selected = []
    sums = set()
    for x in perm:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected, sums

def _try_remove_fast(selected, sums, idx):
    """Remove element at idx; return (new_sel, new_sums). Removing never breaks Sidon."""
    new_sel = selected[:idx] + selected[idx+1:]
    new_sums = set()
    for i in range(len(new_sel)):
        for j in range(i, len(new_sel)):
            new_sums.add(new_sel[i] + new_sel[j])
    return new_sel, new_sums

def _add_to_set(selected, sums, x):
    """Try adding x; return (new_sel, new_sums) if still Sidon, else None."""
    added = [2 * x] + [x + a for a in selected]
    if len(set(added)) != len(added):
        return None
    if sums.intersection(added):
        return None
    new_sel = selected + [x]
    new_sums = sums.union(added)
    return new_sel, new_sums

def _local_search(selected, sums, rng, time_limit):
    """Simulated annealing local search: removals, swaps, and add-then-remove moves."""
    best = (len(selected), list(selected), set(sums))
    cur_sel, cur_sums = list(selected), set(sums)
    start = time.time()
    T = 1.0
    T_min = 0.01
    cooling = 0.995

    while time.time() - start < time_limit and time.time() < DEADLINE:
        if T < T_min:
            T = T_min
        r = rng.random()
        if r < 0.4:
            # Removal move
            if len(cur_sel) <= 1:
                T *= cooling
                continue
            i = rng.randrange(len(cur_sel))
            new_sel, new_sums = _try_remove_fast(cur_sel, cur_sums, i)
            if new_sel is None:
                T *= cooling
                continue
            new_max = _is_maximal(new_sel, new_sums)
            cur_max = _is_maximal(cur_sel, cur_sums)
            new_cost = len(new_sel) + (0 if new_max else 100)
            cur_cost = len(cur_sel) + (0 if cur_max else 100)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                cur_sel, cur_sums = new_sel, new_sums
                if new_max and len(cur_sel) < best[0]:
                    best = (len(cur_sel), list(cur_sel), set(cur_sums))
        elif r < 0.7:
            # Swap: remove one, add another
            if len(cur_sel) <= 1:
                T *= cooling
                continue
            i = rng.randrange(len(cur_sel))
            new_sel = cur_sel[:i] + cur_sel[i+1:]
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    new_sums.add(new_sel[a] + new_sel[b])
            candidates = [x for x in range(1, N+1) if x not in new_sel]
            if not candidates:
                T *= cooling
                continue
            cand = rng.choice(candidates)
            res = _add_to_set(new_sel, new_sums, cand)
            if res is None:
                T *= cooling
                continue
            new_sel2, new_sums2 = res
            new_max = _is_maximal(new_sel2, new_sums2)
            cur_max = _is_maximal(cur_sel, cur_sums)
            new_cost = len(new_sel2) + (0 if new_max else 100)
            cur_cost = len(cur_sel) + (0 if cur_max else 100)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                cur_sel, cur_sums = new_sel2, new_sums2
                if new_max and len(cur_sel) < best[0]:
                    best = (len(cur_sel), list(cur_sel), set(cur_sums))
        else:
            # Add-then-remove
            if len(cur_sel) <= 1:
                T *= cooling
                continue
            candidates = [x for x in range(1, N+1) if x not in cur_sel]
            if not candidates:
                T *= cooling
                continue
            cand = rng.choice(candidates)
            added = [2*cand] + [cand+a for a in cur_sel]
            if len(set(added)) != len(added) or cur_sums.intersection(added):
                T *= cooling
                continue
            i = rng.randrange(len(cur_sel))
            new_sel = cur_sel[:i] + cur_sel[i+1:] + [cand]
            new_sums = set()
            for a in range(len(new_sel)):
                for b in range(a, len(new_sel)):
                    new_sums.add(new_sel[a] + new_sel[b])
            new_max = _is_maximal(new_sel, new_sums)
            cur_max = _is_maximal(cur_sel, cur_sums)
            new_cost = len(new_sel) + (0 if new_max else 100)
            cur_cost = len(cur_sel) + (0 if cur_max else 100)
            delta = new_cost - cur_cost
            if delta <= 0 or rng.random() < np.exp(-delta / max(T, 1e-9)):
                cur_sel, cur_sums = new_sel, new_sums
                if new_max and len(cur_sel) < best[0]:
                    best = (len(cur_sel), list(cur_sel), set(cur_sums))
        T *= cooling
    return best

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    start = time.time()
    iterations = 0

    while time.time() < DEADLINE - 2:
        iterations += 1
        perm = list(range(1, N + 1))
        rng.shuffle(perm)

        selected, sums = _greedy_randomize(perm)

        if len(selected) >= best_size:
            continue

        # Verify validity
        if not _is_sidon(selected):
            continue

        # Check maximality
        if not _is_maximal(selected, sums):
            continue

        best_size = len(selected)
        best_result = list(selected)

        # Try local search to improve
        remaining = DEADLINE - time.time()
        if remaining > 5:
            t_limit = min(3.0, remaining - 5)
            improved = _local_search(selected, sums, rng, t_limit)
            if improved[0] < best_size:
                best_size = improved[0]
                best_result = list(improved[1])

        # If we found a very small set, do a few more then stop
        if best_size <= 16 and iterations > 50:
            break

        if time.time() > DEADLINE - 3:
            break

    if best_result is None:
        # Fallback: deterministic greedy
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        best_result = selected
        best_size = len(selected)

    return best_result
# EVOLVE-BLOCK-END
