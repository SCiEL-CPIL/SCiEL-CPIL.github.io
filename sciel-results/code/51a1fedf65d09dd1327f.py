"""B2[2] set in 1..1000: maximize size under 'every sum has <=2 reps'."""
import time
import numpy as np

MAXN = 1000
LIMIT = 2  # max representations allowed


def _counts_from(arr):
    """Return dict sum->count for an integer array of sums."""
    if arr.size == 0:
        return {}
    u, c = np.unique(arr, return_counts=True)
    return {int(u[i]): int(c[i]) for i in range(len(u))}


def _new_sums(x, chosen):
    """All new sums introduced by adding x to chosen (a<=b)."""
    if chosen.size == 0:
        return np.array([2 * x], dtype=np.int64)
    return np.concatenate([chosen + x, np.array([2 * x], dtype=np.int64)])


def _valid_add(x, chosen, counts):
    """Check if adding x keeps every sum <= LIMIT. Returns (ok, new_sums, new_counts)."""
    ns = _new_sums(x, chosen)
    if ns.size == 0:
        return True, ns, counts
    u, c = np.unique(ns, return_counts=True)
    for i in range(len(u)):
        s = int(u[i])
        if counts.get(s, 0) + int(c[i]) > LIMIT:
            return False, ns, None
    new_counts = dict(counts)
    for i in range(len(u)):
        s = int(u[i])
        new_counts[s] = new_counts.get(s, 0) + int(c[i])
    return True, ns, new_counts


def _sidon_greedy(lo, hi, step, start_offset, deadline):
    """Greedy Sidon set within an arithmetic progression lo + k*step for k=0..K."""
    chosen = []
    counts = {}
    k = 0
    while lo + k * step <= hi:
        if time.time() > deadline:
            break
        x = lo + k * step
        # Sidon: every sum has <=1 rep within this progression.
        ns = []
        for y in chosen:
            ns.append(x + y)
        ns.append(2 * x)
        ok = True
        for s in ns:
            if counts.get(s, 0) + 1 > 1:
                ok = False
                break
        if ok:
            chosen.append(x)
            for s in ns:
                counts[s] = counts.get(s, 0) + 1
        k += 1
    return chosen


def _mod3_seed(deadline):
    """Build a B2[2] seed: union of 3 Sidon sets, one per residue mod 3."""
    all_chosen = []
    for r in (1, 2, 0):
        # residue r in 1..1000
        vals = list(range(r if r != 0 else 3, 1001, 3))
        if not vals:
            continue
        # Greedy Sidon on this residue class.
        chosen = []
        counts = {}
        for x in vals:
            if time.time() > deadline:
                break
            ns = [x + y for y in chosen] + [2 * x]
            ok = True
            for s in ns:
                if counts.get(s, 0) + 1 > 1:
                    ok = False
                    break
            if ok:
                chosen.append(x)
                for s in ns:
                    counts[s] = counts.get(s, 0) + 1
        all_chosen.extend(chosen)
    return all_chosen


def _local_search(initial, deadline, rng):
    """Local search: add any feasible missing element; if blocked, try swaps."""
    chosen = sorted(initial)
    chosen_arr = np.array(chosen, dtype=np.int64)
    # Build counts.
    counts = {}
    for i, a in enumerate(chosen):
        for b in chosen[i:]:
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    size = len(chosen)
    best_size = size
    best_set = list(chosen)

    # Candidate pool: all 1..1000 not in chosen.
    present = set(chosen)
    candidates = [x for x in range(1, MAXN + 1) if x not in present]

    # Shuffle candidates for diversity.
    rng.shuffle(candidates)

    progress = True
    while progress and time.time() < deadline:
        progress = False
        # Try adds in shuffled order.
        for x in candidates[:]:
            if time.time() > deadline:
                break
            if x in present:
                continue
            ok, ns, new_counts = _valid_add(x, chosen_arr, counts)
            if ok:
                chosen.append(x)
                chosen.sort()
                chosen_arr = np.array(chosen, dtype=np.int64)
                counts = new_counts
                present.add(x)
                candidates.remove(x)
                size += 1
                if size > best_size:
                    best_size = size
                    best_set = list(chosen)
                progress = True
        if time.time() > deadline:
            break
        # Try swaps: remove one, add one (size-neutral), only if it enables later adds.
        # Heuristic: for each candidate x that is blocked, try to find y in chosen
        # whose removal unblocks x.
        for x in candidates:
            if time.time() > deadline:
                break
            if x in present:
                continue
            ok, ns, _ = _valid_add(x, chosen_arr, counts)
            if ok:
                continue
            # Find sums that would overflow.
            u, c = np.unique(ns, return_counts=True)
            bad = []
            for i in range(len(u)):
                s = int(u[i])
                if counts.get(s, 0) + int(c[i]) > LIMIT:
                    bad.append(s)
            if not bad:
                continue
            # Try removing a y that contributes to at least one bad sum.
            # y contributes to sum s if there exists z in chosen with y+z=s (or 2y=s).
            # We'll just try a few y's.
            for y in chosen:
                if time.time() > deadline:
                    break
                # Check if removing y helps any bad sum.
                # Recompute counts after removal.
                new_counts2 = {}
                for i, a in enumerate(chosen):
                    if a == y:
                        continue
                    for j in range(i, len(chosen)):
                        b = chosen[j]
                        if b == y:
                            continue
                        s = a + b
                        new_counts2[s] = new_counts2.get(s, 0) + 1
                # Now try adding x.
                ok2, ns2, nc2 = _valid_add(x, np.array([z for z in chosen if z != y], dtype=np.int64), new_counts2)
                if ok2:
                    # Perform swap.
                    chosen.remove(y)
                    present.discard(y)
                    chosen.append(x)
                    chosen.sort()
                    chosen_arr = np.array(chosen, dtype=np.int64)
                    counts = nc2
                    present.add(x)
                    if x in candidates:
                        candidates.remove(x)
                    progress = True
                    break
    return best_set, best_size


def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 120s
    rng = np.random.default_rng(20240601)

    # Initial seed via mod-3 Sidon.
    seed = _mod3_seed(deadline)
    if time.time() > deadline:
        return seed

    best_set = list(seed)
    best_size = len(seed)

    # Run local search from this seed.
    ls_set, ls_size = _local_search(seed, deadline, rng)
    if ls_size > best_size:
        best_size = ls_size
        best_set = ls_set

    # A few randomized restarts with different seeds if time permits.
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        # Perturb: take current best, remove a random few, re-seed.
        if len(best_set) > 10:
            perturbed = list(best_set)
            for _ in range(min(5, len(perturbed) // 10)):
                if time.time() > deadline:
                    break
                idx = int(rng.integers(0, len(perturbed)))
                perturbed.pop(idx)
        else:
            perturbed = _mod3_seed(deadline)
        if time.time() > deadline - 2.0:
            break
        ls_set2, ls_size2 = _local_search(perturbed, deadline, rng)
        if ls_size2 > best_size:
            best_size = ls_size2
            best_set = ls_set2
        # If no improvement after a few restarts, stop.
        if restarts >= 6:
            break

    # Final validation.
    final = sorted(best_set)
    counts = {}
    valid = True
    for i, a in enumerate(final):
        for b in final[i:]:
            s = a + b
            counts[s] = counts.get(s, 0) + 1
    for s, c in counts.items():
        if c > LIMIT:
            valid = False
            break
    if not valid:
        # Fallback to a known-valid seed.
        final = _mod3_seed(deadline)

    return final
