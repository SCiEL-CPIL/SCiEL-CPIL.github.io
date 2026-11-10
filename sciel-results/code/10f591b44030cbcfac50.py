"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 256

# EVOLVE-BLOCK-START
import random
import time


def _is_sidon_and_maximal(selected, N):
    """Check if selected is a valid Sidon set and inclusion-maximal in [1,N]."""
    if len(selected) == 0:
        return False
    selected_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    # Check maximality: no missing element can be added
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        added_sums = set()
        ok = True
        for a in selected:
            s1 = x + a
            if s1 in sums or s1 in added_sums:
                ok = False
                break
            added_sums.add(s1)
        if not ok:
            continue
        s2 = 2 * x
        if s2 not in sums and s2 not in added_sums:
            return False
    return True


def _greedy_max_blocking(N, rng=None, tie_break='max', weight_random=0.0):
    """
    Max-blocking greedy: at each step, among all valid candidates,
    pick the one that blocks the most other valid candidates.

    tie_break: 'max' = always pick max-block, 'random' = random among max-block
    weight_random: if >0, add random noise to block count for diversity
    """
    selected = []
    used_sums = set()
    candidates = set(range(1, N + 1))

    while candidates:
        # Find all valid candidates
        valid = []
        for x in candidates:
            new_sums = {2 * x} | {x + s for s in selected}
            if not new_sums & used_sums:
                valid.append(x)

        if not valid:
            break

        # For each valid candidate, count how many other valid candidates it blocks
        best_x = None
        best_blocks = -1
        best_ties = []

        for x in valid:
            added_sums = {2 * x} | {x + s for s in selected}
            blocks = 0
            for y in valid:
                if y == x:
                    continue
                blocked = False
                if 2 * y in added_sums:
                    blocked = True
                else:
                    for s in selected:
                        if y + s in added_sums:
                            blocked = True
                            break
                    if not blocked:
                        if y + x in used_sums:
                            blocked = True
                if blocked:
                    blocks += 1

            score = blocks
            if weight_random > 0:
                score += rng.random() * weight_random

            if score > best_blocks:
                best_blocks = score
                best_x = x
                best_ties = [x]
            elif score == best_blocks:
                best_ties.append(x)

        # Select the best candidate
        if tie_break == 'random' and len(best_ties) > 1:
            chosen = rng.choice(best_ties)
        else:
            chosen = best_x

        selected.append(chosen)
        used_sums.update({2 * chosen} | {chosen + s for s in selected[:-1]})
        candidates.discard(chosen)

    return selected


def _greedy_with_order(order, N):
    """Greedy construction using a specific element ordering."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected


def _greedy_maximize(base, N):
    """
    Greedily add valid elements (in ascending order) to a base set until
    inclusion-maximal. Returns the maximal superset.
    """
    sel = list(base)
    sel_set = set(sel)
    sums = set()
    for i, a in enumerate(sel):
        for j in range(i, len(sel)):
            sums.add(a + sel[j])
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        added = set()
        ok = True
        for a in sel:
            s = x + a
            if s in sums or s in added:
                ok = False
                break
            added.add(s)
        if ok:
            s2 = 2 * x
            if s2 in sums or s2 in added:
                ok = False
            else:
                added.add(s2)
        if ok:
            sel.append(x)
            sel_set.add(x)
            sums.update(added)
    return sel


def _try_remove(selected, N):
    """
    Try removing one element and greedily re-maximizing. If the resulting
    maximal set is strictly smaller than the input, return it; else None.
    """
    n = len(selected)
    sel_set = set(selected)
    for i in range(n):
        base = [x for x in selected if x != selected[i]]
        ext = _greedy_maximize(base, N)
        if len(ext) < n:
            return ext
    return None


def _refine(selected, N, deadline, rng):
    """
    Iteratively shrink a valid maximal Sidon set using remove-and-re-maximize
    moves, with occasional random restarts from a single random element.
    Returns the smallest maximal set found.
    """
    best = list(selected)
    cur = list(selected)
    while time.time() < deadline:
        if len(cur) <= 1:
            break
        new = _try_remove(cur, N)
        if new is not None:
            if len(new) < len(best):
                best = new
            cur = new
        else:
            # No single removal shrinks the set: diversify via a random seed.
            seed = rng.randint(1, N)
            ext = _greedy_maximize([seed], N)
            if len(ext) < len(best):
                best = ext
            cur = ext
    return best


def construct():
    deadline = time.time() + 54.0  # safety margin before 60s kill
    best_size = N + 1
    best_set = None
    rng = random.Random(42)

    # Strategy 1: Max-blocking greedy (pure) - the key builder
    candidate = _greedy_max_blocking(N, rng, tie_break='max')
    if _is_sidon_and_maximal(candidate, N):
        best_size = len(candidate)
        best_set = candidate

    # Strategy 2: Max-blocking with random tie-breaking among max-block candidates
    for attempt in range(200):
        if time.time() > deadline:
            break
        candidate = _greedy_max_blocking(N, rng, tie_break='random')
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break

    # Strategy 3: Max-blocking with random weighting for diversity
    for attempt in range(200):
        if time.time() > deadline:
            break
        candidate = _greedy_max_blocking(N, rng, tie_break='max', weight_random=1.0)
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break

    # Strategy 4: Random ordering greedy (fallback/diversity)
    for attempt in range(500):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        candidate = _greedy_with_order(order, N)
        if len(candidate) < best_size and _is_sidon_and_maximal(candidate, N):
            best_size = len(candidate)
            best_set = candidate
            if best_size <= 16:
                break

    if best_set is None:
        # Fallback: build a valid maximal set and use it
        best_set = _greedy_maximize([1], N)
        best_size = len(best_set)

    # Strategy 5: Refinement - shrink the best set via remove-and-re-maximize.
    # This is the main jump mechanism: greedy leaves slack that single-element
    # removals can exploit to cut the cardinality further.
    if time.time() < deadline:
        refined = _refine(best_set, N, deadline, rng)
        if len(refined) < best_size and _is_sidon_and_maximal(refined, N):
            best_size = len(refined)
            best_set = refined

    return best_set
# EVOLVE-BLOCK-END
