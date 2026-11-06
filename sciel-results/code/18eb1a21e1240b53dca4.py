"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 25.0  # safety margin under 30s (killed at 60s)
    rng = random.Random(20240517)

    # ---- fast Sidon / addable machinery ----------------------------------
    def is_sidon(S):
        seen = set()
        L = sorted(S)
        for i in range(len(L)):
            a = L[i]
            if 2 * a in seen:
                return False
            seen.add(2 * a)
            for b in L[i + 1:]:
                s = a + b
                if s in seen:
                    return False
                seen.add(s)
        return True

    def addable_elements(S, T):
        """All x in [1,N]\S that keep S∪{x} Sidon."""
        res = []
        Sset = set(S)
        for x in range(1, N + 1):
            if x in Sset:
                continue
            if 2 * x in T:
                continue
            ok = True
            for s in S:
                if x + s in T:
                    ok = False
                    break
            if ok:
                res.append(x)
        return res

    def new_addable_after_add(S, T, x, old_addable):
        """Subset of old_addable still addable after adding x to S (T = sums of S)."""
        new_sums = {2 * x}
        for s in S:
            new_sums.add(x + s)
        T2 = T | new_sums
        Sset = set(S) | {x}
        rem = []
        for y in old_addable:
            if y == x:
                continue
            if 2 * y in T2:
                continue
            ok = True
            for s in Sset:
                if y + s in T2:
                    ok = False
                    break
            if ok:
                rem.append(y)
        return rem

    # ---- structural: maximalize a partial set by always adding the element
    #      that removes the most currently-addable elements -----------------
    def maximalize(S, T, addable, time_limit):
        """Greedy: repeatedly add the addable element that minimizes the
        remaining addable count. Stops when maximal or time runs out."""
        while addable and time.time() < time_limit:
            best_x = None
            best_rem = None
            best_len = len(addable) + 1
            for x in addable:
                rem = new_addable_after_add(S, T, x, addable)
                if len(rem) < best_len:
                    best_len = len(rem)
                    best_x = x
                    best_rem = rem
                    if best_len == 0:
                        break
            if best_x is None:
                break
            S.append(best_x)
            T = T | {2 * best_x} | {best_x + s for s in S[:-1]}
            addable = best_rem
        return S, T, addable

    # ---- refinement: limited top-k backtracking on a partial state --------
    def backtrack(S, T, addable, best, budget):
        """Try to find a smaller maximal set by exploring top-k choices.
        best = [size, set]; mutated in place."""
        def rec(S, T, addable, budget):
            if time.time() > budget:
                return
            if not addable:
                # maximal
                if len(S) < best[0]:
                    best[0] = len(S)
                    best[1] = list(S)
                return
            if len(S) >= best[0]:
                return
            # score each candidate by how many it blocks
            scored = []
            for x in addable:
                rem = new_addable_after_add(S, T, x, addable)
                blocked = len(addable) - 1 - len(rem)
                scored.append((blocked, x, rem))
            scored.sort(key=lambda t: (-t[0], t[1]))
            for blocked, x, rem in scored[:3]:
                S.append(x)
                T2 = T | {2 * x} | {x + s for s in S[:-1]}
                rec(S, T2, rem, budget)
                S.pop()
        rec(S, T, addable, budget)

    # ---- deterministic seed greedy fallback (guaranteed valid) -----------
    def seed_greedy():
        selected, sums = [], set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected

    # ---- main: randomized restarts of maximalization + refinement --------
    best = [N + 1, None]

    # restart 0: empty start, greedy maximalize
    S, T, addable = maximalize([], set(), addable_elements([], set()), deadline)
    if addable:  # not maximal yet (time) -> just ensure it's valid & record
        if is_sidon(S) and len(S) < best[0]:
            best[0] = len(S)
            best[1] = list(S)
    if not addable and is_sidon(S) and len(S) < best[0]:
        best[0] = len(S)
        best[1] = list(S)

    # randomized restarts: seed with a few random elements, then maximalize
    for it in range(40):
        if time.time() > deadline:
            break
        # pick a small random seed of mutually-compatible elements
        S = []
        T = set()
        cand = list(range(1, N + 1))
        rng.shuffle(cand)
        for x in cand:
            if len(S) >= 3:
                break
            added = [x + a for a in S] + [2 * x]
            if len(set(added)) == len(added) and not T.intersection(added):
                S.append(x)
                T.update(added)
        addable = addable_elements(S, T)
        S, T, addable = maximalize(S, T, addable, deadline)
        if is_sidon(S) and len(S) < best[0]:
            best[0] = len(S)
            best[1] = list(S)
            # refine the best found so far with a short backtracking budget
            if time.time() < deadline - 3.0:
                addable0 = addable_elements(S, T)
                backtrack(S, T, addable0, best, deadline - 1.0)
                if is_sidon(best[1]) and len(best[1]) < best[0]:
                    best[0] = len(best[1])

    if best[1] is None or not is_sidon(best[1]):
        best[1] = seed_greedy()
    return best[1]
# EVOLVE-BLOCK-END
