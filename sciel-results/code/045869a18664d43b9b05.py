import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 135  # safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    all_vals = list(range(1, 285))
    remaining = [x for x in range(12, 285)]

    # Best found so far
    best_set = list(prescribed)
    best_size = len(best_set)

    # ---------- Incremental sum maintenance (key speedup) ----------
    def build_sums(s):
        sums = set()
        n = len(s)
        for i in range(n):
            xi = s[i]
            for j in range(i, n):
                sums.add(xi + s[j])
        return sums

    def can_add(x, s, sums):
        # x is not in s; check all new sums are fresh
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_sums(x, s, sums):
        new = set()
        for y in s:
            new.add(x + y)
        new.add(2 * x)
        sums |= new

    def remove_sums(x, s, sums):
        # Remove all sums involving x (x is in s). s is the set WITH x.
        for y in s:
            sums.discard(x + y)

    # ---------- Greedy construction ----------
    def greedy(order, time_limit=2.0):
        s = list(prescribed)
        s_set = set(s)
        sums = build_sums(s)
        start = time.time()
        for x in order:
            if time.time() - start > time_limit:
                break
            if x in s_set:
                continue
            if can_add(x, s, sums):
                s.append(x)
                s_set.add(x)
                add_sums(x, s, sums)
        return s, sums

    rng = random.Random(12345)

    # Quick greedy passes to seed a good starting point
    s, sums = greedy(remaining, time_limit=1.0)
    if len(s) > best_size:
        best_set, best_size = list(s), len(s)
    s, sums = greedy(list(reversed(remaining)), time_limit=1.0)
    if len(s) > best_size:
        best_set, best_size = list(s), len(s)
    for _ in range(40):
        if time.time() > deadline - 10:
            break
        order = remaining[:]
        rng.shuffle(order)
        s, sums = greedy(order, time_limit=0.3)
        if len(s) > best_size:
            best_set, best_size = list(s), len(s)

    # ---------- Local search with incremental sums ----------
    def local_search(start_s, time_limit=40.0):
        nonlocal best_set, best_size
        s = sorted(list(start_s))
        s_set = set(s)
        sums = build_sums(s)
        start = time.time()

        # Phase 1: exhaustive add until stuck
        improved = True
        while improved and time.time() - start < time_limit and time.time() < deadline - 5:
            improved = False
            for x in all_vals:
                if time.time() > deadline - 5:
                    return
                if x in s_set:
                    continue
                if can_add(x, s, sums):
                    s.append(x)
                    s_set.add(x)
                    add_sums(x, s, sums)
                    if len(s) > best_size:
                        best_set, best_size = list(s), len(s)
                    improved = True
                    break

        # Phase 2: single swaps (remove one non-prescribed, add one new)
        if time.time() - start < time_limit and time.time() < deadline - 5:
            for _ in range(300):
                if time.time() > deadline - 5 or time.time() - start > time_limit:
                    break
                removable = [x for x in s if x not in prescribed_set]
                if not removable:
                    break
                found = False
                for x_remove in removable:
                    if time.time() > deadline - 5 or time.time() - start > time_limit:
                        break
                    # temporarily remove
                    s.remove(x_remove)
                    s_set.discard(x_remove)
                    remove_sums(x_remove, s_set | {x_remove}, sums)
                    reduced_sums = set(sums)
                    for x_add in all_vals:
                        if time.time() > deadline - 5:
                            s.append(x_remove)
                            s_set.add(x_remove)
                            add_sums(x_remove, s, sums)
                            return
                        if x_add in s_set:
                            continue
                        if can_add(x_add, s, reduced_sums):
                            s.append(x_add)
                            s_set.add(x_add)
                            add_sums(x_add, s, reduced_sums)
                            if len(s) > best_size:
                                best_set, best_size = list(s), len(s)
                            found = True
                            break
                    if not found:
                        # restore
                        s.append(x_remove)
                        s_set.add(x_remove)
                        add_sums(x_remove, s, reduced_sums)
                    else:
                        break
                if found:
                    continue

        # Phase 3: final exhaustive add pass
        if time.time() - start < time_limit and time.time() < deadline - 5:
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for x in all_vals:
                    if time.time() > deadline - 5:
                        return
                    if x in s_set:
                        continue
                    if can_add(x, s, sums):
                        s.append(x)
                        s_set.add(x)
                        add_sums(x, s, sums)
                        if len(s) > best_size:
                            best_set, best_size = list(s), len(s)
                        improved = True
                        break

    if time.time() < deadline - 40:
        local_search(best_set, time_limit=35.0)

    # ---------- Backtracking with forward checking ----------
    # Try to extend the current best set to a maximal Sidon set via
    # systematic search. Variables = candidate elements; forward check
    # prunes any candidate whose new sums collide.
    def backtrack_extend(start_s, time_limit=20.0):
        nonlocal best_set, best_size
        s = sorted(list(start_s))
        s_set = set(s)
        sums = build_sums(s)
        start = time.time()

        # Order candidates by "rarity" (fewer conflicts first) to prune early
        def candidate_score(x):
            if x in s_set:
                return (1, 0)
            cnt = 0
            for y in s:
                if x + y in sums:
                    cnt += 1
            if 2 * x in sums:
                cnt += 1
            return (0, cnt)

        def search(s, s_set, sums):
            nonlocal best_set, best_size
            if time.time() - start > time_limit or time.time() > deadline - 5:
                return
            if len(s) > best_size:
                best_set, best_size = list(s), len(s)

            # Find a valid candidate to add (forward check: only try valid ones)
            best_cand = None
            best_key = None
            for x in remaining:
                if x in s_set:
                    continue
                if not can_add(x, s, sums):
                    continue
                key = candidate_score(x)
                if best_key is None or key < best_key:
                    best_key = key
                    best_cand = x
                if best_key == (0, 0):
                    break
            if best_cand is None:
                return  # maximal (no valid add)

            x = best_cand
            s.append(x)
            s_set.add(x)
            add_sums(x, s, sums)
            search(s, s_set, sums)
            # undo
            s.pop()
            s_set.discard(x)
            remove_sums(x, s_set | {x}, sums)

        search(s, s_set, sums)

    if time.time() < deadline - 15 and len(best_set) > 4:
        backtrack_extend(best_set, time_limit=12.0)

    # ---------- Final local search polish ----------
    if time.time() < deadline - 5 and len(best_set) > 4:
        local_search(best_set, time_limit=5.0)

    # ---------- Verify ----------
    def is_valid_sidon(s):
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sm = sl[i] + sl[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True

    result = sorted(best_set)
    if not is_valid_sidon(result):
        return list(prescribed)
    return result

if __name__ == "__main__":
    result = construct()
    print(f"Size: {len(result)}")
    print(f"Result: {result}")
