"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

    def time_left():
        return deadline - time.time()

    def rebuild_sums(selected):
        """Rebuild the triple-sum set from a sorted list."""
        sums = set()
        n = len(selected)
        for i in range(n):
            a = selected[i]
            for j in range(i, n):
                b = selected[j]
                for k in range(j, n):
                    sums.add(a + b + selected[k])
        return sums

    def check_candidate(selected, x, sums_set):
        """Check if adding x to selected keeps all triple sums unique.
        selected must be sorted. Returns (True, new_sums) if valid."""
        # New triple sums involving x:
        # 1. (x, x, x) = 3x
        # 2. (x, x, y) for y in selected -> 2x + y
        # 3. (x, y, z) for y <= z in selected -> x + y + z

        if 3 * x in sums_set:
            return False, None

        new_sums = {3 * x}

        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)

        n = len(selected)
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)

        return True, new_sums

    def greedy_forward(start, end, selected, sums_set):
        for x in range(start, end + 1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    def greedy_random(selected, sums_set, candidates):
        for x in candidates:
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    def is_valid_set(values):
        """Verify all unordered triple sums are unique."""
        n = len(values)
        seen = set()
        for i in range(n):
            a = values[i]
            for j in range(i, n):
                b = values[j]
                for k in range(j, n):
                    s = a + b + values[k]
                    if s in seen:
                        return False
                    seen.add(s)
        return True

    # ---- Strategy A: base-expansion construction (guaranteed B_3[1]) ----
    # B=7, digits {0,1,2}, length 4 -> 3^4 = 81 values, drop 0 -> 80 positives.
    # Max value = 2*(1 + 7 + 49 + 343) = 800 <= 4096.
    # No carries in triple sums (max digit sum 6 < 7), so sums are unique.
    B = 7
    digits = [0, 1, 2]
    base_values = []
    for d1 in digits:
        for d2 in digits:
            for d3 in digits:
                for d4 in digits:
                    val = d1 + d2 * B + d3 * (B * B) + d4 * (B * B * B)
                    if val > 0:
                        base_values.append(val)
    base_values.sort()

    best = []
    if base_values and is_valid_set(base_values):
        best = base_values[:]

    # ---- Strategy B: greedy forward from 1 ----
    if time_left() > 5.0:
        sel = []
        sm = set()
        sel, sm = greedy_forward(1, 4096, sel, sm)
        if len(sel) > len(best):
            best = sel[:]

    # ---- Strategy C: greedy backward from 4096 ----
    if time_left() > 5.0:
        sel = []
        sm = set()
        for x in range(4096, 0, -1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(sel, x, sm)
            if valid:
                sel.append(x)
                sm.update(new_sums)
        if len(sel) > len(best):
            best = sel[:]

    # ---- Strategy D: randomized greedy restarts ----
    rng = random.Random(42)
    if time_left() > 5.0:
        for trial in range(30):
            if time_left() < 5.0:
                break
            candidates = list(range(1, 4097))
            rng.shuffle(candidates)
            sel = []
            sm = set()
            sel, sm = greedy_random(sel, sm, candidates)
            if len(sel) > len(best):
                best = sel[:]

    # ---- Strategy E: local search on best (add + swap) ----
    if time_left() > 10.0:
        current = best[:]
        current.sort()
        current_set = set(current)
        current_sums = rebuild_sums(current)

        # Pure additions: try to push size beyond current
        improved = True
        while improved and time_left() > 10.0:
            improved = False
            for _ in range(300):
                if time_left() < 10.0:
                    break
                x = rng.randint(1, 4096)
                if x in current_set:
                    continue
                valid, new_sums = check_candidate(current, x, current_sums)
                if valid:
                    current.append(x)
                    current.sort()
                    current_set.add(x)
                    current_sums.update(new_sums)
                    improved = True
                    if len(current) > len(best):
                        best = current[:]

        # Swap-based local search: replace one element to open room for additions
        if time_left() > 15.0:
            for _ in range(400):
                if time_left() < 15.0:
                    break
                idx = rng.randrange(len(current))
                old = current[idx]
                new = rng.randint(1, 4096)
                if new in current_set or new == old:
                    continue
                # remove old, rebuild sums
                test_list = current[:idx] + current[idx + 1:]
                test_sums = rebuild_sums(test_list)
                # try to add new
                valid, new_sums = check_candidate(sorted(test_list), new, test_sums)
                if valid:
                    # accept swap (size unchanged) but it may enable later additions
                    current = sorted(test_list + [new])
                    current_set = set(current)
                    current_sums = test_sums | new_sums
                    # attempt a few additions after the swap
                    for __ in range(50):
                        if time_left() < 15.0:
                            break
                        xx = rng.randint(1, 4096)
                        if xx in current_set:
                            continue
                        v2, ns2 = check_candidate(current, xx, current_sums)
                        if v2:
                            current.append(xx)
                            current.sort()
                            current_set.add(xx)
                            current_sums.update(ns2)
                            if len(current) > len(best):
                                best = current[:]

    return best
# EVOLVE-BLOCK-END
