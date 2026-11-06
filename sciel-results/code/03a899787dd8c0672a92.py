"""Blocking-priority greedy + aggressive local search for small maximal Sidon sets in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    s = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            t = a + b
            if t in s:
                return False
            s.add(t)
    return True

def _compute_sums(selected):
    sums = set()
    for i, a in enumerate(selected):
        for b in selected[i:]:
            sums.add(a + b)
    return sums

def _is_addable(x, selected, sums, sel_set):
    if x in sel_set:
        return False
    for a in selected:
        if x + a in sums:
            return False
    if 2 * x in sums:
        return False
    return True

def _addable_elements(selected, sums, sel_set):
    """Return the set of all x in [1,N] that can be added while preserving Sidon."""
    out = set()
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums, sel_set):
            out.add(x)
    return out

def _is_maximal_fast(selected, sums, sel_set):
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        if _is_addable(x, selected, sums, sel_set):
            return False
    return True

def _greedy_vertex_cover(seed_elements, rng, time_limit, use_degree=True):
    selected = list(seed_elements)
    sums = _compute_sums(selected)
    sel_set = set(selected)

    addable = _addable_elements(selected, sums, sel_set)

    while addable:
        if time.time() > time_limit:
            return None

        candidates = list(addable)

        if use_degree:
            best_x = None
            best_score = -1
            best_newly_blocked = set()
            rng.shuffle(candidates)

            for x in candidates:
                new_sums = set()
                for a in selected:
                    new_sums.add(x + a)
                new_sums.add(2 * x)

                newly_blocked = set()
                for y in addable:
                    if y == x:
                        continue
                    blocked = False
                    for a in selected:
                        if y + a in sums:
                            blocked = True
                            break
                    if not blocked:
                        if 2 * y in sums:
                            blocked = True
                    if not blocked:
                        if y + x in sums or y + x in new_sums:
                            blocked = True
                        elif 2 * y in new_sums:
                            blocked = True
                        else:
                            for a in selected:
                                if y + a in new_sums:
                                    blocked = True
                                    break
                    if blocked:
                        newly_blocked.add(y)

                score = len(newly_blocked)
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_newly_blocked = newly_blocked

            if best_x is None:
                break
        else:
            best_x = rng.choice(candidates)
            new_sums = set()
            for a in selected:
                new_sums.add(best_x + a)
            new_sums.add(2 * best_x)
            best_newly_blocked = set()
            for y in addable:
                if y == best_x:
                    continue
                blocked = False
                for a in selected:
                    if y + a in sums:
                        blocked = True
                        break
                if not blocked:
                    if 2 * y in sums:
                        blocked = True
                if not blocked:
                    if y + best_x in sums or y + best_x in new_sums:
                        blocked = True
                    elif 2 * y in new_sums:
                        blocked = True
                    else:
                        for a in selected:
                            if y + a in new_sums:
                                blocked = True
                                break
                if blocked:
                    best_newly_blocked.add(y)

        for a in selected:
            sums.add(best_x + a)
        sums.add(2 * best_x)
        selected.append(best_x)
        sel_set.add(best_x)
        addable.discard(best_x)
        addable -= best_newly_blocked

    return selected, sums, sel_set

def _local_search_aggressive(selected, sums, sel_set, rng, time_limit):
    """Removal, 1-swap, and 2-swap local search to shrink a maximal Sidon set."""
    selected = list(selected)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    best_size = len(selected)

    max_rounds = 60
    for rnd in range(max_rounds):
        if time.time() > time_limit:
            return None

        improved = False

        # Phase 1: single removals
        for i in range(len(selected)):
            if time.time() > time_limit:
                return None
            x = selected[i]
            new_selected = selected[:i] + selected[i+1:]
            new_sel_set = sel_set - {x}
            new_sums = _compute_sums(new_selected)
            if _is_maximal_fast(new_selected, new_sums, new_sel_set):
                selected = new_selected
                sums = new_sums
                sel_set = new_sel_set
                best_size = len(selected)
                improved = True
                break

        if improved:
            continue

        # Phase 2: 1-swaps (remove one, add one)
        if len(selected) >= 2:
            for i in range(len(selected)):
                if time.time() > time_limit:
                    return None
                removed = selected[i]
                base_selected = selected[:i] + selected[i+1:]
                base_sel_set = sel_set - {removed}
                base_sums = _compute_sums(base_selected)

                candidates = _addable_elements(base_selected, base_sums, base_sel_set)
                if not candidates:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                found = False
                for x in cand_list[:50]:
                    test_selected = base_selected + [x]
                    test_sel_set = base_sel_set | {x}
                    test_sums = set(base_sums)
                    for a in base_selected:
                        test_sums.add(x + a)
                    test_sums.add(2 * x)

                    if len(test_selected) < best_size and _is_maximal_fast(test_selected, test_sums, test_sel_set):
                        selected = test_selected
                        sums = test_sums
                        sel_set = test_sel_set
                        best_size = len(selected)
                        improved = True
                        found = True
                        break
                if found:
                    break

        if improved:
            continue

        # Phase 3: 2-swaps (remove two, add two)
        if len(selected) >= 3:
            n = len(selected)
            pair_attempts = min(25, n * (n - 1) // 2)
            pairs_tried = set()
            for _ in range(pair_attempts):
                if time.time() > time_limit:
                    return None
                i, j = rng.sample(range(n), 2)
                if i > j:
                    i, j = j, i
                key = (i, j)
                if key in pairs_tried:
                    continue
                pairs_tried.add(key)

                removed1 = selected[i]
                removed2 = selected[j]
                base_selected = [x for idx, x in enumerate(selected) if idx != i and idx != j]
                base_sel_set = sel_set - {removed1, removed2}
                base_sums = _compute_sums(base_selected)

                candidates = _addable_elements(base_selected, base_sums, base_sel_set)
                if len(candidates) < 2:
                    continue

                cand_list = list(candidates)
                rng.shuffle(cand_list)
                limit = min(20, len(cand_list))
                found = False
                for a in range(limit):
                    for b in range(a + 1, limit):
                        x, y = cand_list[a], cand_list[b]
                        test_selected = base_selected + [x, y]
                        test_sel_set = base_sel_set | {x, y}
                        test_sums = set(base_sums)
                        for a2 in base_selected:
                            test_sums.add(x + a2)
                            test_sums.add(y + a2)
                        test_sums.add(2 * x)
                        test_sums.add(2 * y)
                        test_sums.add(x + y)

                        if len(test_selected) < best_size and _is_maximal_fast(test_selected, test_sums, test_sel_set):
                            selected = test_selected
                            sums = test_sums
                            sel_set = test_sel_set
                            best_size = len(selected)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break

        if not improved:
            break

    return selected, sums, sel_set

def _verify_maximal(selected):
    if not _is_sidon(selected):
        return False
    sums = _compute_sums(selected)
    sel_set = set(selected)
    return _is_maximal_fast(selected, sums, sel_set)

def _force_maximal(selected):
    selected = list(selected)
    sums = _compute_sums(selected)
    sel_set = set(selected)
    changed = True
    while changed:
        changed = False
        addable = _addable_elements(selected, sums, sel_set)
        for x in addable:
            for a in selected:
                sums.add(x + a)
            sums.add(2 * x)
            selected.append(x)
            sel_set.add(x)
            changed = True
    return selected, sums, sel_set

def construct():
    start_time = time.time()
    deadline = start_time + 25.0

    # Initial upper bound: simple ascending greedy
    simple_selected, simple_sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in simple_selected] + [2 * x]
        if len(set(added)) == len(added) and not simple_sums.intersection(added):
            simple_selected.append(x)
            simple_sums.update(added)

    best_set = simple_selected
    best_sums = _compute_sums(best_set)
    best_sel_set = set(best_set)
    best_size = len(best_set)

    if not _is_maximal_fast(best_set, best_sums, best_sel_set):
        best_set, best_sums, best_sel_set = _force_maximal(best_set)
        best_size = len(best_set)

    rng = random.Random(42)

    attempts = 0
    while time.time() < deadline - 2.0:
        attempts += 1

        mod = attempts % 10
        if mod == 0:
            seed_elements = []
            use_degree = True
        elif mod == 1:
            seed_elements = rng.sample(range(1, N + 1), rng.randint(1, 3))
            use_degree = True
        elif mod == 2:
            seed_elements = rng.sample(range(1, N + 1), rng.randint(3, 6))
            use_degree = True
        elif mod == 3:
            seed_elements = rng.sample(range(1, N + 1), rng.randint(0, 2))
            use_degree = False
        elif mod == 4:
            seed_elements = rng.sample(range(1, N // 2 + 1), rng.randint(0, 3))
            use_degree = True
        elif mod == 5:
            seed_elements = rng.sample(range(N // 2 + 1, N + 1), rng.randint(0, 3))
            use_degree = True
        else:
            seed_elements = rng.sample(range(1, N + 1), rng.randint(0, 4))
            use_degree = True

        seed_ok = True
        seed_sums = set()
        for i, a in enumerate(seed_elements):
            for b in seed_elements[i:]:
                s = a + b
                if s in seed_sums:
                    seed_ok = False
                    break
                seed_sums.add(s)
            if not seed_ok:
                break
        if not seed_ok:
            continue

        result = _greedy_vertex_cover(seed_elements, rng, deadline - 1.0, use_degree)
        if result is None:
            continue

        final_set, final_sums, final_sel_set = result

        if not _is_maximal_fast(final_set, final_sums, final_sel_set):
            final_set, final_sums, final_sel_set = _force_maximal(final_set)

        if len(final_set) <= best_size:
            if time.time() < deadline - 1.5:
                ls_result = _local_search_aggressive(final_set, final_sums, final_sel_set, rng, deadline - 0.5)
                if ls_result is not None:
                    sel, sm, ss = ls_result
                    if _is_maximal_fast(sel, sm, ss) and len(sel) < best_size:
                        best_size = len(sel)
                        best_set = sel
                        best_sums = sm
                        best_sel_set = ss
                else:
                    if len(final_set) < best_size:
                        best_size = len(final_set)
                        best_set = final_set
                        best_sums = final_sums
                        best_sel_set = final_sel_set
            else:
                if len(final_set) < best_size:
                    best_size = len(final_set)
                    best_set = final_set
                    best_sums = final_sums
                    best_sel_set = final_sel_set

    if time.time() < deadline - 0.5 and len(best_set) > 5:
        ls_result = _local_search_aggressive(best_set, best_sums, best_sel_set, rng, deadline - 0.1)
        if ls_result is not None:
            sel, sm, ss = ls_result
            if _is_maximal_fast(sel, sm, ss) and len(sel) < best_size:
                best_set = sel
                best_size = len(sel)

    return best_set
# EVOLVE-BLOCK-END
