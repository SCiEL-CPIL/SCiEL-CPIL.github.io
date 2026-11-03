<<<<<<< SEARCH
def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: repeatedly pull each element to its smallest
    feasible value, then try 2-element swaps to escape local minima.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        """Pull all elements in row row_idx to their smallest feasible values."""
        row = best[row_idx]
        any_changed = False
        # Process from left to right (after pos 0)
        for pos in range(1, k_plus_1):
            # Find smallest feasible value > row[pos-1]
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                # Check conflicts
                new_diffs = set()
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        break
                    new_diffs.add(d)
                else:
                    # Check diffs with elements after pos
                    ok = True
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - row[pos])
                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_changed = True
                    break
        return any_changed

    # Phase 1: Repeatedly pull all rows to smallest feasible
    for iteration in range(30):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Try 2-element swaps to escape local minima
    for iteration in range(15):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos1 in range(1, k_plus_1):
                for pos2 in range(pos1 + 1, k_plus_1):
                    # Try swapping row[pos1] and row[pos2]
                    v1, v2 = row[pos1], row[pos2]
                    if v1 == v2:
                        continue
                    # After swap, check increasing order
                    if pos1 > 0 and v2 <= row[pos1 - 1]:
                        continue
                    if pos2 < k_plus_1 - 1 and v1 >= row[pos2 + 1]:
                        continue
                    if v1 >= v2:
                        continue
                    # Check feasibility
                    new_diffs = set()
                    for i in range(pos1):
                        d = v2 - row[i]
                        if d <= 0:
                            break
                        new_diffs.add(d)
                    else:
                        ok = True
                        for i in range(pos1 + 1, pos2):
                            d = v2 - row[i]
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                        if ok:
                            for i in range(pos2 + 1, k_plus_1):
                                d = row[i] - v2
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            for i in range(pos1 + 1, pos2):
                                d = v1 - row[i]
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            for i in range(pos2 + 1, k_plus_1):
                                d = row[i] - v1
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            # Remove old diffs
                            old_diffs = set()
                            for i in range(pos1):
                                old_diffs.add(v1 - row[i])
                            for i in range(pos1 + 1, pos2):
                                old_diffs.add(v1 - row[i])
                            for i in range(pos2 + 1, k_plus_1):
                                old_diffs.add(row[i] - v1)
                            for i in range(pos1):
                                old_diffs.add(v2 - row[i])
                            for i in range(pos1 + 1, pos2):
                                old_diffs.add(v2 - row[i])
                            for i in range(pos2 + 1, k_plus_1):
                                old_diffs.add(row[i] - v2)
                            old_diffs.add(v2 - v1)
                            # Check no conflict
                            if not (new_diffs & (used_diffs - old_diffs)):
                                used_diffs.difference_update(old_diffs)
                                used_diffs.update(new_diffs)
                                row[pos1], row[pos2] = v2, v1
                                any_changed = True
                                break
                if any_changed:
                    break
            if any_changed:
                break
        if not any_changed:
            break

    # Phase 3: Final pull pass
    for iteration in range(10):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
=======
def _improve_scope(solution, n_rows, k, deadline):
    """
    Aggressive scope reduction: repeatedly pull each element to its smallest
    feasible value, then try 2-element swaps to escape local minima.
    """
    if not solution:
        return solution

    best = [list(row) for row in solution]
    best_scope = max(max(row) for row in best)
    k_plus_1 = k + 1

    # Collect all used differences
    used_diffs = set()
    for row in best:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(row[j] - row[i])

    def _pull_to_smallest(row_idx):
        """Pull all elements in row row_idx to their smallest feasible values."""
        row = best[row_idx]
        any_changed = False
        # Process from left to right (after pos 0)
        for pos in range(1, k_plus_1):
            # Find smallest feasible value > row[pos-1]
            for new_val in range(row[pos - 1] + 1, row[pos] + 1):
                # Check conflicts
                new_diffs = set()
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        break
                    new_diffs.add(d)
                else:
                    # Check diffs with elements after pos
                    ok = True
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                    if not ok:
                        continue
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(row[pos] - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - row[pos])
                    # Apply
                    used_diffs.difference_update(old_diffs)
                    used_diffs.update(new_diffs)
                    row[pos] = new_val
                    any_changed = True
                    break
        return any_changed

    def _try_single_element_shift(row_idx):
        """Try shifting a single element to a nearby value."""
        row = best[row_idx]
        for pos in range(1, k_plus_1):
            original = row[pos]
            # Try values slightly larger and smaller
            for delta in [1, -1, 2, -2, 3, -3]:
                new_val = original + delta
                if new_val <= row[pos - 1] or (pos < k_plus_1 - 1 and new_val >= row[pos + 1]):
                    continue
                # Check feasibility
                new_diffs = set()
                ok = True
                for i in range(pos):
                    d = new_val - row[i]
                    if d <= 0 or d in used_diffs:
                        ok = False
                        break
                    new_diffs.add(d)
                if ok:
                    for j in range(pos + 1, k_plus_1):
                        d = row[j] - new_val
                        if d <= 0 or d in used_diffs:
                            ok = False
                            break
                        new_diffs.add(d)
                if ok:
                    # Remove old diffs
                    old_diffs = set()
                    for i in range(pos):
                        old_diffs.add(original - row[i])
                    for j in range(pos + 1, k_plus_1):
                        old_diffs.add(row[j] - original)
                    # Check no conflict
                    if not (new_diffs & (used_diffs - old_diffs)):
                        used_diffs.difference_update(old_diffs)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        return True
        return False

    # Phase 1: Repeatedly pull all rows to smallest feasible
    for iteration in range(50):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 2: Try single element shifts to escape local minima
    for iteration in range(30):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _try_single_element_shift(row_idx):
                any_changed = True
        if not any_changed:
            break

    # Phase 3: Try 2-element swaps to escape local minima
    for iteration in range(20):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            row = best[row_idx]
            for pos1 in range(1, k_plus_1):
                for pos2 in range(pos1 + 1, k_plus_1):
                    # Try swapping row[pos1] and row[pos2]
                    v1, v2 = row[pos1], row[pos2]
                    if v1 == v2:
                        continue
                    # After swap, check increasing order
                    if pos1 > 0 and v2 <= row[pos1 - 1]:
                        continue
                    if pos2 < k_plus_1 - 1 and v1 >= row[pos2 + 1]:
                        continue
                    if v1 >= v2:
                        continue
                    # Check feasibility
                    new_diffs = set()
                    for i in range(pos1):
                        d = v2 - row[i]
                        if d <= 0:
                            break
                        new_diffs.add(d)
                    else:
                        ok = True
                        for i in range(pos1 + 1, pos2):
                            d = v2 - row[i]
                            if d <= 0 or d in used_diffs:
                                ok = False
                                break
                            new_diffs.add(d)
                        if ok:
                            for i in range(pos2 + 1, k_plus_1):
                                d = row[i] - v2
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            for i in range(pos1 + 1, pos2):
                                d = v1 - row[i]
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            for i in range(pos2 + 1, k_plus_1):
                                d = row[i] - v1
                                if d <= 0 or d in used_diffs:
                                    ok = False
                                    break
                                new_diffs.add(d)
                        if ok:
                            # Remove old diffs
                            old_diffs = set()
                            for i in range(pos1):
                                old_diffs.add(v1 - row[i])
                            for i in range(pos1 + 1, pos2):
                                old_diffs.add(v1 - row[i])
                            for i in range(pos2 + 1, k_plus_1):
                                old_diffs.add(row[i] - v1)
                            for i in range(pos1):
                                old_diffs.add(v2 - row[i])
                            for i in range(pos1 + 1, pos2):
                                old_diffs.add(v2 - row[i])
                            for i in range(pos2 + 1, k_plus_1):
                                old_diffs.add(row[i] - v2)
                            old_diffs.add(v2 - v1)
                            # Check no conflict
                            if not (new_diffs & (used_diffs - old_diffs)):
                                used_diffs.difference_update(old_diffs)
                                used_diffs.update(new_diffs)
                                row[pos1], row[pos2] = v2, v1
                                any_changed = True
                                break
                if any_changed:
                    break
            if any_changed:
                break
        if not any_changed:
            break

    # Phase 4: Final pull pass
    for iteration in range(20):
        if time.time() > deadline - 3.0:
            break
        any_changed = False
        for row_idx in range(n_rows):
            if _pull_to_smallest(row_idx):
                any_changed = True
        if not any_changed:
            break

    new_scope = max(max(row) for row in best)
    if new_scope < best_scope:
        return best
    else:
        return [list(row) for row in solution]
>>>>>>> REPLACE
