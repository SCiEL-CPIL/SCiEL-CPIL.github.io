import time
import random

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_all_diffs(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def validate(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(seed, scope_limit, compactness=1.0):
        """
        Build rows greedily. compactness in (0, 1]:
          1.0 -> always pick smallest valid
          0.0 -> pick uniformly at random among valid
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False

                # Search window
                window = int(20 + 40 * (1 - compactness))
                search_end = min(last + window + rng.randint(0, 15), scope_limit)

                candidates = list(range(last + 1, search_end + 1))

                # Bias: try smaller values first with probability compactness
                if rng.random() < compactness:
                    # Mostly in order, with small perturbation
                    for i in range(len(candidates) - 1, 0, -1):
                        if rng.random() < 0.1 * (1 - compactness):
                            candidates[i], candidates[i - 1] = candidates[i - 1], candidates[i]
                else:
                    rng.shuffle(candidates)

                for x in candidates:
                    new_diffs = [x - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(x)
                    used.update(new_diffs)
                    found = True
                    break

                if not found:
                    # Wider search
                    for x in range(last + 1, scope_limit + 1):
                        new_diffs = [x - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(x)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    def local_compact(rows, seed, max_iters=500):
        """
        Local search: pick random element, try to move it to a smaller valid position.
        Also try to move it to a different position in the same row.
        """
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)
        used_set = compute_all_diffs(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = rows[row_idx]
            if len(row) < 2:
                continue

            elem_idx = rng.randint(1, len(row) - 1)
            old_val = row[elem_idx]
            if old_val <= 1:
                continue

            # Remove element and its diffs
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            # Try to re-insert at a smaller value at the same position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    if new_val - a in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        if a - new_val in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_idx, new_val)
                    for a in row[:elem_idx]:
                        used_set.add(new_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(rows)
                if new_scope < scope:
                    scope = new_scope
            else:
                # Try to re-insert at a different position with smaller value
                found2 = False
                for pos in range(1, len(row) + 1):
                    if pos == elem_idx:
                        continue
                    min_v = row[pos - 1] + 1 if pos > 0 else 1
                    max_v = row[pos] - 1 if pos < len(row) else old_val - 1
                    if min_v > max_v:
                        continue
                    for new_val in range(max_v, min_v - 1, -1):
                        valid = True
                        for a in row[:pos]:
                            if new_val - a in used_set:
                                valid = False
                                break
                        if valid:
                            for a in row[pos:]:
                                if a - new_val in used_set:
                                    valid = False
                                    break
                        if valid:
                            row.insert(pos, new_val)
                            for a in row[:pos]:
                                used_set.add(new_val - a)
                            for a in row[pos + 1:]:
                                used_set.add(a - new_val)
                            found2 = True
                            break
                    if found2:
                        break

                if found2:
                    new_scope = get_scope(rows)
                    if new_scope < scope:
                        scope = new_scope
                else:
                    # Restore original
                    row.insert(elem_idx, old_val)
                    for a in row[:elem_idx]:
                        used_set.add(old_val - a)
                    for a in row[elem_idx + 1:]:
                        used_set.add(a - old_val)

        return rows, scope

    def combine_rows(candidates):
        """
        Greedily combine rows from different candidates to minimize scope.
        """
        # Sort candidates by scope
        sorted_cands = sorted(candidates, key=lambda c: get_scope(c))

        used_diffs = set()
        result_rows = []

        for row_idx in range(n_rows):
            best_row = None
            best_scope = float('inf')

            for cand in sorted_cands:
                row = cand[row_idx]
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])

                if row_diffs & used_diffs:
                    continue

                scope = max(row)
                if scope < best_scope:
                    best_scope = scope
                    best_row = row

            if best_row is not None:
                result_rows.append(best_row)
                for i in range(len(best_row)):
                    for j in range(i + 1, len(best_row)):
                        used_diffs.add(best_row[j] - best_row[i])
            else:
                result_rows.append(sorted_cands[0][row_idx])

        return result_rows

    # Main algorithm
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Generate diverse candidates
    candidates = []
    num_candidates = 300
    for i in range(num_candidates):
        if time.time() > deadline - 30:
            break

        # Vary compactness and scope limit
        compactness = 0.3 + 0.7 * (i % 10) / 10
        scope_limit = lb + 80 + (i % 20) * 10

        seed = i * 7919 + 13
        rows = greedy_construct(seed, scope_limit, compactness)

        if rows is not None:
            candidates.append(rows)

    # Phase 2: Find best single candidate
    for cand in candidates:
        scope = get_scope(cand)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in cand]

    # Phase 3: Combine rows from different candidates
    if len(candidates) > 1:
        combined = combine_rows(candidates)
        if validate(combined):
            scope = get_scope(combined)
            if scope < best_scope:
                best_scope = scope
                best_rows = combined

    # Phase 4: Local compaction on best solution
    if best_rows is not None:
        improved, imp_scope = local_compact(best_rows, 42, max_iters=1000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

        for seed in [123, 456, 789, 1011, 1013]:
            if time.time() > deadline - 15:
                break
            improved, imp_scope = local_compact(best_rows, seed, max_iters=500)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 5: Iterated local search with perturbations
    if best_rows is not None and time.time() < deadline - 25:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope

        for ils_iter in range(80):
            if time.time() > deadline - 5:
                break

            rng_ils = random.Random(10000 + ils_iter)
            perturbed = [list(r) for r in current_rows]
            row_idx = rng_ils.randint(0, n_rows - 1)
            row = perturbed[row_idx]

            if len(row) > 2:
                elem_pos = rng_ils.randint(1, len(row) - 1)
                old_val = row[elem_pos]

                used_set = compute_all_diffs(perturbed)
                for a in row[:elem_pos]:
                    used_set.discard(old_val - a)
                for a in row[elem_pos + 1:]:
                    used_set.discard(a - old_val)
                row.pop(elem_pos)

                new_pos = rng_ils.randint(1, len(row))
                if new_pos != elem_pos:
                    min_v = row[new_pos - 1] + 1 if new_pos > 0 else 1
                    max_v = row[new_pos] - 1 if new_pos < len(row) else current_scope + 10

                    valid_val = None
                    for v in range(min_v, max_v + 1):
                        ok = True
                        for a in row[:new_pos]:
                            if v - a in used_set:
                                ok = False
                                break
                        if ok:
                            for a in row[new_pos:]:
                                if a - v in used_set:
                                    ok = False
                                    break
                        if ok:
                            valid_val = v
                            break

                    if valid_val is not None:
                        row.insert(new_pos, valid_val)
                        for a in row[:new_pos]:
                            used_set.add(valid_val - a)
                        for a in row[new_pos + 1:]:
                            used_set.add(a - valid_val)
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used_set.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used_set.add(a - old_val)

            improved_rows, imp_scope = local_compact(perturbed, 20000 + ils_iter, max_iters=300)

            if imp_scope < current_scope:
                current_rows = improved_rows
                current_scope = imp_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]

    # Final compaction
    if best_rows is not None and time.time() < deadline - 2:
        improved, imp_scope = local_compact(best_rows, 99999, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    return best_rows


if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        used = set()
        valid = True
        for row in result:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        print(f"Duplicate diff {d}")
                    used.add(d)
        print(f"Valid: {valid}")
    else:
        print("No valid construction found")
