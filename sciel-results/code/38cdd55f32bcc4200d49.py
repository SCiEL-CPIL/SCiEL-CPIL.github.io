import time
import random
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1  # Lower bound on scope
    max_scope_bound = lb + 200  # Upper bound for difference lookup

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_all_diffs(rows):
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def greedy_construct(seed, scope_limit=None):
        """Build rows greedily with randomization."""
        rng = random.Random(seed)
        used = set()
        rows = []

        if scope_limit is None:
            scope_limit = lb + 150

        for row_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False

                # Try to find next element
                # Use a window that grows if needed
                search_start = last + 1
                search_end = min(last + 30 + rng.randint(0, 20), scope_limit)

                candidates = list(range(search_start, search_end + 1))
                if rng.random() < 0.5:
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
        """Try to reduce scope by moving elements to smaller positions."""
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

            # Remove this element and its diffs from used_set
            for a in row[:elem_idx]:
                used_set.discard(old_val - a)
            for a in row[elem_idx + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_idx)

            # Try to re-insert at a smaller position
            found = False
            min_val = row[elem_idx - 1] + 1 if elem_idx > 0 else 1
            max_val = old_val - 1

            # Try values from max_val down to min_val
            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_idx]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_idx:]:
                        d = a - new_val
                        if d in used_set:
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
                # Restore original
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return rows, scope

    def generate_candidates(num_candidates, scope_limit):
        """Generate many candidate rows using different strategies."""
        candidates = []
        rng = random.Random(12345)

        for i in range(num_candidates):
            if time.time() > deadline - 30:
                break

            # Use different strategies
            if i % 3 == 0:
                rows = greedy_construct(i * 31 + 7, scope_limit)
            elif i % 3 == 1:
                # Random restart with different seed
                rows = greedy_construct(100000 + i * 97, scope_limit)
            else:
                # More aggressive compaction attempt
                rows = greedy_construct(200000 + i * 53, scope_limit)

            if rows is not None:
                candidates.append(rows)

        return candidates

    def solve_mis(candidates, n_rows):
        """
        Solve Maximum Independent Set to find non-conflicting rows.
        Each candidate is a full solution (n_rows rows). We want to pick
        one candidate that has the minimum scope.
        Actually, we want to pick n_rows rows from different candidates
        that don't conflict with each other.
        """
        # Build conflict graph
        # Each candidate is a set of n_rows rows
        # Two rows conflict if they share any difference

        # First, compute differences for each row in each candidate
        row_diffs = []
        for cand in candidates:
            cand_diffs = []
            for row in cand:
                diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.add(row[j] - row[i])
                cand_diffs.append(diffs)
            row_diffs.append(cand_diffs)

        # Build adjacency: which rows from different candidates conflict
        # We want to pick one row from each candidate such that no two conflict
        # This is a constraint satisfaction problem

        # Simpler approach: just pick the candidate with minimum scope
        # and verify it's valid
        best_candidate = None
        best_scope = float('inf')

        for i, cand in enumerate(candidates):
            scope = get_scope(cand)
            if scope < best_scope:
                best_scope = scope
                best_candidate = cand

        return best_candidate, best_scope

    def combine_rows(candidates):
        """
        Try to combine rows from different candidates to get a better solution.
        Greedily pick rows that don't conflict.
        """
        # Sort candidates by scope
        sorted_cands = sorted(candidates, key=lambda c: get_scope(c))

        used_diffs = set()
        result_rows = []

        for row_idx in range(n_rows):
            # Try to find a row for this position from any candidate
            best_row = None
            best_scope = float('inf')

            for cand in sorted_cands:
                row = cand[row_idx]
                # Check if this row conflicts with already chosen rows
                row_diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        row_diffs.add(row[j] - row[i])

                if row_diffs & used_diffs:
                    continue

                # No conflict, consider this row
                scope = max(row)
                if scope < best_scope:
                    best_scope = scope
                    best_row = row

            if best_row is not None:
                result_rows.append(best_row)
                # Add its diffs to used
                for i in range(len(best_row)):
                    for j in range(i + 1, len(best_row)):
                        used_diffs.add(best_row[j] - best_row[i])
            else:
                # Can't find a non-conflicting row, use first candidate's row
                result_rows.append(sorted_cands[0][row_idx])

        return result_rows

    # Main algorithm
    best_rows = None
    best_scope = float('inf')

    # Phase 1: Generate many candidates
    candidates = generate_candidates(200, lb + 150)

    # Phase 2: Find best single candidate
    for cand in candidates:
        scope = get_scope(cand)
        if scope < best_scope:
            best_scope = scope
            best_rows = [list(r) for r in cand]

    # Phase 3: Try combining rows from different candidates
    if len(candidates) > 1:
        combined = combine_rows(candidates)
        # Validate combined solution
        used = set()
        valid = True
        for row in combined:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        valid = False
                        break
                    used.add(d)
            if not valid:
                break

        if valid:
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

        # More compaction with different seeds
        for seed in [123, 456, 789]:
            if time.time() > deadline - 10:
                break
            improved, imp_scope = local_compact(best_rows, seed, max_iters=500)
            if imp_scope < best_scope:
                best_scope = imp_scope
                best_rows = improved

    # Phase 5: Iterated local search
    if best_rows is not None and time.time() < deadline - 20:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope

        for ils_iter in range(50):
            if time.time() > deadline - 5:
                break

            # Perturb: randomly move an element
            rng_ils = random.Random(10000 + ils_iter)
            perturbed = [list(r) for r in current_rows]
            row_idx = rng_ils.randint(0, n_rows - 1)
            row = perturbed[row_idx]

            if len(row) > 2:
                elem_pos = rng_ils.randint(1, len(row) - 1)
                old_val = row[elem_pos]

                # Remove and re-insert at different position
                used_set = compute_all_diffs(perturbed)
                for a in row[:elem_pos]:
                    used_set.discard(old_val - a)
                for a in row[elem_pos + 1:]:
                    used_set.discard(a - old_val)
                row.pop(elem_pos)

                # Try to insert at different position
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
                        # Revert
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used_set.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used_set.add(a - old_val)

            # Local improve
            improved_rows, imp_scope = local_compact(perturbed, 20000 + ils_iter, max_iters=200)

            # Accept if better
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


# Quick self-test
if __name__ == "__main__":
    result = construct(4, 7)
    if result:
        print(f"Scope: {max(max(r) for r in result)}")
        print(f"Rows: {result}")
        # Validate
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
