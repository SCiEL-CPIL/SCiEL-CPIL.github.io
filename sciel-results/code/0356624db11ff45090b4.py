import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    k_plus_1 = k + 1
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs

    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate_rows(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(seed, strategy='basic'):
        """
        Construct rows using greedy with randomization.
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        if strategy == 'random_order':
            row_order = list(range(n_rows))
            rng.shuffle(row_order)
            row_slots = [None] * n_rows

            for row_idx in row_order:
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False
                    if strategy == 'compact':
                        search_end = last + 30 + rng.randint(0, 20)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 40 + rng.randint(0, 30)

                    candidates = list(range(last + 1, min(search_end, lb + 200)))
                    if rng.random() < 0.6:
                        for i in range(len(candidates) - 1, 0, -1):
                            if rng.random() < 0.1:
                                candidates[i], candidates[i-1] = candidates[i-1], candidates[i]
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
                        for x in range(last + 1, lb + 300):
                            new_diffs = [x - a for a in row]
                            if any(d in used for d in new_diffs):
                                continue
                            row.append(x)
                            used.update(new_diffs)
                            found = True
                            break
                        if not found:
                            return None
                row_slots[row_idx] = row

            if any(r is None for r in row_slots):
                return None
            return row_slots
        else:
            for row_idx in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    last = row[-1]
                    found = False

                    if strategy == 'compact':
                        search_end = last + 25 + rng.randint(0, 15)
                    elif strategy == 'spread':
                        search_end = last + 50 + rng.randint(0, 40)
                    else:
                        search_end = last + 35 + rng.randint(0, 25)

                    candidates = list(range(last + 1, min(search_end, lb + 200)))
                    if rng.random() < 0.5:
                        pass
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
                        for x in range(last + 1, lb + 300):
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

    def backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
        """
        Backtracking search to build n_rows of k+1 increasing integers starting at 0,
        with all pairwise differences distinct across all rows, and max element <= max_scope.
        """
        rng = random.Random(seed_offset * 12345 + 678)
        rows = []
        k_plus_1 = k + 1

        max_diff = max_scope
        used_diffs = [False] * (max_diff + 1)

        def build_row(pos, current_mark, row_so_far):
            if time.time() > deadline:
                return None

            if pos == k_plus_1:
                return list(row_so_far)

            valid_candidates = []
            max_x = max_scope - (k_plus_1 - pos - 1)
            if max_x <= current_mark:
                return None

            for x in range(current_mark + 1, max_x + 1):
                conflict = False
                for a in row_so_far:
                    d = x - a
                    if used_diffs[d]:
                        conflict = True
                        break
                if not conflict:
                    valid_candidates.append(x)

            if not valid_candidates:
                return None

            n_smallest = min(12, len(valid_candidates))
            smallest = valid_candidates[:n_smallest]

            if len(valid_candidates) > n_smallest and rng.random() < 0.3:
                extra = rng.sample(valid_candidates[n_smallest:], min(6, len(valid_candidates) - n_smallest))
                candidates_to_try = smallest + extra
            else:
                candidates_to_try = smallest

            rng.shuffle(candidates_to_try)

            for x in candidates_to_try:
                new_diffs = []
                for a in row_so_far:
                    d = x - a
                    new_diffs.append(d)
                    used_diffs[d] = True

                res = build_row(pos + 1, x, row_so_far + [x])
                if res is not None:
                    return res

                for d in new_diffs:
                    used_diffs[d] = False

            return None

        for i in range(n_rows):
            row = build_row(1, 0, [0])
            if row is None:
                return None
            rows.append(row)
            for j in range(k_plus_1):
                for i2 in range(j + 1, k_plus_1):
                    used_diffs[row[i2] - row[j]] = True

        return rows

    # Strategy 1: Fast greedy baseline
    best_solution = greedy_construct(42, 'basic')
    if not best_solution:
        best_solution = greedy_construct(123, 'compact')
    if not best_solution:
        best_solution = greedy_construct(456, 'spread')
    if not best_solution:
        return []

    best_scope = get_scope(best_solution)

    if best_scope <= lb:
        return best_solution

    # Strategy 2: Try to find better solutions with randomized backtracking
    # We do NOT use improve_scope as it is unsafe for global constraints.
    attempt = 0
    current_target = best_scope
    while time.time() < deadline - 8.0:
        # Try to find a solution with scope strictly less than current_target
        target_scope = current_target - 1
        if target_scope < lb:
            break

        found = False
        # Try a few times with different seeds for this target
        for _ in range(10):
            if time.time() > deadline - 8.0:
                break

            solution = backtracking_search(n_rows, k, target_scope, deadline, attempt)
            if solution:
                if validate_rows(solution):
                    new_scope = get_scope(solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = solution
                        found = True
                        # Update target to new best to continue improving
                        current_target = best_scope
                    break
            attempt += 1

        if not found:
            # If we couldn't find a better solution with target_scope, 
            # maybe try a slightly higher target? Or just break?
            # Given the time, we'll try to reduce the target by 1 only if we found one.
            # If we didn't find one, we might be stuck. 
            # Let's try to search for ANY valid solution with scope < best_scope, 
            # not just best_scope - 1.
            # Actually, backtracking_search(max_scope) finds a solution with max element <= max_scope.
            # So if we set max_scope = best_scope - 1, it finds a solution with scope <= best_scope - 1.
            # If it fails, it means no solution exists with that scope (or we ran out of time/luck).
            # We can try randomizing the search more.
            pass

    # Final validation
    if best_solution and validate_rows(best_solution):
        return best_solution
    
    # Fallback to a simple valid construction if something went wrong
    # This shouldn't happen if greedy_construct works, but just in case
    for seed in [1, 2, 3, 4, 5]:
        sol = greedy_construct(seed, 'basic')
        if sol and validate_rows(sol):
            return sol
            
    return []


# Quick self-test
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
