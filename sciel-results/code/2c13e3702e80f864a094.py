import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit

    # Total number of differences that must be distinct
    total_diffs = n_rows * k * (k + 1) // 2

    # Lower bound on scope: M(M+1)/2 >= total_diffs
    # M(M+1) >= 2*total_diffs
    # M >= (-1 + sqrt(1 + 8*total_diffs)) / 2
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)

    best_rows = None
    best_scope = float('inf')

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def is_valid_and_scope(rows):
        """
        Returns (is_valid, scope, conflicts_count, diffs_set)
        """
        used_diffs = set()
        valid = True
        scope = 0
        conflicts = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used_diffs:
                        valid = False
                        conflicts += 1
                    else:
                        used_diffs.add(d)
            if max(row) > scope:
                scope = max(row)
        return valid, scope, conflicts, used_diffs

    def get_cost(rows):
        """
        Cost = scope if valid, else scope + penalty * conflicts
        """
        valid, scope, conflicts, _ = is_valid_and_scope(rows)
        if valid:
            return scope
        else:
            return scope + 100 * conflicts

    def greedy_construct():
        """Greedy construction: for each row, pick the smallest valid next element."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    def greedy_random_construct():
        """Randomized greedy: pick from a small set of valid candidates to diversify."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Look ahead for a few valid candidates
                candidates = []
                search_limit = x + 20
                while x < search_limit:
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                        if len(candidates) >= 8:
                            break
                
                if not candidates:
                    # Fallback: expand window
                    while True:
                        x += 1
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                            break
                        if x > 1000:
                            return None
                
                # Pick a random candidate, biased towards smaller ones
                weights = [1.0 / ((i + 1) * (i + 1)) for i in range(len(candidates))]
                total_w = sum(weights)
                r = random.random() * total_w
                acc = 0
                chosen_idx = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen_idx = i
                        break
                
                chosen_val = candidates[chosen_idx]
                row.append(chosen_val)
                used_diffs.update({chosen_val - a for a in row[:-1]})
                x = chosen_val - 1  # Next search starts from here
            
            rows.append(row)
        return rows

    # Phase 1: Pure greedy
    greedy_rows = greedy_construct()
    greedy_scope = get_scope(greedy_rows)
    if is_valid(greedy_rows):
        best_rows = greedy_rows
        best_scope = greedy_scope

    # Phase 2: Randomized greedy with restarts
    while time.time() < deadline - 10:
        rows = greedy_random_construct()
        if rows is None:
            continue
        if is_valid(rows):
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [r[:] for r in rows]
                if best_scope <= lower_bound:
                    return best_rows

    # Phase 3: Local search on the best solution found
    if best_rows is not None and time.time() < deadline - 2:
        current_rows = [r[:] for r in best_rows]
        current_scope = best_scope
        current_valid = True
        current_conflicts = 0
        current_cost = get_cost(current_rows)

        T = 50.0
        alpha = 0.997
        min_T = 0.5
        steps = 0
        no_improve_steps = 0
        max_no_improve = 2000

        def get_row_diffs(row):
            diffs = set()
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    diffs.add(row[j] - row[i])
            return diffs

        while time.time() < deadline - 2:
            if no_improve_steps > max_no_improve:
                # Reheat
                T = 15.0
                no_improve_steps = 0

            # Generate Move
            # Pick random row and index
            r_idx = random.randint(0, n_rows - 1)
            i_idx = random.randint(1, k)  # 1 to k (0 is fixed)

            # Generate new value
            delta_range = random.randint(1, 15)
            delta = random.randint(-delta_range, delta_range)
            new_val = current_rows[r_idx][i_idx] + delta

            # Check bounds
            lower_bound_idx = current_rows[r_idx][i_idx - 1] + 1
            if i_idx < k:
                upper_bound_idx = current_rows[r_idx][i_idx + 1] - 1
            else:
                upper_bound_idx = current_rows[r_idx][i_idx] + 30  # Allow expansion

            if new_val < lower_bound_idx or new_val > upper_bound_idx:
                continue

            # Apply move
            old_val = current_rows[r_idx][i_idx]
            current_rows[r_idx][i_idx] = new_val

            # Calculate new cost
            new_valid, new_scope, new_conflicts, _ = is_valid_and_scope(current_rows)
            new_cost = get_cost(current_rows)

            # Delta
            delta_cost = new_cost - current_cost

            # Accept/Reject
            if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
                current_cost = new_cost
                current_scope = new_scope
                current_valid = new_valid
                current_conflicts = new_conflicts

                if current_valid and current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    no_improve_steps = 0
                    if best_scope <= lower_bound:
                        return best_rows
                else:
                    no_improve_steps += 1
            else:
                # Reject, revert
                current_rows[r_idx][i_idx] = old_val
                no_improve_steps += 1

            T *= alpha
            if T < min_T:
                T = 15.0

            steps += 1

            # Occasionally, try a "Row Regeneration" move
            if steps % 500 == 0 and random.random() < 0.1:
                reg_idx = random.randint(0, n_rows - 1)
                # Compute diffs of other rows
                other_diffs = set()
                for i, r in enumerate(current_rows):
                    if i == reg_idx:
                        continue
                    other_diffs.update(get_row_diffs(r))

                # Generate new row using greedy_random logic but constrained by other_diffs
                new_row = [0]
                x = 0
                success = True
                while len(new_row) < k + 1:
                    candidates = []
                    search_limit = x + 20
                    while x < search_limit:
                        x += 1
                        new_diffs = {x - a for a in new_row}
                        if not (new_diffs & other_diffs):
                            candidates.append(x)
                            if len(candidates) >= 8:
                                break

                    if not candidates:
                        while True:
                            x += 1
                            new_diffs = {x - a for a in new_row}
                            if not (new_diffs & other_diffs):
                                candidates.append(x)
                                break
                            if x > 1000:
                                success = False
                                break
                        if not success:
                            break

                    weights = [1.0 / ((i + 1) * (i + 1)) for i in range(len(candidates))]
                    total_w = sum(weights)
                    r = random.random() * total_w
                    acc = 0
                    chosen_idx = 0
                    for i, w in enumerate(weights):
                        acc += w
                        if r <= acc:
                            chosen_idx = i
                            break

                    chosen_val = candidates[chosen_idx]
                    new_row.append(chosen_val)
                    x = chosen_val - 1

                if success:
                    # Check if the new row is valid with the other rows
                    test_rows = [r[:] for r in current_rows]
                    test_rows[reg_idx] = new_row
                    if is_valid(test_rows):
                        new_scope = get_scope(test_rows)
                        if new_scope < best_scope:
                            best_scope = new_scope
                            best_rows = [r[:] for r in test_rows]
                        current_rows = test_rows
                        current_scope = new_scope
                        current_valid = True
                        current_conflicts = 0
                        current_cost = new_scope
                        no_improve_steps = 0

    return best_rows
