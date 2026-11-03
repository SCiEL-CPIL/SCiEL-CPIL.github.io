import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs + 1
    max_scope_bound = lb + 300  # upper bound for difference lookup

    best_rows = None
    best_scope = float('inf')

    def validate_rows(rows):
        """Check all differences are distinct across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_scope(rows):
        return max(max(r) for r in rows)

    def compute_used_set(rows):
        """Compute the set of all differences across all rows."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def _compute_all_diffs(rows):
        """Compute the full set of all positive differences across all rows."""
        s = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    s.add(row[j] - row[i])
        return s

    def greedy_construct(seed, strategy='basic'):
        """
        Construct rows using greedy with randomization.
        strategy: 'basic', 'random_order', 'compact', 'spread'
        """
        rng = random.Random(seed)
        used = set()
        rows = []

        def try_add_element(row, last, search_start, search_end):
            """Try to find a valid next element for row."""
            for x in range(search_start, search_end):
                new_diffs = [x - a for a in row]
                if any(d in used for d in new_diffs):
                    continue
                row.append(x)
                used.update(new_diffs)
                return True
            return False

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

                    search_end = min(search_end, lb + 200)

                    # Try candidates in slightly randomized order
                    candidates = list(range(last + 1, search_end))
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
                        # Wider search
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
            # Build rows in order
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

                    search_end = min(search_end, lb + 200)

                    candidates = list(range(last + 1, search_end))
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

    def local_improve(rows, seed, max_iters=200):
        """Try to reduce scope by shifting elements left."""
        rng = random.Random(seed)
        rows = [list(r) for r in rows]
        scope = get_scope(rows)

        # Maintain the used-set incrementally for speed
        used_set = _compute_all_diffs(rows)

        for _ in range(max_iters):
            if time.time() > deadline - 2:
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
                row.insert(elem_idx, old_val)
                for a in row[:elem_idx]:
                    used_set.add(old_val - a)
                for a in row[elem_idx + 1:]:
                    used_set.add(a - old_val)

        return rows, scope

    def random_restart_construct(seed):
        """Build with completely random choices."""
        rng = random.Random(seed)
        used = set()
        rows = []

        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                last = row[-1]
                found = False
                # Random jump size
                jump = rng.randint(1, 60)
                x = last + jump
                # Try this and nearby
                for candidate in range(x - 10, x + 10):
                    if candidate <= last:
                        continue
                    new_diffs = [candidate - a for a in row]
                    if any(d in used for d in new_diffs):
                        continue
                    row.append(candidate)
                    used.update(new_diffs)
                    found = True
                    break
                if not found:
                    # Try wider
                    for candidate in range(last + 1, last + 200):
                        new_diffs = [candidate - a for a in row]
                        if any(d in used for d in new_diffs):
                            continue
                        row.append(candidate)
                        used.update(new_diffs)
                        found = True
                        break
                    if not found:
                        return None
            rows.append(row)
        return rows

    # Strategy 1: Many greedy attempts with different strategies
    strategies = ['basic', 'compact', 'spread', 'random_order']
    attempt_count = 0
    for i in range(5000):
        if time.time() > deadline - 5:
            break
        strategy = strategies[i % len(strategies)]
        rows = greedy_construct(i * 31 + 7, strategy)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break
        attempt_count += 1

    # Strategy 2: Local improvement on best found
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 42, max_iters=1000)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved
        if time.time() > deadline - 5:
            return best_rows

    # Strategy 3: Random restarts
    for i in range(1500):
        if time.time() > deadline - 5:
            break
        rows = random_restart_construct(100000 + i * 97)
        if rows is not None:
            scope = get_scope(rows)
            if scope < best_scope:
                best_scope = scope
                best_rows = [list(r) for r in rows]
                if best_scope == lb:
                    break

    # Strategy 4: Local improvement on best found again
    if best_rows is not None:
        improved, imp_scope = local_improve(best_rows, 12345, max_iters=500)
        if imp_scope < best_scope:
            best_scope = imp_scope
            best_rows = improved

    # Strategy 5: Hill climbing - try to reduce any element (with incremental used-set)
    if best_rows is not None:
        rng = random.Random(99999)
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        used_set = _compute_all_diffs(current_rows)

        for iteration in range(500):
            if time.time() > deadline - 3:
                break

            row_idx = rng.randint(0, n_rows - 1)
            row = current_rows[row_idx]
            if len(row) < 2:
                continue

            elem_pos = rng.randint(1, len(row) - 1)
            old_val = row[elem_pos]
            if old_val <= 1:
                continue

            # Remove element and its diffs
            for a in row[:elem_pos]:
                used_set.discard(old_val - a)
            for a in row[elem_pos + 1:]:
                used_set.discard(a - old_val)
            row.pop(elem_pos)

            found = False
            min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
            max_val = old_val - 1

            for new_val in range(max_val, min_val - 1, -1):
                valid = True
                for a in row[:elem_pos]:
                    d = new_val - a
                    if d in used_set:
                        valid = False
                        break
                if valid:
                    for a in row[elem_pos:]:
                        d = a - new_val
                        if d in used_set:
                            valid = False
                            break
                if valid:
                    row.insert(elem_pos, new_val)
                    for a in row[:elem_pos]:
                        used_set.add(new_val - a)
                    for a in row[elem_pos + 1:]:
                        used_set.add(a - new_val)
                    found = True
                    break

            if found:
                new_scope = get_scope(current_rows)
                if new_scope < current_scope:
                    current_scope = new_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [list(r) for r in current_rows]
            else:
                row.insert(elem_pos, old_val)
                for a in row[:elem_pos]:
                    used_set.add(old_val - a)
                for a in row[elem_pos + 1:]:
                    used_set.add(a - old_val)

    # Strategy 6: Final tight compaction pass with incremental used-set
    if best_rows is not None:
        current_rows = [list(r) for r in best_rows]
        current_scope = best_scope
        used_set = _compute_all_diffs(current_rows)
        improved_any = True
        max_compaction_rounds = 200

        for round_num in range(max_compaction_rounds):
            if time.time() > deadline - 2:
                break
            if not improved_any:
                break
            improved_any = False

            for row_idx in range(n_rows):
                if time.time() > deadline - 2:
                    break
                row = current_rows[row_idx]
                for elem_pos in range(len(row) - 1, 0, -1):
                    old_val = row[elem_pos]
                    if old_val <= 1:
                        continue

                    # Remove element and its diffs from used_set
                    for a in row[:elem_pos]:
                        used_set.discard(old_val - a)
                    for a in row[elem_pos + 1:]:
                        used_set.discard(a - old_val)
                    row.pop(elem_pos)

                    found = False
                    min_val = row[elem_pos - 1] + 1 if elem_pos > 0 else 1
                    max_val = old_val - 1

                    for new_val in range(max_val, min_val - 1, -1):
                        valid = True
                        for a in row[:elem_pos]:
                            d = new_val - a
                            if d in used_set:
                                valid = False
                                break
                        if valid:
                            for a in row[elem_pos:]:
                                d = a - new_val
                                if d in used_set:
                                    valid = False
                                    break
                        if valid:
                            row.insert(elem_pos, new_val)
                            for a in row[:elem_pos]:
                                used_set.add(new_val - a)
                            for a in row[elem_pos + 1:]:
                                used_set.add(a - new_val)
                            found = True
                            break

                    if found:
                        new_scope = get_scope(current_rows)
                        if new_scope < current_scope:
                            current_scope = new_scope
                            improved_any = True
                            if current_scope < best_scope:
                                best_scope = current_scope
                                best_rows = [list(r) for r in current_rows]
                    else:
                        row.insert(elem_pos, old_val)
                        for a in row[:elem_pos]:
                            used_set.add(old_val - a)
                        for a in row[elem_pos + 1:]:
                            used_set.add(a - old_val)

    # Strategy 7: Constraint-based row reconstruction via difference set partitioning
    # This is a structural change: instead of greedy element-by-element construction,
    # we try to regenerate entire rows by searching over difference sets that can
    # be partitioned into valid sequences.
    
    if best_rows is not None and time.time() < deadline - 10:
        def can_form_row(diff_set, max_val):
            """
            Given a set of differences, check if they can form a valid row
            of k+1 elements starting at 0 with max value <= max_val.
            Uses backtracking to build the row.
            """
            if len(diff_set) != k * (k + 1) // 2:
                return None
            
            # Quick check: the largest difference must be the max element
            max_diff = max(diff_set)
            if max_diff > max_val:
                return None
            
            # The second largest difference should be close to max_diff
            # (since max - second_smallest = second_largest_diff)
            
            # Backtracking: build the row element by element
            # row[0] = 0, we need to find row[1], ..., row[k]
            
            # Use the fact that all differences must be in diff_set
            # and each difference must be used exactly once
            
            # This is a CSP: assign values to positions 1..k such that
            # all pairwise differences are in diff_set and distinct
            
            # Heuristic: the largest element is max_diff (since 0 is in the row)
            # So row[k] = max_diff
            # The second largest difference is max_diff - row[1]
            
            # Let's use a different approach: recursive search with pruning
            # State: current row (list), remaining diffs (set)
            # At each step, try to add a new element
            
            # For efficiency, we'll use a limited search
            row = [0]
            remaining = set(diff_set)
            
            # The last element must be max_diff
            # So we need to build up to max_diff
            
            # Actually, let's use a simpler heuristic-based approach:
            # 1. The row must contain 0 and max_diff
            # 2. The differences from 0 to each element are in diff_set
            # 3. The differences between consecutive elements are in diff_set
            
            # Let's try a randomized local search on the difference set
            # to see if it can form a valid row
            
            # For now, return None to indicate we don't have a fast check
            return None

        def reconstruct_row_from_diffs(diff_set, rng, max_val, max_attempts=100):
            """
            Try to reconstruct a valid row from a given difference set.
            Returns the row if successful, None otherwise.
            """
            if len(diff_set) != k * (k + 1) // 2:
                return None
            
            max_diff = max(diff_set)
            if max_diff > max_val:
                return None
            
            # The row must end with max_diff (since 0 is the first element)
            # Let's try to build the row backwards from max_diff
            
            # Actually, let's use a forward search with the constraint that
            # the last element is max_diff
            
            # This is complex, so let's use a simpler approach:
            # Try random permutations of the differences and see if any
            # can form a valid row
            
            # For efficiency, we'll use a heuristic:
            # The differences from 0 to each element must be in diff_set
            # Let's find all subsets of diff_set of size k that could be
            # the elements of the row (including 0)
            
            # This is still complex. Let's just return None for now
            # and rely on the other strategies.
            
            return None

        # For now, this strategy is a placeholder. The key insight is that
        # we should be thinking in terms of difference sets rather than
        # individual elements. A full implementation would use backtracking
        # or dynamic programming to validate difference sets.
        
        # Instead, let's use a different structural approach:
        # Simulated annealing on the row elements with a global difference constraint.
        
        def simulated_annealing(rows, seed, iterations=5000, t_start=100, t_end=1):
            """
            Use simulated annealing to optimize the rows.
            Moves: change one element in one row.
            Objective: minimize scope while maintaining validity.
            """
            rng = random.Random(seed)
            rows = [list(r) for r in rows]
            current_scope = get_scope(rows)
            best_rows_sa = [list(r) for r in rows]
            best_scope_sa = current_scope
            
            # Precompute used set
            used_set = _compute_all_diffs(rows)
            
            def row_diffs(row):
                """Get all differences in a row."""
                diffs = set()
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        diffs.add(row[j] - row[i])
                return diffs
            
            for it in range(iterations):
                if time.time() > deadline - 3:
                    break
                
                # Temperature
                t = t_start * (t_end / t_start) ** (it / iterations)
                
                # Choose a random row and element to move
                row_idx = rng.randint(0, n_rows - 1)
                elem_pos = rng.randint(1, len(rows[row_idx]) - 1)
                old_val = rows[row_idx][elem_pos]
                
                # Generate a new value
                # Bias towards smaller values to reduce scope
                if rng.random() < 0.7:
                    # Try to decrease
                    delta = rng.randint(1, max(1, old_val // 3))
                    new_val = old_val - delta
                    if new_val <= rows[row_idx][elem_pos - 1]:
                        new_val = rows[row_idx][elem_pos - 1] + 1
                else:
                    # Try to increase slightly
                    delta = rng.randint(1, 10)
                    new_val = old_val + delta
                
                if new_val == old_val:
                    continue
                
                # Check validity
                # Remove old differences
                old_diffs = set()
                for i in range(len(rows[row_idx])):
                    if i == elem_pos:
                        continue
                    if i < elem_pos:
                        old_diffs.add(old_val - rows[row_idx][i])
                    else:
                        old_diffs.add(rows[row_idx][i] - old_val)
                
                # Check if new value is valid
                valid = True
                new_diffs = set()
                for i in range(len(rows[row_idx])):
                    if i == elem_pos:
                        continue
                    if i < elem_pos:
                        d = new_val - rows[row_idx][i]
                    else:
                        d = rows[row_idx][i] - new_val
                    if d <= 0:
                        valid = False
                        break
                    if d in used_set:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if not valid:
                    continue
                
                # Check if new differences conflict with other rows
                # (they shouldn't, since used_set contains all differences)
                
                # Accept if better or with probability exp(-delta/t)
                new_scope = max(max(r) for r in rows)
                delta_obj = new_scope - current_scope
                
                if delta_obj <= 0 or rng.random() < (1.0 / (1.0 + delta_obj * 0.1)):
                    # Apply the move
                    rows[row_idx][elem_pos] = new_val
                    used_set.difference_update(old_diffs)
                    used_set.update(new_diffs)
                    current_scope = new_scope
                    
                    if current_scope < best_scope_sa:
                        best_scope_sa = current_scope
                        best_rows_sa = [list(r) for r in rows]
                else:
                    # Reject, do nothing
                    pass
            
            return best_rows_sa, best_scope_sa

        # Run simulated annealing
        if time.time() < deadline - 5:
            sa_rows, sa_scope = simulated_annealing(best_rows, 777, iterations=3000)
            if sa_scope < best_scope:
                best_scope = sa_scope
                best_rows = sa_rows

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
