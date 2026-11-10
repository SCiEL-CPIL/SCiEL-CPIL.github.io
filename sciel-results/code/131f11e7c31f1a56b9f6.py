import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using:
    1. Backtracking with MRV heuristic on the difference-set packing view
    2. Simulated annealing for refinement
    3. Row-replacement moves for larger structural changes
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    def time_remaining():
        return deadline - time.time()

    def validate(rows):
        """Check if rows form a valid difference triangle set. Return (valid, max_entry)."""
        all_diffs = set()
        max_entry = 0
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in all_diffs:
                        return False, float('inf')
                    all_diffs.add(d)
            if row:
                max_entry = max(max_entry, row[-1])
        return True, max_entry

    def compute_used_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    used.add(row[j] - row[i])
        return used

    def greedy_initial(n_rows, k, rng=None, perturb=0):
        """
        Greedy initialization with optional randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                candidates = []
                search_limit = x + 50 + (perturb * 10)
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 20:
                        return None

                if not candidates:
                    x = search_limit
                    continue

                if rng is not None and len(candidates) > 1:
                    top = min(3, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def backtracking_construct(n_rows, k, rng, max_scope=150):
        """
        Backtracking solver using the difference-set packing view.
        Each row is a set of k*(k+1)//2 differences that must form a clique
        (all pairwise differences within the row's elements are distinct).
        
        We select the next row to fill based on MRV (most constrained variable),
        i.e., the row with the smallest remaining feasible scope.
        """
        diff_per_row = k * (k + 1) // 2
        total_diffs_needed = n_rows * diff_per_row
        
        # We'll represent each row as a list of elements
        rows = [None] * n_rows
        used_diffs = set()
        
        def get_scope(row_partial, used):
            """Estimate the scope (number of valid completions) for a partial row."""
            if row_partial is None:
                row_partial = []
            # Count how many valid candidates exist for the next element
            if len(row_partial) == 0:
                # First element must be 0
                return 1
            current_max = row_partial[-1] if row_partial else 0
            count = 0
            for x in range(current_max + 1, current_max + 50):
                valid = True
                for a in row_partial:
                    d = x - a
                    if d in used:
                        valid = False
                        break
                if valid:
                    count += 1
                    if count > 100:
                        break
            return count

        def try_complete_row(row_idx, used_diffs, max_scope):
            """Try to complete row row_idx given current used_diffs."""
            row = [0]
            x = 1
            while len(row) < k + 1:
                found = False
                # Search for valid candidates
                search_limit = x + max_scope
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    if valid:
                        # Check if this leads to a feasible completion
                        row.append(x)
                        used_diffs.update(new_diffs)
                        
                        # Recursively complete
                        if len(row) == k + 1:
                            return row, used_diffs
                        else:
                            result = try_complete_row_recursive(row, used_diffs, k, max_scope)
                            if result is not None:
                                return result
                            else:
                                # Backtrack
                                for d in new_diffs:
                                    used_diffs.discard(d)
                                row.pop()
                        
                        break
                    x += 1
                if not found:
                    return None
                x = row[-1] + 1
            return row, used_diffs

        def try_complete_row_recursive(row, used_diffs, k, max_scope):
            """Recursively complete a row."""
            if len(row) == k + 1:
                return row, used_diffs
            
            x = row[-1] + 1
            search_limit = x + max_scope
            while x < search_limit:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    
                    result = try_complete_row_recursive(row, used_diffs, k, max_scope)
                    if result is not None:
                        return result
                    
                    # Backtrack
                    for d in new_diffs:
                        used_diffs.discard(d)
                    row.pop()
                x += 1
            return None

        # Try to construct all rows using backtracking
        rows = [None] * n_rows
        used_diffs = set()
        
        # Order rows by some heuristic (e.g., try to fill rows in a balanced way)
        row_order = list(range(n_rows))
        rng.shuffle(row_order)
        
        for ri in row_order:
            if time.time() > deadline - 25:
                return None
            
            # Try to construct row ri
            row = [0]
            x = 1
            success = False
            
            while len(row) < k + 1 and not success:
                found_candidate = False
                search_limit = x + 100
                
                while x < search_limit:
                    new_diffs = set()
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        row.append(x)
                        used_diffs.update(new_diffs)
                        found_candidate = True
                        
                        # Try to complete the rest of this row
                        if len(row) == k + 1:
                            success = True
                            break
                        else:
                            # Recursively try to complete
                            result = try_complete_row_recursive(row, used_diffs, k, 50)
                            if result is not None:
                                row, used_diffs = result
                                success = True
                                break
                            else:
                                # Backtrack
                                for d in new_diffs:
                                    used_diffs.discard(d)
                                row.pop()
                    
                    x += 1
                
                if not found_candidate:
                    # Failed to find any candidate, try different approach
                    # Reset and try with larger scope
                    x = 1
                    # If we've tried enough, give up
                    if len(row) <= 1:
                        return None
                    break
            
            if not success:
                return None
            
            rows[ri] = row
        
        return rows

    def simulated_annealing(rows, n_rows, k, max_iters=5000):
        """
        Simulated annealing to minimize the largest entry.
        Moves: shift an element to a nearby valid position.
        """
        rows = [row[:] for row in rows]
        used_diffs = compute_used_diffs(rows)
        current_score = max(row[-1] for row in rows)
        best_score = current_score
        best_rows = [row[:] for row in rows]
        
        T = 50.0
        cooling = 0.9995
        
        for iteration in range(max_iters):
            if time.time() > deadline - 15:
                break
            
            # Pick a random element to move
            ri = random.randint(0, n_rows - 1)
            ci = random.randint(1, len(rows[ri]) - 1)
            row = rows[ri]
            current_val = row[ci]
            
            # Try to move to a nearby value
            delta = random.randint(-10, 10)
            new_val = current_val + delta
            
            # Check bounds
            if ci > 0 and new_val <= row[ci - 1]:
                continue
            if ci < len(row) - 1 and new_val >= row[ci + 1]:
                continue
            if new_val <= 0:
                continue
            
            # Check validity
            old_diffs = set()
            for i in range(ci):
                old_diffs.add(row[ci] - row[i])
            for j in range(ci + 1, len(row)):
                old_diffs.add(row[j] - row[ci])
            
            temp_used = used_diffs - old_diffs
            
            new_diffs = set()
            valid = True
            for i in range(ci):
                d = new_val - row[i]
                if d <= 0 or d in temp_used:
                    valid = False
                    break
                new_diffs.add(d)
            if valid:
                for j in range(ci + 1, len(row)):
                    d = row[j] - new_val
                    if d <= 0 or d in temp_used:
                        valid = False
                        break
                    new_diffs.add(d)
            
            if not valid:
                continue
            
            # Compute new score
            new_score = max(r[-1] for r in rows)
            # Temporarily update to compute new max
            old_last = row[-1]
            if ci == len(row) - 1:
                new_score = max(max(r[-1] for r in rows if r is not row), new_val)
            else:
                new_score = current_score
            
            delta_score = new_score - current_score
            
            # Accept if better or with probability exp(-delta/T)
            if delta_score < 0 or random.random() < math.exp(-delta_score / T):
                # Accept move
                rows[ri][ci] = new_val
                used_diffs = temp_used | new_diffs
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [r[:] for r in rows]
            
            T *= cooling
        
        return best_rows, best_score

    def row_replacement_move(rows, n_rows, k, rng, max_scope=150):
        """
        Rebuild an entire row against the other rows' differences.
        This enables larger structural changes than single-element shifts.
        """
        rows_copy = [row[:] for row in rows]
        
        # Pick a row to replace
        ri = rng.randint(0, n_rows - 1)
        
        # Compute used diffs from other rows
        other_used = set()
        for i in range(n_rows):
            if i == ri:
                continue
            row = rows_copy[i]
            for a in range(len(row)):
                for b in range(a + 1, len(row)):
                    other_used.add(row[b] - row[a])
        
        # Try to construct a new row for position ri
        new_row = [0]
        x = 1
        success = False
        
        while len(new_row) < k + 1:
            found = False
            search_limit = x + max_scope
            
            while x < search_limit:
                new_diffs = set()
                valid = True
                for a in new_row:
                    d = x - a
                    if d in other_used:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    new_row.append(x)
                    other_used.update(new_diffs)
                    found = True
                    break
                x += 1
            
            if not found:
                return None
            x = new_row[-1] + 1
        
        rows_copy[ri] = new_row
        return rows_copy

    # Phase 1: Multiple randomized greedy restarts
    best_rows = None
    best_score = float('inf')
    rng = random.Random(42)

    # First, deterministic greedy as baseline
    det_rows = greedy_initial(n_rows, k, rng=None)
    if det_rows is not None:
        valid, score = validate(det_rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in det_rows]

    # Randomized greedy restarts
    n_restarts = 80
    for restart in range(n_restarts):
        if time_remaining() < 30:
            break

        perturb = restart % 7
        seed_val = 42 + restart * 13
        rng_restart = random.Random(seed_val)

        rows = greedy_initial(n_rows, k, rng=rng_restart, perturb=perturb)
        if rows is None:
            continue

        valid, score = validate(rows)
        if valid and score < best_score:
            best_score = score
            best_rows = [row[:] for row in rows]

    # Try backtracking construction
    bt_rows = backtracking_construct(n_rows, k, rng, max_scope=120)
    if bt_rows is not None:
        valid, score = validate(bt_rows)
        if valid and score < best_score:
            best_score = score
            best_rows = bt_rows

    if best_rows is None:
        # Fallback: simple construction
        rows = []
        offset = 0
        for i in range(n_rows):
            row = [0]
            for j in range(1, k + 1):
                row.append(offset + j * (k + 1))
            rows.append(row)
            offset += (k + 1) * (k + 1)
        return rows

    # Phase 2: Simulated annealing refinement
    improved_rows, improved_score = simulated_annealing(best_rows, n_rows, k, max_iters=8000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    # Phase 3: Row replacement moves
    for attempt in range(50):
        if time_remaining() < 20:
            break
        
        new_rows = row_replacement_move(best_rows, n_rows, k, rng, max_scope=100)
        if new_rows is None:
            continue
        
        valid, score = validate(new_rows)
        if valid and score < best_score:
            best_score = score
            best_rows = new_rows
            
            # Refine with simulated annealing
            improved_rows, improved_score = simulated_annealing(best_rows, n_rows, k, max_iters=3000)
            if improved_score < best_score:
                best_score = improved_score
                best_rows = improved_rows

    # Phase 4: More simulated annealing with different parameters
    improved_rows, improved_score = simulated_annealing(best_rows, n_rows, k, max_iters=5000)
    if improved_score < best_score:
        best_score = improved_score
        best_rows = improved_rows

    return best_rows
# EVOLVE-BLOCK-END
