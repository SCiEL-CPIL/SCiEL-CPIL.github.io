"""Difference triangle set via randomized greedy and simulated annealing with row regeneration."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Construct a difference triangle set using randomized greedy initialization
    and simulated annealing with row-regeneration moves to minimize the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

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

    def compute_row_diffs(row):
        """Compute the set of differences for a single row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def compute_all_diffs(rows):
        """Compute the set of all used differences."""
        used = set()
        for row in rows:
            used.update(compute_row_diffs(row))
        return used

    def greedy_initial(n_rows, k, rng, perturb=0):
        """
        Greedy initialization with randomization.
        """
        used_diffs = set()
        rows = []
        for r_idx in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                if time.time() > deadline - 5:
                    return None
                # Find all valid candidates in a window
                candidates = []
                # Dynamic window size
                search_limit = x + 20 + (perturb * 10)
                
                # If we are stuck, expand
                if not candidates and x > 1:
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
                        candidates.append((x, new_diffs))
                    x += 1
                    if time.time() > deadline - 5:
                        return None

                if not candidates:
                    # Expand search significantly
                    x = search_limit
                    continue

                # Choose candidate
                if len(candidates) > 1 and rng.random() < 0.7:
                    # Pick randomly from top 5 for diversity
                    top = min(5, len(candidates))
                    chosen_idx = rng.randint(0, top - 1)
                    chosen_val, chosen_diffs = candidates[chosen_idx]
                else:
                    chosen_val, chosen_diffs = candidates[0]

                row.append(chosen_val)
                used_diffs.update(chosen_diffs)
                x = chosen_val + 1

            rows.append(row)
        return rows

    def reconstruct_row(rng, blocked_diffs, k, temp, current_row=None):
        """
        Reconstruct a row given blocked differences.
        Uses randomized greedy.
        """
        row = [0]
        x = 1
        # Start search from 1
        while len(row) < k + 1:
            if time.time() > deadline - 5:
                return None
            
            candidates = []
            # Search window
            # If temp is high, look wider? No, just pick randomly from valid ones.
            search_limit = x + 50 
            
            while x < search_limit:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in blocked_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
                if time.time() > deadline - 5:
                    return None

            if not candidates:
                # Expand
                x = search_limit
                continue

            # Selection strategy based on temperature
            if temp > 0.1 and len(candidates) > 1:
                # Random selection, weighted towards smaller values slightly?
                # Just uniform random from top 10 for diversity
                top = min(10, len(candidates))
                chosen_idx = rng.randint(0, top - 1)
                chosen_val, chosen_diffs = candidates[chosen_idx]
            else:
                # Greedy: smallest
                chosen_val, chosen_diffs = candidates[0]

            row.append(chosen_val)
            # Note: we don't update blocked_diffs here because this row's internal diffs
            # are valid by construction (we checked against blocked_diffs which excludes
            # the row's own diffs, and we ensure new_diffs are distinct within the row
            # implicitly? No, we must ensure new_diffs don't conflict with existing row diffs.
            # Wait, the `valid` check above only checks against `blocked_diffs`.
            # It does NOT check against `row`'s existing diffs.
            # We need to ensure `new_diffs` are distinct from each other and from `row`'s diffs.
            # Actually, `row` is being built. `new_diffs` are diffs between `x` and `row[0..i]`.
            # We need to ensure `new_diffs` are not in `blocked_diffs` AND not in `set(compute_row_diffs(row))`.
            
            # Let's refine the check inside the loop.
            # The current loop logic is flawed for this function.
            # Let's rewrite the inner loop of reconstruct_row properly.
            break # Break out to rewrite logic properly below

        # Rewrite reconstruct_row logic to be correct
        row = [0]
        used_in_row = set()
        x = 1
        while len(row) < k + 1:
            if time.time() > deadline - 5:
                return None
            
            candidates = []
            search_limit = x + 100
            
            while x < search_limit:
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in blocked_diffs or d in used_in_row:
                        valid = False
                        break
                    new_diffs.add(d)
                if valid:
                    candidates.append((x, new_diffs))
                x += 1
                if time.time() > deadline - 5:
                    return None

            if not candidates:
                x = search_limit
                continue

            if temp > 0.1 and len(candidates) > 1:
                top = min(10, len(candidates))
                chosen_idx = rng.randint(0, top - 1)
                chosen_val, chosen_diffs = candidates[chosen_idx]
            else:
                chosen_val, chosen_diffs = candidates[0]

            row.append(chosen_val)
            used_in_row.update(chosen_diffs)
            x = chosen_val + 1

        return row

    # Main Search
    best_result = None
    best_score = float('inf')
    rng = random.Random(42)

    # 1. Initial Greedy
    det_rows = greedy_initial(n_rows, k, rng, perturb=0)
    if det_rows:
        valid, max_entry = validate(det_rows)
        if valid:
            best_result = [r[:] for r in det_rows]
            best_score = max_entry

    # 2. Simulated Annealing with Row Regeneration
    if best_result:
        rows = [r[:] for r in best_result]
        all_diffs = compute_all_diffs(rows)
        T = 10.0
        T_min = 0.01
        cooling = 0.95
        
        # Pre-compute row diffs for efficiency
        row_diffs = [compute_row_diffs(r) for r in rows]
        
        current_max = max(r[-1] for r in rows)
        
        while time.time() < deadline - 10:
            if T < T_min:
                # Restart with higher T if stuck? Or just break
                T = T_min
                # Optional: random restart of one row
                r_idx = rng.randint(0, n_rows - 1)
                blocked = all_diffs - row_diffs[r_idx]
                new_row = reconstruct_row(rng, blocked, k, 1.0, rows[r_idx])
                if new_row:
                    rows[r_idx] = new_row
                    row_diffs[r_idx] = compute_row_diffs(new_row)
                    all_diffs = compute_all_diffs(rows)
                    current_max = max(r[-1] for r in rows)
                    if current_max < best_score:
                        best_score = current_max
                        best_result = [r[:] for r in rows]
                continue

            # Pick a row to regenerate
            r_idx = rng.randint(0, n_rows - 1)
            
            # Calculate blocked diffs (all diffs except those in row r_idx)
            blocked = all_diffs - row_diffs[r_idx]
            
            # Reconstruct
            new_row = reconstruct_row(rng, blocked, k, T, rows[r_idx])
            
            if new_row is None:
                T *= cooling
                continue
            
            new_row_diffs = compute_row_diffs(new_row)
            
            # Check validity against blocked (should be valid by construction, but double check)
            if new_row_diffs & blocked:
                T *= cooling
                continue
            
            # Calculate new global max
            # We can optimize this by keeping a max-heap or just recalculating for n_rows=4
            new_max = max(rows[i][-1] if i != r_idx else new_row[-1] for i in range(n_rows))
            
            delta_E = new_max - current_max
            
            # Metropolis Acceptance
            if delta_E < 0 or rng.random() < math.exp(-delta_E / T):
                # Accept
                rows[r_idx] = new_row
                row_diffs[r_idx] = new_row_diffs
                # Update all_diffs
                all_diffs = all_diffs - row_diffs[r_idx] # Wait, this is wrong. 
                # all_diffs was the union. 
                # We should rebuild all_diffs or update it.
                # Since n_rows is small, rebuilding is cheap.
                all_diffs = compute_all_diffs(rows)
                
                current_max = new_max
                
                if current_max < best_score:
                    best_score = current_max
                    best_result = [r[:] for r in rows]
            
            T *= cooling

    if best_result is None:
        # Fallback: simple greedy
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    return best_result
# EVOLVE-BLOCK-END
