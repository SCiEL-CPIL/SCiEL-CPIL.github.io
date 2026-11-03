import time
import random
import math

def construct(n_rows, k):
    row_len = k + 1
    # Estimate a safe maximum value for difference array indexing
    # Based on the problem description, scope is around 105-112 for (7,5).
    # Let's set a generous bound to be safe for other n_rows, k if passed.
    # n_rows * k * (k+1) / 2 differences total.
    # If they are distinct, max diff is at least sqrt(2 * total_diffs).
    # But we are minimizing max entry.
    # Let's use a dynamic bound or a large enough fixed bound.
    # For n_rows=7, k=5, total diffs = 7 * 15 = 105.
    # Max entry 112 means diffs up to 112.
    # Let's set MAX_DIFF = 500 to be safe for small inputs, or calculate based on greedy.
    
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < row_len:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    # Initial solution
    best_rows = greedy_construct()
    best_max = max(r[-1] for r in best_rows)
    
    # Determine a safe upper bound for differences for array-based counting
    # We can use the current best_max plus some margin
    max_diff_bound = best_max + 50
    
    def get_diffs(row):
        diffs = []
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.append(row[j] - row[i])
        return diffs

    def build_counts(rows, bound):
        counts = [0] * (bound + 1)
        for r in rows:
            for d in get_diffs(r):
                if d <= bound:
                    counts[d] += 1
                else:
                    # Should not happen if bound is sufficient
                    pass
        return counts

    def count_collisions(counts):
        coll = 0
        for c in counts:
            if c > 1:
                coll += c - 1
        return coll

    def scope_of(rows):
        return max(r[-1] for r in rows)

    def sum_of_entries(rows):
        return sum(sum(r) for r in rows)

    def generate_candidate_row(used_counts, max_val, rng, randomize_factor=0.1):
        """
        Generates a row of length row_len starting at 0.
        Ensures no differences conflict with used_counts.
        Tries to keep values small.
        """
        row = [0]
        local_diffs = set()
        last = 0
        
        for _ in range(row_len - 1):
            found = False
            # We want to try small values first, but allow some randomness
            # Start from last+1
            start = last + 1
            # To allow some variation, we might skip the very first few valid options
            # Or pick a random valid option from a small range of valids
            
            # Collect a few valid candidates to choose from
            candidates = []
            for x in range(start, max_val + 1):
                new_diffs = {x - a for a in row}
                # Check internal conflicts (new_diffs should be unique within row, which is true if x > last)
                # Check external conflicts with used_counts
                conflict = False
                for d in new_diffs:
                    if used_counts[d] > 0:
                        conflict = True
                        break
                if not conflict:
                    candidates.append(x)
                    if len(candidates) > 10: # Don't collect too many
                        break
                
            if not candidates:
                return None
            
            # Choose a candidate
            # Bias towards smaller values
            if rng.random() < randomize_factor:
                # Pick randomly from first 5 candidates
                pick_idx = min(len(candidates) - 1, rng.randint(0, 4))
            else:
                pick_idx = 0
            
            x = candidates[pick_idx]
            row.append(x)
            last = x
            
        return row

    def polish(rows, max_time=2.0):
        """
        Try to reduce scope by decrementing entries.
        """
        start_t = time.time()
        rows = [r[:] for r in rows]
        
        # Ensure valid first
        all_diffs = []
        for r in rows:
            for d in get_diffs(r):
                all_diffs.append(d)
        if len(set(all_diffs)) != len(all_diffs):
            return rows
            
        improved = True
        while improved and (time.time() - start_t) < max_time:
            improved = False
            for r_idx in range(n_rows):
                for i_idx in range(row_len - 1, 0, -1):
                    old_val = rows[r_idx][i_idx]
                    new_val = old_val - 1
                    if new_val <= rows[r_idx][i_idx - 1]:
                        continue
                    if i_idx < row_len - 1 and new_val >= rows[r_idx][i_idx + 1]:
                        continue
                    
                    # Check validity
                    # Remove old diffs, add new diffs, check for duplicates
                    # Efficient check:
                    # 1. Identify diffs that change
                    # 2. Check if any new diff already exists elsewhere
                    
                    # Simple full check is O(n_rows * k^2) which is small
                    # But let's do a partial check
                    row = rows[r_idx]
                    old_diffs = get_diffs(row)
                    row[i_idx] = new_val
                    new_diffs = get_diffs(row)
                    
                    # Create a set of all diffs from other rows
                    other_diffs = set()
                    for j in range(n_rows):
                        if j != r_idx:
                            for d in get_diffs(rows[j]):
                                other_diffs.add(d)
                    
                    # Check if new_diffs has any duplicates within itself or with other_diffs
                    # Within itself: since row is strictly increasing, diffs are unique?
                    # No, e.g., [0, 2, 4] has diffs 2, 2, 4. Wait.
                    # [0, 2, 4]: 2-0=2, 4-2=2, 4-0=4. Diffs: 2, 2, 4.
                    # The problem says "all differences ... are distinct over all rows".
                    # This implies diffs within a row must also be distinct?
                    # "all differences between two entries of the same row are distinct over all rows"
                    # This phrasing usually means the set of all differences from all rows is a set (no duplicates).
                    # So yes, within a row, diffs must be distinct too.
                    
                    if len(set(new_diffs)) != len(new_diffs):
                        row[i_idx] = old_val
                        continue
                        
                    # Check against other rows
                    conflict = False
                    for d in new_diffs:
                        if d in other_diffs:
                            conflict = True
                            break
                            
                    if not conflict:
                        improved = True
                    else:
                        row[i_idx] = old_val
                        
        return rows

    # Main SA Loop
    deadline = time.time() + 115.0
    rng = random.Random(42)
    
    current_rows = [r[:] for r in best_rows]
    current_max = scope_of(current_rows)
    current_sum = sum_of_entries(current_rows)
    
    # Initialize counts
    # We need a bound large enough. Let's use current_max + 20
    max_bound = current_max + 20
    counts = build_counts(current_rows, max_bound)
    current_coll = count_collisions(counts)
    
    # Cost function
    # We want to minimize Scope.
    # If Collisions > 0, we are invalid. We want to drive collisions to 0 first.
    # So Cost = Collisions * 10000 + Scope + 0.1 * Sum
    
    def get_cost(rows, coll, s):
        scope = scope_of(rows)
        return coll * 10000 + scope + 0.1 * s

    current_cost = get_cost(current_rows, current_coll, current_sum)
    best_cost = current_cost
    best_rows_sa = [r[:] for r in current_rows]
    
    T = 10.0
    cooling = 0.9999
    min_T = 0.1
    
    while time.time() < deadline - 5.0:
        # Pick a random row to rebuild
        r_idx = rng.randint(0, n_rows - 1)
        old_row = current_rows[r_idx]
        
        # Remove old row's diffs from counts
        for d in get_diffs(old_row):
            if d <= max_bound:
                counts[d] -= 1
        
        # Generate new row
        # We try to generate a row that is valid with respect to the other rows.
        # We pass the current counts (which excludes old_row)
        new_row = generate_candidate_row(counts, max_bound, rng)
        
        if new_row is None:
            # Failed to generate, restore old row
            for d in get_diffs(old_row):
                if d <= max_bound:
                    counts[d] += 1
            continue
            
        # Calculate new metrics
        new_diffs = get_diffs(new_row)
        
        # Check if new_row has internal distinct diffs
        if len(set(new_diffs)) != len(new_diffs):
            # Invalid row, restore
            for d in get_diffs(old_row):
                if d <= max_bound:
                    counts[d] += 1
            continue
            
        # Check for conflicts with other rows (counts currently holds diffs of other rows)
        new_coll_delta = 0
        for d in new_diffs:
            if d <= max_bound:
                if counts[d] > 0:
                    new_coll_delta += 1 # This diff will become a collision if counts[d] was >= 1
                    # Actually, if counts[d] is 1, it becomes 2 (1 collision). If counts[d] is 0, it becomes 1 (0 collisions).
                    # So delta in collisions is 1 if counts[d] >= 1.
                    # Wait, collision count is sum(max(0, c-1)).
                    # If c=0, new c=1, delta coll = 0.
                    # If c=1, new c=2, delta coll = 1.
                    # If c=2, new c=3, delta coll = 1.
                    # So yes, delta is 1 if counts[d] >= 1.
            else:
                # Diff out of bounds, shouldn't happen if max_bound is sufficient
                pass
        
        # Update counts with new row
        for d in new_diffs:
            if d <= max_bound:
                counts[d] += 1
                
        new_max = scope_of([current_rows[i] if i != r_idx else new_row for i in range(n_rows)])
        # More efficient:
        other_maxes = [current_rows[i][-1] for i in range(n_rows) if i != r_idx]
        new_max = max([new_row[-1]] + other_maxes)
        
        new_sum = sum_of_entries([current_rows[i] if i != r_idx else new_row for i in range(n_rows)])
        # Efficient sum:
        old_row_sum = sum(old_row)
        new_row_sum = sum(new_row)
        new_sum = current_sum - old_row_sum + new_row_sum
        
        new_coll = current_coll + new_coll_delta
        new_cost = get_cost([current_rows[i] if i != r_idx else new_row for i in range(n_rows)], new_coll, new_sum)
        
        # Accept/Reject
        delta = new_cost - current_cost
        if delta < 0 or (T > min_T and rng.random() < math.exp(-delta / T)):
            # Accept
            current_rows[r_idx] = new_row
            current_sum = new_sum
            current_coll = new_coll
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_rows_sa = [r[:] for r in current_rows]
                
        else:
            # Reject, restore old row in counts
            for d in new_diffs:
                if d <= max_bound:
                    counts[d] -= 1
            for d in get_diffs(old_row):
                if d <= max_bound:
                    counts[d] += 1
            
        T *= cooling
        if T < min_T:
            T = min_T

    # Final Polish
    if time.time() < deadline - 2.0:
        polished = polish(best_rows_sa, max_time=min(3.0, deadline - time.time() - 2.0))
        if polished:
            pol_max = scope_of(polished)
            if pol_max < scope_of(best_rows_sa):
                best_rows_sa = polished

    return best_rows_sa
