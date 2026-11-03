import time
import random
import sys
import math

def construct(n_rows, k):
    if n_rows <= 0:
        return []
    if k <= 0:
        return [[0] * (k + 1) for _ in range(n_rows)]

    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    m = k + 1  # entries per row
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # Lower bound for scope
    # There are total_diffs distinct differences in [1, scope].
    # So scope >= total_diffs.
    # Actually, the max difference in a row is the last element.
    # So max element >= total_diffs.
    lower_bound = total_diffs
    
    # --- Helper Functions ---
    
    def get_diffs(row):
        """Get set of all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def is_valid_solution(rows):
        """Check if the solution is valid (all diffs distinct)."""
        seen = set()
        for row in rows:
            if len(row) != m or row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            diffs = get_diffs(row)
            if len(diffs) != diffs_per_row:
                return False # Duplicates within row
            if seen.intersection(diffs):
                return False # Duplicates across rows
            seen.update(diffs)
        return True

    def greedy_row(used_diffs, max_scope, rng, bias='small'):
        """
        Greedily construct a row given used_diffs and max_scope.
        bias='small' tries to keep values small.
        Returns row or None.
        """
        row = [0]
        for _ in range(k):
            found = False
            # Determine search range
            lo = row[-1] + 1
            hi = max_scope
            
            if lo > hi:
                return None
                
            # Generate candidates
            # To make it fast, we don't iterate all if range is huge, but for scope ~134 it's fine.
            # However, for SA, we want randomness.
            if bias == 'small':
                # Try smaller values first, with some randomness
                # This is not pure greedy, but randomized greedy
                candidates = []
                # Sample some small values and some random
                # Actually, for regeneration in SA, we want to explore the space.
                # Let's just iterate from lo to hi but maybe stop early?
                # No, we need a valid row.
                
                # Efficient check: 
                # We need x such that (x - a) not in used_diffs for all a in row.
                
                # Optimization: Precompute forbidden values for next step?
                # Forbidden next values are: a + d for a in row, d in used_diffs.
                # This is expensive to compute dynamically for every step.
                # For scope ~134, simple loop is O(k * scope * used_diffs_size) which is manageable.
                
                # Let's try candidates in a randomized order biased towards smaller values
                # to encourage compactness, but allow jumps.
                
                # Strategy: 
                # 1. Try a few small values (lo to lo+5)
                # 2. Try random values
                # 3. If that fails, do full scan (slow but necessary for correctness in tight bounds)
                
                # For simplicity and robustness in SA, let's do a full scan but with random restarts/shuffling if we want to be stochastic.
                # But greedy_row is used for initialization and regeneration.
                # For regeneration in SA, we want a valid row quickly.
                
                # Let's use a simple randomized greedy:
                # Pick a random valid x from [lo, hi].
                # If we pick too large, we might fail later.
                # So bias towards smaller.
                
                attempts = 0
                while not found and attempts < 50:
                    # Sample x
                    # Probability proportional to 1/(x+1) to bias small?
                    # Or just uniform in [lo, min(hi, lo + 10)] first?
                    
                    # Let's try a mix:
                    if rng.random() < 0.5:
                        # Try small range
                        x = lo + rng.randint(0, min(10, hi - lo))
                    else:
                        # Try wider range
                        x = lo + rng.randint(0, hi - lo)
                    
                    # Check validity
                    valid = True
                    for a in row:
                        if (x - a) in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        # We don't update used_diffs here immediately to allow backtracking?
                        # No, this is a one-pass greedy. If we fail later, we return None.
                        # So we MUST update used_diffs? 
                        # Wait, if we update used_diffs, and then fail later, we have modified the global state.
                        # So greedy_row should take a COPY of used_diffs or not modify the global one until success.
                        # But copying is expensive.
                        
                        # Alternative: greedy_row returns the row, and the caller updates the global state.
                        # But we need to check against the CURRENT used_diffs + diffs of the partial row.
                        
                        # Let's refactor: greedy_row(used_diffs, ...)
                        # It checks against used_diffs.
                        # It does NOT update used_diffs.
                        # The caller is responsible for updating if the row is accepted.
                        
                        # However, to check subsequent elements, we need to consider diffs already added in this row.
                        # So we need a local set of diffs for the current row.
                        
                        # Let's rewrite the loop logic to use a local set.
                        pass 
                # The above logic is getting messy. Let's write a clean version.
                
                return None # Placeholder, will be rewritten below

    def generate_row(used_diffs, max_scope, rng):
        """
        Generate a valid row of length m starting with 0, max entry <= max_scope.
        Does not modify used_diffs.
        Returns row or None.
        """
        row = [0]
        local_diffs = set()
        
        for _ in range(k):
            lo = row[-1] + 1
            hi = max_scope
            
            if lo > hi:
                return None
            
            # Collect valid candidates
            # To speed up, we can just iterate.
            # Scope is small (~134).
            
            # Heuristic: Try smaller values first to keep scope low?
            # Or random?
            # For SA regeneration, random is better to explore.
            # For initial greedy, small is better.
            
            # Let's try a randomized approach:
            # Shuffle the range? No, that's O(N log N).
            # Just pick a random candidate?
            
            # Better: Iterate from lo to hi, but with a random start point?
            # Or just iterate from lo to hi. If we find one, take it.
            # This is "Greedy".
            
            # To make it stochastic for SA:
            # With probability p, take the first valid one.
            # With probability 1-p, take a random valid one.
            
            valid_candidates = []
            for x in range(lo, hi + 1):
                ok = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        ok = False
                        break
                if ok:
                    valid_candidates.append(x)
                    # If we want "Greedy", we break here.
                    # If we want "Random", we collect all.
                    # Let's collect all for flexibility.
            
            if not valid_candidates:
                return None
            
            # Choose a candidate
            # Bias towards smaller values to help minimize scope naturally
            # Weighted random choice?
            # Or just pick the smallest?
            # For SA, we want to explore. Let's pick randomly from the first 10 valid candidates?
            # Or just the first one (Greedy) for initialization, and Random for SA?
            
            # Let's use a parameter.
            # For now, let's pick the smallest valid one (Greedy) for initialization.
            # For SA, we will use a different function or logic.
            
            # Actually, for the "Regenerate Row" move in SA, we want to generate a DIFFERENT row.
            # If we always pick the smallest, we'll just get the same row.
            # So for SA, we should pick randomly from valid candidates.
            
            # Let's assume this function is used for Greedy Init.
            x = valid_candidates[0]
            
            row.append(x)
            for a in row[:-1]:
                local_diffs.add(x - a)
                
        return row

    def generate_row_random(used_diffs, max_scope, rng):
        """
        Generate a valid row, picking random valid candidates to encourage diversity.
        """
        row = [0]
        local_diffs = set()
        
        for _ in range(k):
            lo = row[-1] + 1
            hi = max_scope
            
            if lo > hi:
                return None
            
            valid_candidates = []
            for x in range(lo, hi + 1):
                ok = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in local_diffs:
                        ok = False
                        break
                if ok:
                    valid_candidates.append(x)
            
            if not valid_candidates:
                return None
            
            # Pick randomly
            x = rng.choice(valid_candidates)
            
            row.append(x)
            for a in row[:-1]:
                local_diffs.add(x - a)
                
        return row

    def compact_row(row, used_diffs_other, rng):
        """
        Try to reduce the scope of a single row by shifting elements left.
        used_diffs_other: set of diffs from all OTHER rows.
        Returns the compacted row.
        """
        # We want to minimize the last element.
        # We can try to move each element to the left.
        # This is a local search.
        
        # Simple approach:
        # Iterate through positions 1 to k-1.
        # Try to decrease row[i] as much as possible while maintaining validity.
        
        # However, changing row[i] affects diffs with row[j] for j>i.
        # So we should go from right to left?
        # If we change row[i], we affect diffs with 0..i-1 and i+1..k.
        # It's complex.
        
        # Alternative: 
        # Re-generate the row using a greedy that is very biased towards small values,
        # but constrained by used_diffs_other.
        # This is effectively what generate_row does.
        # If we call generate_row with used_diffs_other, we get the "greediest" row.
        # So compact_row can just be generate_row.
        
        return generate_row(used_diffs_other, max(row), rng)

    # --- Main Logic ---
    
    # 1. Initialization: Greedy
    rng = random.Random(42)
    used_diffs = set()
    rows = []
    
    for i in range(n_rows):
        # Use a large initial scope
        max_scope = lower_bound * 3 
        row = generate_row(used_diffs, max_scope, rng)
        if row is None:
            # Fallback: even larger scope
            max_scope = lower_bound * 5
            row = generate_row(used_diffs, max_scope, rng)
        if row is None:
            # Should not happen
            return [[0] * m for _ in range(n_rows)]
        
        rows.append(row)
        # Update used_diffs
        row_diffs = get_diffs(row)
        used_diffs.update(row_diffs)
        
    best_rows = [r[:] for r in rows]
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Optimization: Simulated Annealing
    # Objective: Minimize scope = max(max(r) for r in rows)
    
    current_rows = [r[:] for r in rows]
    current_used_diffs = set()
    for r in current_rows:
        current_used_diffs.update(get_diffs(r))
        
    current_scope = max(max(r) for r in current_rows)
    
    T = 10.0  # Initial temperature
    cooling_rate = 0.995
    
    # Time budget for SA
    sa_deadline = start_time + 120.0
    
    while time.time() < sa_deadline:
        if T < 0.1:
            break
            
        # Choose a move
        # Move 1: Regenerate a random row
        # Move 2: Local tweak (try to move an element left)
        
        move_type = rng.random()
        
        if move_type < 0.7:
            # Regenerate a row
            idx = rng.randint(0, n_rows - 1)
            
            # Remove current row's diffs
            old_row = current_rows[idx]
            old_diffs = get_diffs(old_row)
            for d in old_diffs:
                current_used_diffs.discard(d)
            
            # Determine max scope for new row
            # Try to keep it within current_scope or slightly less
            target_scope = current_scope - rng.randint(0, 2)
            if target_scope < lower_bound:
                target_scope = lower_bound
            
            new_row = generate_row_random(current_used_diffs, target_scope, rng)
            
            if new_row is None:
                # If it failed, try with current_scope
                new_row = generate_row_random(current_used_diffs, current_scope, rng)
            
            if new_row is not None:
                new_diffs = get_diffs(new_row)
                new_scope = max(max(r) for r in current_rows if current_rows.index(r) != idx) if n_rows > 1 else 0
                # Wait, calculating new_scope is expensive.
                # Just calculate the scope of the new set.
                
                # Tentatively apply
                current_rows[idx] = new_row
                for d in new_diffs:
                    current_used_diffs.add(d)
                
                new_total_scope = max(max(r) for r in current_rows)
                
                delta = new_total_scope - current_scope
                
                # Acceptance
                if delta < 0:
                    current_scope = new_total_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Probability of accepting worse move
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        current_scope = new_total_scope
                    else:
                        # Reject: revert
                        current_rows[idx] = old_row
                        for d in new_diffs:
                            current_used_diffs.discard(d)
                        for d in old_diffs:
                            current_used_diffs.add(d)
            else:
                # Revert
                current_rows[idx] = old_row
                for d in old_diffs:
                    current_used_diffs.add(d)
            
        else:
            # Local tweak: Pick a random row and a random element (not first, not last?)
            # Try to move it left.
            idx = rng.randint(0, n_rows - 1)
            row = current_rows[idx]
            
            # Pick a position to move
            # Moving the last element is most effective for scope reduction
            # But moving earlier elements can help.
            # Let's pick the last element (k-th index)
            pos = k
            
            old_val = row[pos]
            
            # Try to find a smaller valid value for row[pos]
            # Constraints:
            # 1. row[pos] > row[pos-1]
            # 2. row[pos] - row[i] not in used_diffs (for i < pos)
            #    Note: used_diffs here should exclude the diffs of the current row?
            #    No, the global used_diffs contains all diffs.
            #    When we change row[pos], we remove old diffs involving pos and add new ones.
            
            # Remove old diffs involving pos
            old_diffs_involved = set()
            for i in range(pos):
                d = old_val - row[i]
                old_diffs_involved.add(d)
                current_used_diffs.discard(d)
            
            # Try new values
            lo = row[pos-1] + 1
            hi = old_val - 1
            
            found = False
            new_val = None
            
            # Try values in random order? Or increasing?
            # To minimize scope, we want the smallest valid value.
            # But we are in SA, so maybe random?
            # Let's try the smallest valid value (Greedy improvement).
            # If we find one, we accept it (delta < 0).
            
            for x in range(lo, hi + 1):
                valid = True
                for i in range(pos):
                    d = x - row[i]
                    if d in current_used_diffs:
                        valid = False
                        break
                if valid:
                    new_val = x
                    found = True
                    break
            
            if found:
                # Apply change
                row[pos] = new_val
                for i in range(pos):
                    d = new_val - row[i]
                    current_used_diffs.add(d)
                
                new_total_scope = max(max(r) for r in current_rows)
                delta = new_total_scope - current_scope
                
                if delta < 0:
                    current_scope = new_total_scope
                    if current_scope < best_scope:
                        best_scope = current_scope
                        best_rows = [r[:] for r in current_rows]
                else:
                    # Revert
                    row[pos] = old_val
                    for d in old_diffs_involved:
                        current_used_diffs.add(d)
                    # Note: new_val diffs were added, need to remove them if reverting?
                    # Wait, if we revert, we must remove the new diffs we just added.
                    for i in range(pos):
                        current_used_diffs.discard(new_val - row[i])
                    # And re-add old diffs? No, we discarded them before.
                    for d in old_diffs_involved:
                        current_used_diffs.add(d)
            else:
                # Revert
                for d in old_diffs_involved:
                    current_used_diffs.add(d)
        
        T *= cooling_rate

    # 3. Final Compaction
    # Try to squeeze the best solution further using deterministic local search
    # We can try to regenerate each row with a very tight bound (best_scope)
    # and see if we can get a smaller one.
    
    final_rows = [r[:] for r in best_rows]
    final_used_diffs = set()
    for r in final_rows:
        final_used_diffs.update(get_diffs(r))
        
    # Iterate a few times
    for _ in range(10):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(n_rows):
            old_row = final_rows[i]
            old_diffs = get_diffs(old_row)
            for d in old_diffs:
                final_used_diffs.discard(d)
            
            # Try to generate a row with scope < best_scope
            # Or just greedy with current best_scope
            # If we can generate a row with max element < current max of other rows?
            # No, we just want to minimize the global max.
            
            # Try to generate a row that fits in best_scope - 1?
            # If successful, it might reduce the global max if this row was the bottleneck.
            
            # Let's just try to regenerate with best_scope and see if we get a "better" (smaller max) row.
            # Actually, generate_row is greedy and picks the smallest valid candidates.
            # So it should produce the "smallest" row possible given the constraints.
            
            new_row = generate_row(final_used_diffs, best_scope, rng)
            
            if new_row is not None:
                new_max = max(new_row)
                old_max = max(old_row)
                
                if new_max < old_max:
                    final_rows[i] = new_row
                    final_used_diffs.update(get_diffs(new_row))
                    improved = True
                else:
                    # Revert
                    final_used_diffs.update(old_diffs)
            else:
                # Revert
                final_used_diffs.update(old_diffs)
                
        if not improved:
            break
            
    # Verify final solution
    if is_valid_solution(final_rows):
        return final_rows
    else:
        # Fallback to best_rows if final_rows is invalid (should not happen)
        return best_rows

if __name__ == "__main__":
    sol = construct(4, 7)
    used = set()
    valid = True
    for row in sol:
        if len(row) != 8 or row[0] != 0:
            valid = False
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d in used:
                    valid = False
                used.add(d)
    print("n_rows=4 k=7 valid:", valid, "scope:", max(max(r) for r in sol))
