import random
import time
import math

def construct(n_rows, k):
    """
    Constructs a difference triangle set with n_rows rows of k+1 integers.
    Minimizes the largest entry using optimized Simulated Annealing.
    """
    start_time = time.time()
    time_limit = 115  # Leave 5s buffer for safety
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def greedy_init(max_val=None):
        """Generate an initial valid set using greedy method."""
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                x = row[-1] + 1
                while True:
                    if max_val is not None and x > max_val:
                        return None # Failed to fit within max_val
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used_diffs):
                        break
                    x += 1
                row.append(x)
                used_diffs.update({x - a for a in row[:-1]})
            rows.append(row)
        return rows

    def compute_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def solve():
        # Initialize with best of several greedy attempts
        best_init_rows = None
        best_init_scope = float('inf')
        
        # Try a few greedy inits with different random seeds or just one
        # Greedy is deterministic if we always pick smallest x.
        # To vary, we can pick random valid x in greedy.
        for _ in range(5):
            rows = []
            used_diffs = set()
            valid = True
            for _ in range(n_rows):
                row = [0]
                while len(row) < k + 1:
                    # Pick a random candidate in a window to diversify
                    start_x = row[-1] + 1
                    # Look ahead a bit
                    candidates = []
                    x = start_x
                    while x < start_x + 20 and len(candidates) < 5:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used_diffs):
                            candidates.append(x)
                        x += 1
                    if not candidates:
                        # Fallback to greedy smallest
                        x = start_x
                        while True:
                            new_diffs = {x - a for a in row}
                            if not (new_diffs & used_diffs):
                                candidates.append(x)
                                break
                            x += 1
                    
                    chosen = random.choice(candidates)
                    row.append(chosen)
                    used_diffs.update({chosen - a for a in row[:-1]})
                rows.append(row)
            
            scope = get_scope(rows)
            if scope < best_init_scope:
                best_init_scope = scope
                best_init_rows = rows[:]
        
        rows = best_init_rows
        used_diffs = set()
        for row in rows:
            used_diffs.update(compute_row_diffs(row))
        
        best_rows = rows[:]
        best_scope = get_scope(rows)
        current_scope = best_scope
        
        # Simulated Annealing
        T = 1.0
        T_min = 0.01
        cooling = 0.999
        max_iters = 500000
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit:
                break
            
            # Move: Pick a random row, regenerate it
            idx = random.randint(0, n_rows - 1)
            old_row = rows[idx]
            old_diffs = compute_row_diffs(old_row)
            
            # Remove old diffs
            for d in old_diffs:
                used_diffs.discard(d)
            
            # Generate new row
            # Bias towards smaller numbers
            current_max = max(max(r) for r in rows)
            # Allow expansion slightly
            limit = current_max + 5
            
            new_row = [0]
            valid = True
            for _ in range(k):
                candidates = []
                x = new_row[-1] + 1
                # Scan for candidates
                # To speed up, don't scan too far
                while x < limit and len(candidates) < 10:
                    new_diffs = {x - a for a in new_row}
                    if not (new_diffs & used_diffs):
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    valid = False
                    break
                
                # Choose randomly, but with slight bias to smaller
                # Weight by 1/x? Or just uniform. Uniform is fine for exploration.
                chosen = random.choice(candidates)
                new_row.append(chosen)
                # Add new diffs to used_diffs temporarily? 
                # No, we check against used_diffs which currently lacks old_row's diffs.
                # We need to ensure new_row's diffs don't conflict with *other* rows' diffs.
                # used_diffs contains diffs from all rows except idx.
                # So checking `new_diffs & used_diffs` is correct for conflicts with others.
                # But we also need to ensure new_row's internal diffs are distinct?
                # The loop `for x ...` ensures x > prev, so diffs are positive.
                # Are they distinct?
                # If new_row is [0, 2, 4], diffs are 2, 4, 2. Conflict!
                # My candidate generation `new_diffs = {x - a for a in new_row}` creates a set.
                # If `len(new_diffs) != len(new_row)`, there is an internal conflict.
                # We must check this.
                
                # Let's re-verify internal distinctness in candidate generation
                # Actually, if we build the row sequentially, we just need to ensure
                # the new element x doesn't create a duplicate diff with any previous element
                # that is already in the row's diff set?
                # No, the row's diff set is not maintained in `used_diffs` yet.
                # We need to check if `x - a` is already in the set of diffs generated by `new_row` so far.
                
                # Let's maintain a local set for new_row diffs.
                # This requires refactoring the inner loop.
                
            # Refactor inner loop for correctness
            new_row = [0]
            new_row_diffs = set()
            valid = True
            for _ in range(k):
                candidates = []
                x = new_row[-1] + 1
                while x < limit and len(candidates) < 10:
                    # Check internal conflicts
                    local_diffs = {x - a for a in new_row}
                    if len(local_diffs) != len(new_row):
                        x += 1
                        continue
                    # Check global conflicts
                    if local_diffs & used_diffs:
                        x += 1
                        continue
                    candidates.append(x)
                    x += 1
                
                if not candidates:
                    valid = False
                    break
                
                chosen = random.choice(candidates)
                new_row.append(chosen)
                # Update local diffs
                new_row_diffs.update({chosen - a for a in new_row[:-1]})
                
            if not valid:
                # Rollback: restore old diffs
                used_diffs.update(old_diffs)
                continue
            
            # Calculate new scope
            new_scope = max(max(r) for r in rows if r is not old_row) # This is inefficient
            # Better:
            # current_scope is known.
            # new_scope = max(current_scope, max(new_row)) if old_row was the max, else current_scope?
            # Actually, if old_row was the max, new_scope might decrease.
            # Let's just compute max(new_row) and compare.
            max_new_row = max(new_row)
            # The global scope is max of all rows.
            # We don't have a fast way to get max of other rows without storing them.
            # Let's just compute get_scope(rows) after updating? 
            # No, that's O(N*K).
            
            # Let's maintain a list of maxes?
            # For now, simple check:
            # If max_new_row < current_scope, scope might decrease or stay.
            # If max_new_row > current_scope, scope increases.
            
            # For SA, we need the actual scope.
            # Let's compute it. N=7, K=5. Very fast.
            rows[idx] = new_row
            new_scope_val = get_scope(rows)
            
            delta = new_scope_val - current_scope
            
            if delta < 0 or (T > T_min and random.random() < math.exp(-delta / T)):
                # Accept
                # used_diffs is already updated (old removed, new not added yet? 
                # Wait, in the valid block, we did NOT add new_row_diffs to used_diffs.
                # We only checked against it.
                # So we must add them now.
                used_diffs.update(new_row_diffs)
                
                current_scope = new_scope_val
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = rows[:]
            else:
                # Reject: rollback
                rows[idx] = old_row
                used_diffs.update(old_diffs)
                
            T *= cooling
            
            if best_scope <= 105:
                break
                
        return best_rows

    return solve()
