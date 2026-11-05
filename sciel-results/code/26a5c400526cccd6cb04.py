"""Program for Difference Triangle Set (4,7) - Simulated Annealing with Composite Energy."""
import random
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # We need to represent the state.
    # State: A 2D array `rows` of shape (n_rows, k+1) containing integers.
    # rows[i][0] is always 0.
    # rows[i][j] > rows[i][j-1] for j > 0.
    
    # Energy Function:
    # E = w_coll * (number_of_collision_pairs) + w_scope * (max_entry)^2
    # Actually, counting collision pairs is O(N^2) on diffs if done naively.
    # Better: Count the number of duplicate differences.
    # If a difference appears c times, it contributes c-1 to the "excess" count.
    # Sum of (count[d] - 1) for all d where count[d] > 1.
    # This is linear in the number of diffs if we use a histogram.
    # Number of diffs = n_rows * k * (k+1) / 2.
    # For n_rows=4, k=7, diffs = 4 * 28 = 112.
    # Max possible diff is scope.
    # We can maintain a histogram of differences.
    
    def get_scope(rows):
        return max(max(row) for row in rows) if rows else 0

    def compute_energy(rows, used_mask=None, diff_counts=None):
        """
        Compute energy. If diff_counts is provided, use it for speed.
        diff_counts: array of counts for each difference value.
        """
        scope = get_scope(rows)
        scope_term = scope * scope
        
        if diff_counts is not None:
            # Number of collisions = sum(count[d] - 1) for d where count[d] > 1
            # This is equivalent to sum(count[d]) - sum(1 for d where count[d] > 0)
            # Which is total_diffs - number_of_distinct_diffs
            total_diffs = n_rows * diffs_per_row
            distinct_diffs = np.sum(diff_counts > 0)
            collisions = total_diffs - distinct_diffs
        else:
            # Fallback (slow)
            seen = set()
            collisions = 0
            for row in rows:
                for i in range(len(row)):
                    for j in range(i + 1, len(row)):
                        d = row[j] - row[i]
                        if d in seen:
                            collisions += 1
                        else:
                            seen.add(d)
        
        # We want to minimize collisions first, then scope.
        # Or a composite.
        # Let's try: E = collisions * 1000 + scope
        # This prioritizes validity.
        # To allow navigation through invalid states, we need to accept worse states.
        # But if we prioritize collisions strongly, we might get stuck in a "valid but large scope" local min.
        # A better composite: E = collisions * (scope + 1) + scope
        # Or just E = collisions * 10000 + scope.
        
        # Let's use a weighted sum that emphasizes collisions but allows scope reduction if collisions don't increase.
        # Actually, standard SA on "invalid" states requires the energy to reflect the "distance" to validity.
        # Let's use: E = collisions * 1000 + scope
        
        return collisions * 1000 + scope

    def init_diff_counts(rows):
        max_val = get_scope(rows)
        counts = np.zeros(max_val + 1, dtype=np.int32)
        for row in rows:
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    counts[d] += 1
        return counts

    def update_diff_counts(rows, r_idx, c_idx, old_val, new_val, counts):
        """
        Update the difference counts array when rows[r_idx][c_idx] changes from old_val to new_val.
        rows[r_idx][0] is never changed. c_idx >= 1.
        """
        row = rows[r_idx]
        
        # Remove old differences involving c_idx
        # Diffs (i, c_idx) for i < c_idx: old_val - row[i]
        # Diffs (c_idx, j) for j > c_idx: row[j] - old_val
        
        for i in range(c_idx):
            d = old_val - row[i]
            if d >= 0 and d < len(counts):
                counts[d] -= 1
            else:
                # Should not happen if we maintain bounds, but safe guard
                pass
                
        for j in range(c_idx + 1, len(row)):
            d = row[j] - old_val
            if d >= 0 and d < len(counts):
                counts[d] -= 1
        
        # Add new differences involving c_idx
        # Diffs (i, c_idx) for i < c_idx: new_val - row[i]
        # Diffs (c_idx, j) for j > c_idx: row[j] - new_val
        
        # Ensure counts array is large enough for new_val
        # new_val < row[c_idx+1] <= current_scope.
        # However, new_val might be smaller, but diffs are positive.
        # Max diff is max(row) - min(row) = max(row).
        # If new_val is the max, diff is new_val.
        # We assume counts is sized to the current max scope.
        # If new_val causes a new max, we might need to resize.
        # To simplify, we'll keep counts sized to the initial max or expand if needed.
        
        for i in range(c_idx):
            d = new_val - row[i]
            if d < len(counts):
                counts[d] += 1
            else:
                # Resize
                new_len = max(len(counts), d + 1)
                counts_new = np.zeros(new_len, dtype=np.int32)
                counts_new[:len(counts)] = counts
                counts_new[d] += 1
                # We can't easily reassign the local variable to the outer scope in Python without a wrapper or return
                # This is a problem for in-place update.
                # Solution: Pre-allocate counts to a reasonably large size, e.g., 2 * initial_scope + 100
                # Or, just don't update counts in-place for resizing, but recompute if scope increases significantly?
                # No, let's just pre-allocate a large buffer.
                # Max scope is bounded by the search space.
                # Let's assume the caller ensures counts is large enough.
                pass
        
        for j in range(c_idx + 1, len(row)):
            d = row[j] - new_val
            if d < len(counts):
                counts[d] += 1
        
        # Note: The resize issue above means we need to be careful.
        # Let's modify the strategy:
        # 1. Start with a greedy solution.
        # 2. Determine max_scope_greedy.
        # 3. Allocate counts array of size max_scope_greedy + 100.
        # 4. In SA, restrict moves such that new_val does not exceed this bound?
        #    Or, if it does, we just don't update counts correctly?
        #    Better: If new_val leads to a diff >= len(counts), we reject the move or expand counts.
        #    Expanding numpy arrays is expensive.
        #    Let's fix the size of counts to be, say, 2000. If a diff exceeds this, it's a huge scope increase, likely bad.
        #    Given the target is ~134, 2000 is safe.

    # Revised Plan for Speed and Safety:
    # 1. Greedy Construct to get initial rows and scope.
    # 2. Set MAX_DIFF_BOUND = max(initial_scope * 2, 500).
    # 3. Allocate counts = np.zeros(MAX_DIFF_BOUND, dtype=np.int32).
    # 4. In SA loop:
    #    - Pick r_idx, c_idx.
    #    - Pick new_val.
    #    - Check if new_val creates diffs > MAX_DIFF_BOUND. If so, reject (or allow if we don't track counts for them, but that breaks energy calc).
    #    - Update counts.
    #    - Calculate energy.
    #    - Accept/Reject.
    
    # Greedy Construction
    def greedy_construct(max_val):
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            current_max = 0
            for _pos in range(k):
                found = False
                next_val = current_max + 1
                while next_val <= max_val:
                    valid = True
                    for prev in row:
                        diff = next_val - prev
                        if diff in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(next_val)
                        current_max = next_val
                        for prev in row[:-1]:
                            used_diffs.add(next_val - prev)
                        found = True
                        break
                    next_val += 1
                if not found:
                    return None
            rows.append(row)
        return rows

    # Initialize
    initial_max = total_diffs + 100
    rows = greedy_construct(initial_max)
    if not rows:
        # Fallback: simple construction
        rows = []
        base = 1
        for _ in range(n_rows):
            row = [0]
            for i in range(1, k+1):
                row.append(base * (i+1))
            rows.append(row)
            base *= (k+2)
    
    best_rows = [r[:] for r in rows]
    best_scope = get_scope(best_rows)
    best_energy = float('inf')
    
    # Determine bound for difference counts
    # We expect scope to decrease, but might temporarily increase.
    # Let's set a generous bound.
    MAX_DIFF_BOUND = max(best_scope * 3, 500)
    
    counts = np.zeros(MAX_DIFF_BOUND, dtype=np.int32)
    
    # Initialize counts
    for row in rows:
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                if d < MAX_DIFF_BOUND:
                    counts[d] += 1
                else:
                    # This shouldn't happen for greedy
                    pass
    
    current_energy = compute_energy(rows, None, counts)
    
    # Simulated Annealing
    T = 1000.0
    cooling = 0.999
    rng = random.Random(42)
    
    # Precompute row lengths for speed
    row_len = k + 1
    
    while time.time() < deadline - 5.0:
        # Pick a random move
        r_idx = rng.randint(0, n_rows - 1)
        c_idx = rng.randint(1, k) # 1 to k
        
        row = rows[r_idx]
        old_val = row[c_idx]
        
        # Determine range for new_val
        min_val = row[c_idx - 1] + 1
        max_val_limit = row[c_idx + 1] - 1 if c_idx < k else MAX_DIFF_BOUND - 1
        
        if min_val > max_val_limit:
            continue
            
        # Bias towards smaller values to reduce scope
        # Uniform random in [min_val, max_val_limit]
        # Or, to encourage scope reduction, pick from [min_val, old_val] with high probability
        if rng.random() < 0.7:
            # Try to reduce
            if min_val < old_val:
                new_val = rng.randint(min_val, old_val)
            else:
                new_val = old_val
        else:
            # Try any valid value
            new_val = rng.randint(min_val, max_val_limit)
            
        if new_val == old_val:
            continue
            
        # Check if new_val creates diffs out of bounds
        # Max diff involving c_idx:
        # If c_idx is last, diff is new_val - row[0] = new_val.
        # If c_idx is first (impossible, c_idx>=1), diff is row[k] - new_val.
        # Generally, diffs are bounded by max(row) - min(row).
        # If new_val is within [row[c_idx-1]+1, row[c_idx+1]-1], the max diff is bounded by current row max.
        # So if current row max < MAX_DIFF_BOUND, we are safe.
        
        # Update counts
        # Remove old
        for i in range(c_idx):
            d = old_val - row[i]
            if d < MAX_DIFF_BOUND:
                counts[d] -= 1
        for j in range(c_idx + 1, row_len):
            d = row[j] - old_val
            if d < MAX_DIFF_BOUND:
                counts[d] -= 1
                
        # Add new
        for i in range(c_idx):
            d = new_val - row[i]
            if d < MAX_DIFF_BOUND:
                counts[d] += 1
        for j in range(c_idx + 1, row_len):
            d = row[j] - new_val
            if d < MAX_DIFF_BOUND:
                counts[d] += 1
                
        # Calculate new energy
        total_diffs = n_rows * diffs_per_row
        distinct_diffs = np.sum(counts > 0)
        collisions = total_diffs - distinct_diffs
        new_scope = get_scope(rows)
        # Note: get_scope is O(n_rows * k). We can optimize by tracking max per row.
        # For now, it's fast enough.
        
        new_energy = collisions * 1000 + new_scope
        
        # Accept/Reject
        if new_energy < current_energy or (T > 0 and rng.random() < math.exp(-(new_energy - current_energy) / T)):
            # Accept
            row[c_idx] = new_val
            current_energy = new_energy
            
            if new_scope < best_scope:
                best_scope = new_scope
                best_rows = [r[:] for r in rows]
                if collisions == 0:
                    # We found a valid solution with smaller scope.
                    # We can stop if we hit the lower bound?
                    # Lower bound is total_diffs (112).
                    # If new_scope <= total_diffs, we can't do better on scope if valid?
                    # Actually, scope can be smaller than total_diffs? No, counting bound says scope >= 112.
                    # So if new_scope == 112 and valid, it's optimal.
                    pass
                    
        else:
            # Reject, revert counts
            for i in range(c_idx):
                d = new_val - row[i]
                if d < MAX_DIFF_BOUND:
                    counts[d] -= 1
            for j in range(c_idx + 1, row_len):
                d = row[j] - new_val
                if d < MAX_DIFF_BOUND:
                    counts[d] -= 1
                    
            for i in range(c_idx):
                d = old_val - row[i]
                if d < MAX_DIFF_BOUND:
                    counts[d] += 1
            for j in range(c_idx + 1, row_len):
                d = row[j] - old_val
                if d < MAX_DIFF_BOUND:
                    counts[d] += 1
            
            # Revert row value (it wasn't changed)
            # row[c_idx] is still old_val
            
        # Cool down
        T *= cooling
        if T < 1.0:
            T = 1.0
            # Optionally restart with higher T if stuck?
            # Let's just continue cooling to 0.5 or so.
            if T <= 0.5:
                # Restart
                T = 100.0
                # Re-initialize rows to best_rows?
                # No, let's just keep the current state and restart T.
                # Or, re-seed from best_rows to avoid getting stuck in a bad basin.
                if rng.random() < 0.5:
                    rows = [r[:] for r in best_rows]
                    # Re-init counts
                    counts = np.zeros(MAX_DIFF_BOUND, dtype=np.int32)
                    for row in rows:
                        for i in range(len(row)):
                            for j in range(i + 1, len(row)):
                                d = row[j] - row[i]
                                if d < MAX_DIFF_BOUND:
                                    counts[d] += 1
                    current_energy = compute_energy(rows, None, counts)

    return best_rows

# EVOLVE-BLOCK-END
