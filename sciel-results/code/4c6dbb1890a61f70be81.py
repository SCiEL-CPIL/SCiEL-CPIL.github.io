import time
import random
import numpy as np

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    # Helper to generate all pairs (i, j) for a row
    def get_pairs():
        pairs = []
        for i in range(row_len):
            for j in range(i+1, row_len):
                pairs.append((i, j))
        return pairs
        
    pairs = get_pairs()
    n_pairs = len(pairs)
    
    def calculate_score(matrix):
        """
        Returns (score, conflicts).
        Score is the sum of squared differences from ideal spacing? 
        Actually, for this problem, the objective is to minimize the max entry.
        However, the hard constraint is that all differences a_j - a_i are distinct.
        
        Let's define the cost function for SA:
        1. Hard constraint: All differences must be unique.
        2. Soft objective: Minimize the maximum value in the matrix.
        
        Since minimizing max value directly is non-smooth, we can use a penalty-based approach.
        Cost = (max_val) + L * (number_of_conflicts)
        But we need to be careful. If we have conflicts, the solution is invalid.
        
        Alternative: 
        Initialize with a valid greedy solution.
        Then, perform moves that try to reduce the max_val while maintaining validity, 
        or allow temporary invalidity with a heavy penalty.
        
        Let's use a penalty method:
        Energy = max_val + lambda * (conflict_count)
        We want to minimize Energy.
        
        However, checking conflict_count is O(n_rows * n_pairs^2) if done naively.
        We can maintain a global set of differences.
        """
        max_val = 0
        diffs = set()
        conflicts = 0
        for r in range(n_rows):
            row = matrix[r]
            if row[-1] > max_val:
                max_val = row[-1]
            for i in range(row_len):
                ri = row[i]
                for j in range(i+1, row_len):
                    d = row[j] - ri
                    if d in diffs:
                        conflicts += 1
                    else:
                        diffs.add(d)
        return max_val + 1000 * conflicts, max_val, conflicts

    def greedy_initial():
        """Generate a valid initial solution using greedy."""
        matrix = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            matrix.append(row)
        return matrix

    # Start with a valid greedy solution
    current_matrix = greedy_initial()
    current_score, current_max, current_conflicts = calculate_score(np.array(current_matrix))
    best_matrix = [r[:] for r in current_matrix]
    best_score = current_score
    best_max = current_max
    
    if current_conflicts > 0:
        # Greedy should be valid, but just in case
        pass

    rng = random.Random(42)
    T = 100.0 # Initial temperature
    cooling = 0.9995
    
    # We will optimize by changing individual entries.
    # Since rows must be increasing and start at 0, we can only change a[i] for i > 0.
    # Also, a[i] must be > a[i-1].
    
    while time.time() < DEADLINE:
        # Choose a random row and a random index > 0 to change
        r_idx = rng.randint(0, n_rows - 1)
        # We can't change index 0 (it's always 0)
        i_idx = rng.randint(1, row_len - 1)
        
        old_val = current_matrix[r_idx][i_idx]
        
        # Propose a new value
        # We want to potentially reduce the max value.
        # So we should bias towards smaller values, but not break increasing order.
        # Range for new_val: (current_matrix[r_idx][i_idx-1], current_matrix[r_idx][i_idx+1] if i_idx < row_len-1 else current_max+10)
        
        lo = current_matrix[r_idx][i_idx-1] + 1
        hi = current_matrix[r_idx][i_idx+1] - 1 if i_idx < row_len - 1 else current_max + 20
        
        if lo > hi:
            # Can't move? Maybe we can expand if we're at the end?
            # If i_idx is the last one, hi is large.
            # If i_idx is in the middle, we are constrained by neighbors.
            # If lo > hi, we can't make a valid move that keeps order?
            # Actually, if we change a[i], we must ensure a[i-1] < new_val < a[i+1].
            # If lo > hi, it means a[i-1] + 1 >= a[i+1] - 1, so the gap is too small to fit an integer?
            # No, if gap is 1, lo = a[i-1]+1 = a[i], hi = a[i+1]-1 = a[i-1]+1-1 = a[i-1]. lo > hi.
            # In that case, we can only keep the value or... wait.
            # If the gap is small, we might be stuck.
            # Let's allow expanding the row if necessary? No, we want to minimize max.
            # If we can't reduce, we might need to increase a neighbor to create space?
            # This suggests a more complex move: shift a suffix of the row.
            
            # For now, skip if no valid integer exists in the gap.
            continue
            
        # Bias towards lower values to reduce scope
        # But we must avoid conflicts.
        # Let's try a random value in [lo, hi]
        # To encourage reduction, we can sample from a distribution skewed towards lo.
        # Or just pick a random one, and rely on SA to accept worse (higher max) if it resolves conflicts?
        # But we start valid. So we should only accept if it stays valid or improves max.
        # If we allow invalid states, we need to be very careful.
        
        # Let's try to find a value that minimizes conflicts locally.
        # This is expensive.
        
        # Simpler: Pick a random candidate in [lo, hi]
        # If we pick a value that creates a conflict, the penalty increases.
        # If we pick a value that reduces max_val (only possible if i_idx is the last element and we reduce it),
        # the score decreases.
        
        # Actually, reducing an intermediate element doesn't reduce the max unless it IS the max.
        # So, we should focus on the last element of the row, or elements that contribute to the max.
        
        # Let's refine the move:
        # 1. If we pick the last element, we can reduce it (if it's the global max).
        # 2. If we pick an intermediate element, we can change it, but max doesn't change (unless we increase it beyond current max, which is bad).
        
        # So, a better move strategy:
        # - With prob 0.5, pick the last element of a random row. Try to reduce it.
        # - With prob 0.5, pick a random intermediate element. Try to change it (might help resolve conflicts if we were in a bad state, but we start good).
        
        # Since we start valid, we want to maintain validity.
        # Let's check if the new value creates conflicts.
        
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
            
        # Calculate new diffs for this row
        # Old diffs involving i_idx:
        # (i_idx, j) for j > i_idx: a[j] - a[i]
        # (i, i_idx) for i < i_idx: a[i_idx] - a[i]
        
        # This is getting complicated to update incrementally without a full recalculation.
        # Given n_rows=7, k=6 (row_len=7), n_pairs=21.
        # Total diffs = 147.
        # Recalculating score is O(147) which is fast.
        
        # Let's just do full recalculation for simplicity and correctness first.
        # If it's too slow, we optimize.
        
        # Save old state
        old_row = current_matrix[r_idx][:]
        current_matrix[r_idx][i_idx] = new_val
        
        # Check validity of the row itself (increasing order)
        # We already ensured lo > prev and hi < next, so order is preserved.
        
        new_score, new_max, new_conflicts = calculate_score(np.array(current_matrix))
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            current_max = new_max
            current_conflicts = new_conflicts
            
            if current_conflicts == 0 and current_max < best_max:
                best_max = current_max
                best_score = current_score
                best_matrix = [r[:] for r in current_matrix]
        else:
            # Revert
            current_matrix[r_idx] = old_row
        
        T *= cooling
        if T < 0.1:
            T = 0.1 # Floor temperature to allow some exploration

    # Return the best valid solution found
    # Ensure best_matrix is valid
    # If best_conflicts was not tracked, we assume best_max implies valid if we only updated when conflicts==0
    # In the code above, we only update best if current_conflicts == 0.
    # So best_matrix is valid.
    
    return best_matrix

# Wait, the return type in the prompt says:
# "interface: construct(n_rows, k) -> list of n_rows rows"
# "The evaluator passes this task's n_rows and k."
# "Each row is a list of k + 1 integers..."
# "Score: the scope... minimized"
# "Keep the function names and signatures of the seed program."
# The seed program returns `rows` (a list of lists).
# My code returns `best_matrix` which is a list of lists.
# This matches.

# One detail: `calculate_score` uses `np.array`. 
# `current_matrix` is a list of lists.
# `np.array(current_matrix)` creates a 2D array.
# This is fine.

# Let's double check the `lo` and `hi` logic.
# If `i_idx` is 0, we skip (rng.randint(1, ...)).
# If `i_idx` is `row_len - 1` (last), `hi` is `current_max + 20`.
# This allows increasing the last element if necessary to resolve a conflict?
# But we start valid. Increasing the last element increases the max, which is bad.
# However, if we are stuck in a local minimum where we can't reduce the max without creating conflicts,
# and we are using SA, we might need to increase it to escape?
# But the penalty for conflicts is 1000.
# If we increase max by 1, score increases by 1.
# If we resolve a conflict, score decreases by 1000.
# So SA will prefer resolving conflicts even if it increases max slightly.
# This is good for escaping local minima.

# However, if we increase the max, we might make it harder to reduce it later.
# But the "scope" is the max entry.
# We want the smallest possible max entry.

# Is there a risk that `best_matrix` is not the one with the smallest max?
# We update `best_matrix` only if `current_conflicts == 0` and `current_max < best_max`.
# This ensures `best_matrix` is always valid and has the best max seen so far.

# What if we never find a valid solution better than the initial greedy?
# Then we return the initial greedy.
# This is safe.

# Time complexity:
# Each iteration:
# - O(1) to pick indices.
# - O(1) to pick value.
# - O(n_rows * n_pairs) to calculate score.
# n_rows=7, n_pairs=21. 7*21 = 147 operations.
# 115 seconds available.
# Python can do ~10^7 simple ops per second? Maybe less with overhead.
# 147 ops * 10^6 iterations = 1.4 * 10^8 ops.
# This might be tight in 115s.
# Let's optimize `calculate_score` to not use numpy array conversion if possible, or make it faster.

def construct(n_rows, k):
    t0 = time.time()
    DEADLINE = t0 + 115.0
    row_len = k + 1
    
    def get_pairs():
        pairs = []
        for i in range(row_len):
            for j in range(i+1, row_len):
                pairs.append((i, j))
        return pairs
        
    pairs = get_pairs()
    
    def calculate_score(matrix):
        max_val = 0
        diffs = set()
        conflicts = 0
        for r in range(n_rows):
            row = matrix[r]
            if row[-1] > max_val:
                max_val = row[-1]
            for i in range(row_len):
                ri = row[i]
                for j in range(i+1, row_len):
                    d = row[j] - ri
                    if d in diffs:
                        conflicts += 1
                    else:
                        diffs.add(d)
        return max_val + 1000 * conflicts, max_val, conflicts

    def greedy_initial():
        matrix = []
        used_diffs = set()
        for _ in range(n_rows):
            row = [0]
            last = 0
            while len(row) < row_len:
                last += 1
                new_diffs = {last - a for a in row}
                if not (new_diffs & used_diffs):
                    row.append(last)
                    used_diffs |= new_diffs
            matrix.append(row)
        return matrix

    current_matrix = greedy_initial()
    current_score, current_max, current_conflicts = calculate_score(current_matrix)
    best_matrix = [r[:] for r in current_matrix]
    best_score = current_score
    best_max = current_max
    
    rng = random.Random(42)
    T = 100.0
    cooling = 0.9995
    
    while time.time() < DEADLINE:
        r_idx = rng.randint(0, n_rows - 1)
        i_idx = rng.randint(1, row_len - 1)
        
        old_val = current_matrix[r_idx][i_idx]
        
        lo = current_matrix[r_idx][i_idx-1] + 1
        hi = current_matrix[r_idx][i_idx+1] - 1 if i_idx < row_len - 1 else current_max + 20
        
        if lo > hi:
            continue
            
        new_val = rng.randint(lo, hi)
        if new_val == old_val:
            continue
            
        old_row = current_matrix[r_idx][:]
        current_matrix[r_idx][i_idx] = new_val
        
        new_score, new_max, new_conflicts = calculate_score(current_matrix)
        
        delta = new_score - current_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            current_score = new_score
            current_max = new_max
            current_conflicts = new_conflicts
            
            if current_conflicts == 0 and current_max < best_max:
                best_max = current_max
                best_score = current_score
                best_matrix = [r[:] for r in current_matrix]
        else:
            current_matrix[r_idx] = old_row
        
        T *= cooling
        if T < 0.1:
            T = 0.1

    return best_matrix
