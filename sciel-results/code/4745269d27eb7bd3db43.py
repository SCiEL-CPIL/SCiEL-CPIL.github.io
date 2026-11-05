"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    deadline = time.time() + 140.0  # safety margin before 150s kill
    start_time = time.time()

    # Parameters
    n = n_rows
    m = k + 1  # number of entries per row
    total_diffs = n * (m * (m - 1) // 2)

    # Known best for (7,5) is 112. We'll try to match or beat it.
    # Simple counting bound: total_diffs <= M*(M+1)/2 => M >= ceil((sqrt(1+8*total_diffs)-1)/2)
    lower_bound = int(math.ceil((math.sqrt(1 + 8 * total_diffs) - 1) / 2))

    # We'll use a randomized local search / hill climbing approach
    # Start with greedy construction, then try to improve

    def greedy_construct(n_rows, k, rng=None):
        """Greedy: pick smallest available mark for each position."""
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            for _ in range(k):
                x = row[-1] + 1
                while True:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        break
                    x += 1
                row.append(x)
                used |= new_diffs
            rows.append(row)
        return rows, max(max(r) for r in rows)

    def validate(rows, n_rows, k):
        """Check if all differences are distinct across all rows."""
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def row_diffs(row):
        """Get all positive differences in a row."""
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def all_diffs(rows):
        """Get all differences across all rows."""
        all_d = set()
        for row in rows:
            all_d |= row_diffs(row)
        return all_d

    def try_improve(rows, n_rows, k, max_time, rng):
        """Try to reduce the maximum entry by local search."""
        best_rows = [list(r) for r in rows]
        best_max = max(max(r) for r in best_rows)
        used = all_diffs(best_rows)
        
        # Try to reduce the largest element
        # Find the row with the largest element
        for _ in range(1000):
            if time.time() > max_time:
                break
            if best_max <= lower_bound:
                break
                
            # Find which row has the max element
            max_row_idx = max(range(n_rows), key=lambda i: best_rows[i][-1])
            row = best_rows[max_row_idx]
            max_val = row[-1]
            
            # Try to reduce the last element
            for new_last in range(max_val - 1, 0, -1):
                if new_last <= row[-2]:
                    break
                # Check if replacing last element works
                new_row = row[:-1] + [new_last]
                # Check increasing
                if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                    continue
                # Check differences
                new_row_d = row_diffs(new_row)
                # Remove old row's diffs from used
                old_row_d = row_diffs(row)
                # Check if new diffs conflict with other rows
                other_diffs = set()
                for i in range(n_rows):
                    if i != max_row_idx:
                        other_diffs |= row_diffs(best_rows[i])
                if not (new_row_d & other_diffs):
                    # Valid replacement
                    best_rows[max_row_idx] = new_row
                    best_max = max(max(r) for r in best_rows)
                    used = all_diffs(best_rows)
                    break
        
        # Also try reducing elements in the middle
        for _ in range(200):
            if time.time() > max_time:
                break
            if best_max <= lower_bound:
                break
            
            # Pick a random row and a random non-first, non-last position
            row_idx = rng.randint(0, n_rows - 1)
            pos = rng.randint(1, k)  # positions 1..k (0-indexed, so not first)
            row = best_rows[row_idx]
            old_val = row[pos]
            
            # Try smaller values
            for new_val in range(old_val - 1, row[pos-1], -1):
                if pos < k and new_val >= row[pos+1]:
                    continue
                new_row = row[:]
                new_row[pos] = new_val
                # Check increasing
                if any(new_row[i] >= new_row[i+1] for i in range(len(new_row)-1)):
                    continue
                new_row_d = row_diffs(new_row)
                other_diffs = set()
                for i in range(n_rows):
                    if i != row_idx:
                        other_diffs |= row_diffs(best_rows[i])
                if not (new_row_d & other_diffs):
                    best_rows[row_idx] = new_row
                    best_max = max(max(r) for r in best_rows)
                    used = all_diffs(best_rows)
                    break
        
        return best_rows, best_max

    # Initial greedy construction
    rng = random.Random(42)
    rows, cur_max = greedy_construct(n, k, rng)
    
    # Try to improve
    rows, cur_max = try_improve(rows, n, k, deadline - 10.0, rng)
    
    # Try multiple random restarts
    attempts = 0
    while time.time() < deadline - 10.0:
        attempts += 1
        # Randomized greedy: sometimes pick a slightly larger value to allow better packing
        used = set()
        rows = []
        success = True
        for _ in range(n):
            row = [0]
            for pos in range(k):
                # Try values from row[-1]+1 upward
                x = row[-1] + 1
                found = False
                # Sometimes allow a small random jump
                max_x = row[-1] + 1 + rng.randint(0, 3)
                while x <= max_x:
                    new_diffs = {x - a for a in row}
                    if not (new_diffs & used):
                        found = True
                        break
                    x += 1
                if not found:
                    # Fallback: keep going higher
                    x = row[-1] + 1
                    while True:
                        new_diffs = {x - a for a in row}
                        if not (new_diffs & used):
                            found = True
                            break
                        x += 1
                        if x > 200:  # safety
                            success = False
                            break
                if not success:
                    break
                row.append(x)
                used |= new_diffs
            if not success:
                break
            rows.append(row)
        
        if len(rows) == n and validate(rows, n, k):
            new_max = max(max(r) for r in rows)
            if new_max < cur_max:
                cur_max = new_max
                # Try to improve further
                rows, cur_max = try_improve(rows, n, k, deadline - 10.0, rng)
        
        if cur_max <= lower_bound:
            break
    
    return rows

# EVOLVE-BLOCK-END
