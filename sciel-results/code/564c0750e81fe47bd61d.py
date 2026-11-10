import time
import random
from collections import defaultdict

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry (the scope).
    
    Uses a search approach with multiple strategies:
    1. Greedy construction (baseline)
    2. Randomized greedy with restarts
    3. Local search / hill climbing on the maximum entry
    """
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_rows = None
    best_score = float('inf')
    
    def score(rows):
        if rows is None:
            return float('inf')
        return max(max(row) for row in rows)
    
    def valid(rows):
        if not rows:
            return False
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def greedy_random(seed_val, n_rows, k):
        """Randomized greedy: at each step, try candidates in randomized order."""
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            attempts = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                elif x > 500:
                    # Too large, restart this row with slight perturbation
                    break
                attempts += 1
                if attempts > 1000:
                    return None
            if len(row) < k + 1:
                return None
            rows.append(row)
        return rows
    
    def greedy_perturbed(seed_val, n_rows, k, perturb_prob=0.1):
        """Greedy with occasional random jumps to escape bad local choices."""
        rng = random.Random(seed_val)
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                # Decide how far to jump
                if rng.random() < perturb_prob:
                    jump = rng.randint(1, 5)
                else:
                    jump = 1
                x += jump
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                if x > 300 and len(row) < k + 1:
                    return None
            if len(row) < k + 1:
                return None
            rows.append(row)
        return rows
    
    def local_search(rows, n_rows, k, iterations=100, rng=None):
        """Try to improve a valid solution by reducing the max entry.
        This is a heuristic local search that tries to shift elements down."""
        if rng is None:
            rng = random.Random()
        
        rows = [row[:] for row in rows]
        current_max = max(max(row) for row in rows)
        improved = True
        
        for _ in range(iterations):
            found_improvement = False
            for r_idx in range(n_rows):
                for pos in range(k, 0, -1):  # from last element to first
                    old_val = rows[r_idx][pos]
                    if pos == 0:
                        continue  # can't change 0
                    # Try to reduce this element
                    for new_val in range(old_val - 1, rows[r_idx][pos - 1], -1):
                        # Check if changing rows[r_idx][pos] to new_val keeps validity
                        # We need to check all differences involving this position
                        # This is expensive, so we do a quick check
                        new_row = rows[r_idx][:]
                        new_row[pos] = new_val
                        # Quick validity check for this row
                        row_diffs = set()
                        for i in range(len(new_row)):
                            for j in range(i + 1, len(new_row)):
                                d = new_row[j] - new_row[i]
                                if d in row_diffs:
                                    break
                                row_diffs.add(d)
                        else:
                            # Check against other rows' differences
                            # Collect all diffs from other rows
                            other_used = set()
                            for other_r in range(n_rows):
                                if other_r == r_idx:
                                    continue
                                orow = rows[other_r]
                                for i in range(len(orow)):
                                    for j in range(i + 1, len(orow)):
                                        other_used.add(orow[j] - orow[i])
                            if row_diffs & other_used:
                                continue
                            # Valid! Apply the change
                            rows[r_idx][pos] = new_val
                            new_max = max(max(row) for row in rows)
                            if new_max < current_max:
                                current_max = new_max
                                found_improvement = True
                                break
            if not found_improvement:
                break
        
        return rows, current_max
    
    def greedy_backtrack(seed_val, n_rows, k, max_depth=3):
        """Greedy with backtracking on the last few elements of each row."""
        rng = random.Random(seed_val)
        used = set()
        rows = []
        
        def try_row(row_prefix, current_used, target_len):
            """Recursively try to complete a row from prefix, with backtracking."""
            if len(row_prefix) == target_len:
                return True
            
            x = row_prefix[-1] + 1
            # Try candidates in slightly randomized order
            candidates = list(range(x, x + 50))
            rng.shuffle(candidates)
            
            for cand in candidates:
                new_diffs = {cand - a for a in row_prefix}
                if not (new_diffs & current_used):
                    if try_row(row_prefix + [cand], current_used | new_diffs, target_len):
                        return True
            return False
        
        for _ in range(n_rows):
            row = [0]
            # First, try greedy for most of the row
            x = 0
            success = True
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                else:
                    # Greedy failed, try backtracking from here
                    # Reset and use backtracking for the remaining part
                    # This is expensive, so only do it if we haven't gone too far
                    if len(row) < k + 1:
                        # Try to complete with backtracking
                        row_prefix = row[:]
                        # Remove the last element that failed
                        if row_prefix:
                            row_prefix = row_prefix[:-1]
                        # Restore used set (simplified - just retry)
                        # For simplicity, just fail this row
                        success = False
                        break
                    break
            
            if not success:
                # Try a fresh start with backtracking
                row = [0]
                if try_row(row, used, k + 1):
                    used |= {row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))}
                else:
                    return None
            else:
                # Verify the row was completed
                if len(row) < k + 1:
                    return None
            
            rows.append(row)
        
        return rows
    
    # Strategy 1: Pure greedy (deterministic)
    rows = greedy_random(42, n_rows, k)
    if rows is not None and valid(rows):
        s = score(rows)
        if s < best_score:
            best_score = s
            best_rows = [r[:] for r in rows]
    
    # Strategy 2: Multiple randomized greedy attempts
    n_attempts = 50
    attempt_num = 0
    while time.time() < deadline and attempt_num < n_attempts:
        seed = attempt_num * 7919 + 13
        rows = greedy_perturbed(seed, n_rows, k, perturb_prob=0.05 + 0.1 * (attempt_num % 10) / 10.0)
        if rows is not None and valid(rows):
            s = score(rows)
            if s < best_score:
                best_score = s
                best_rows = [r[:] for r in rows]
        attempt_num += 1
    
    # Strategy 3: Local search on best solution
    if best_rows is not None and time.time() < deadline:
        rng = random.Random(12345)
        improved_rows, improved_score = local_search(best_rows, n_rows, k, iterations=50, rng=rng)
        if improved_score < best_score:
            best_score = improved_score
            best_rows = [r[:] for r in improved_rows]
    
    # Strategy 4: More randomized greedy with different perturbation levels
    attempt_num = 0
    while time.time() < deadline and attempt_num < 100:
        seed = 100000 + attempt_num * 31
        perturb = 0.01 + 0.2 * (attempt_num % 20) / 20.0
        rows = greedy_perturbed(seed, n_rows, k, perturb_prob=perturb)
        if rows is not None and valid(rows):
            s = score(rows)
            if s < best_score:
                best_score = s
                best_rows = [r[:] for r in rows]
        attempt_num += 1
    
    # Strategy 5: Try a different greedy order - pick the smallest available x
    # but with a "look ahead" preference
    while time.time() < deadline:
        seed = 200000 + attempt_num * 17
        rng = random.Random(seed)
        used = set()
        rows = []
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
                if x > 200:
                    success = False
                    break
            if not success:
                break
            rows.append(row)
        if success and valid(rows):
            s = score(rows)
            if s < best_score:
                best_score = s
                best_rows = [r[:] for r in rows]
        attempt_num += 1
        if attempt_num > 500:
            break
    
    # Strategy 6: Greedy with backtracking on last few elements
    if time.time() < deadline:
        for seed in range(20):
            if time.time() >= deadline:
                break
            rows = greedy_backtrack(300000 + seed * 41, n_rows, k, max_depth=3)
            if rows is not None and valid(rows):
                s = score(rows)
                if s < best_score:
                    best_score = s
                    best_rows = [r[:] for r in rows]
    
    if best_rows is not None:
        return best_rows
    
    # Fallback: return a simple valid construction even if suboptimal
    # Use the original greedy
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                row.append(x)
                used |= new
        rows.append(row)
    return rows
# EVOLVE-BLOCK-END
