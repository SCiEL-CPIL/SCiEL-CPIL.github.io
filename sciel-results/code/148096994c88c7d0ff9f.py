import time
import random
import sys

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Baseline greedy construction
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not (new_diffs & used):
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows

    baseline_rows = greedy_construct()
    best_rows = baseline_rows
    best_scope = max(max(r) for r in best_rows)

    # Lower bound for scope M
    total_diffs = n_rows * k * (k + 1) // 2
    lb = total_diffs
    
    # Precompute bit masks for differences up to a reasonable max
    # Max scope is likely < 200
    MAX_SCOPE = 200
    bit_of = [0] * (MAX_SCOPE + 1)
    for d in range(1, MAX_SCOPE + 1):
        bit_of[d] = 1 << d

    def solve_scope(M):
        """
        Try to find a valid set of n_rows rows with max entry <= M.
        Returns the list of rows if found, else None.
        """
        if time.time() > deadline:
            return None
            
        rows = []
        
        def dfs_clean(row_idx, current_row, current_row_mask, used_mask):
            if time.time() > deadline:
                return False, None
                
            if row_idx == n_rows:
                return True, rows
                
            if len(current_row) == k + 1:
                new_used_mask = used_mask | current_row_mask
                rows.append(current_row)
                
                rem_rows = n_rows - row_idx - 1
                rem_diffs_needed = rem_rows * (k * (k + 1) // 2)
                used_diffs_count = new_used_mask.bit_count()
                available_diffs = M - used_diffs_count
                
                if available_diffs < rem_diffs_needed:
                    rows.pop()
                    return False, None

                success, res_rows = dfs_clean(row_idx + 1, [0], 0, new_used_mask)
                if success:
                    return True, res_rows
                rows.pop()
                return False, None

            last_val = current_row[-1]
            rem_elements = k + 1 - len(current_row)
            max_x = M - rem_elements + 1
            
            if max_x < last_val + 1:
                return False, None

            candidates = []
            for x in range(last_val + 1, max_x + 1):
                new_diffs_mask = 0
                valid = True
                for a in current_row:
                    d = x - a
                    if d > M:
                        valid = False
                        break
                    bit = bit_of[d]
                    if (used_mask | current_row_mask) & bit:
                        valid = False
                        break
                    new_diffs_mask |= bit
                if valid:
                    candidates.append((x, new_diffs_mask))
            
            if not candidates:
                return False, None

            # Randomization is handled by the caller shuffling or using a random seed
            # Here we just iterate. To allow randomization, we can shuffle candidates
            # but that requires a random generator.
            
            for x, new_diffs_mask in candidates:
                new_row = current_row + [x]
                new_row_mask = current_row_mask | new_diffs_mask
                
                success, res_rows = dfs_clean(row_idx, new_row, new_row_mask, used_mask)
                if success:
                    return True, res_rows

            return False, None

        def attempt_search(randomize, rng):
            if randomize:
                # We need to modify dfs_clean to use rng for shuffling
                # This is tricky with the current structure.
                # Alternative: Generate candidates and shuffle them.
                pass
            
            # For now, let's just run it. The deterministic order might be enough
            # or we can add a simple randomization by picking a random starting offset?
            # No, the structure is fixed.
            
            # Let's implement a version of dfs that shuffles candidates
            def dfs_random(row_idx, current_row, current_row_mask, used_mask):
                if time.time() > deadline:
                    return False, None
                    
                if row_idx == n_rows:
                    return True, rows
                    
                if len(current_row) == k + 1:
                    new_used_mask = used_mask | current_row_mask
                    rows.append(current_row)
                    
                    rem_rows = n_rows - row_idx - 1
                    rem_diffs_needed = rem_rows * (k * (k + 1) // 2)
                    used_diffs_count = new_used_mask.bit_count()
                    available_diffs = M - used_diffs_count
                    
                    if available_diffs < rem_diffs_needed:
                        rows.pop()
                        return False, None

                    success, res_rows = dfs_random(row_idx + 1, [0], 0, new_used_mask)
                    if success:
                        return True, res_rows
                    rows.pop()
                    return False, None

                last_val = current_row[-1]
                rem_elements = k + 1 - len(current_row)
                max_x = M - rem_elements + 1
                
                if max_x < last_val + 1:
                    return False, None

                candidates = []
                for x in range(last_val + 1, max_x + 1):
                    new_diffs_mask = 0
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d > M:
                            valid = False
                            break
                        bit = bit_of[d]
                        if (used_mask | current_row_mask) & bit:
                            valid = False
                            break
                        new_diffs_mask |= bit
                    if valid:
                        candidates.append((x, new_diffs_mask))
                
                if not candidates:
                    return False, None

                rng.shuffle(candidates)
                
                for x, new_diffs_mask in candidates:
                    new_row = current_row + [x]
                    new_row_mask = current_row_mask | new_diffs_mask
                    
                    success, res_rows = dfs_random(row_idx, new_row, new_row_mask, used_mask)
                    if success:
                        return True, res_rows

                return False, None

            if randomize:
                return dfs_random(0, [0], 0, 0)
            else:
                return dfs_clean(0, [0], 0, 0)

        # Try deterministic first
        sol = attempt_search(False, None)
        if sol:
            return sol
            
        # Try randomized restarts
        rng = random.Random(42)
        for _ in range(50):
            if time.time() > deadline:
                break
            sol = attempt_search(True, rng)
            if sol:
                return sol
                
        return None

    # Search for M
    for M in range(lb, best_scope + 1):
        if time.time() > deadline:
            break
        sol = solve_scope(M)
        if sol:
            return sol
            
    return best_rows
