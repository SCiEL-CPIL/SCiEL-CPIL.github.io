import time
import random
import sys
from itertools import combinations

# EVOLVE-BLOCK-START
def construct(n_rows, k):
    # Deadline guard
    start_time = time.time()
    timeout = 112  # Safe margin for 120s limit
    
    k_plus_1 = k + 1
    n_diffs_per_row = k_plus_1 * k // 2  # Number of differences per row
    
    def elapsed():
        return time.time() - start_time
    
    def row_diffs_mask(row):
        """Compute bitmask of all differences in a row."""
        mask = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                d = row[j] - row[i]
                mask |= (1 << d)
        return mask
    
    def get_max(rows):
        return max(max(r) for r in rows)
    
    def is_valid(rows):
        """Check if all differences across all rows are distinct."""
        used = 0
        for row in rows:
            mask = row_diffs_mask(row)
            if used & mask:
                return False
            used |= mask
        return True
    
    # 1. Fast greedy construction with randomization
    def greedy_construct(seed=None):
        if seed is not None:
            random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                x = row[-1] + 1
                while True:
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        row.append(x)
                        used |= new_diffs
                        break
                    x += 1
            rows.append(row)
        return rows
    
    # 2. Randomized greedy with lookahead
    def randomized_greedy(seed):
        random.seed(seed)
        used = 0
        rows = []
        for _ in range(n_rows):
            row = [0]
            while len(row) < k_plus_1:
                # Try several candidates, pick one that minimizes future conflicts
                best_x = None
                best_score = float('inf')
                start_x = row[-1] + 1
                # Look ahead window
                window = min(20, 2 * (k_plus_1 - len(row)))
                for x in range(start_x, start_x + window):
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (used >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        # Score: prefer smaller x, but also consider how many
                        # future differences it blocks
                        score = x
                        best_x = x
                        if best_score > score:
                            best_score = score
                        # Early exit if we found a very good candidate
                        if x == start_x:
                            break
                if best_x is None:
                    # Fallback: find any valid x
                    x = row[-1] + 1
                    while True:
                        valid = True
                        new_diffs = 0
                        for a in row:
                            d = x - a
                            if (used >> d) & 1:
                                valid = False
                                break
                            if (new_diffs >> d) & 1:
                                valid = False
                                break
                            new_diffs |= (1 << d)
                        if valid:
                            break
                        x += 1
                    best_x = x
                    new_diffs = 0
                    for a in row:
                        d = best_x - a
                        new_diffs |= (1 << d)
                else:
                    new_diffs = 0
                    for a in row:
                        d = best_x - a
                        new_diffs |= (1 << d)
                row.append(best_x)
                used |= new_diffs
            rows.append(row)
        return rows
    
    # 3. Local search: try to reduce individual elements
    def local_search_reduce(rows, max_iterations=50000):
        """Try to reduce the maximum element by local search."""
        rows = [list(r) for r in rows]
        used = 0
        for row in rows:
            used |= row_diffs_mask(row)
        
        current_max = get_max(rows)
        improved = True
        iter_count = 0
        
        while improved and elapsed() < timeout and iter_count < max_iterations:
            improved = False
            iter_count += 1
            
            # Find the position with the maximum value
            max_val = -1
            max_r = -1
            max_i = -1
            for r in range(n_rows):
                for i in range(len(rows[r])):
                    if rows[r][i] > max_val:
                        max_val = rows[r][i]
                        max_r = r
                        max_i = i
            
            if max_val <= 0:
                break
            
            row = rows[max_r]
            current_val = row[max_i]
            
            # Determine valid range
            min_val = row[max_i - 1] + 1 if max_i > 0 else 1
            max_allowed = current_val - 1
            if max_i < len(row) - 1:
                max_allowed = min(max_allowed, row[max_i + 1] - 1)
            
            if max_allowed < min_val:
                continue
            
            # Calculate old differences
            old_diffs = 0
            for a in row:
                if a != current_val:
                    d = abs(a - current_val)
                    old_diffs |= (1 << d)
            
            used_without_old = used & ~old_diffs
            
            # Try to find a smaller valid value
            for v in range(min_val, max_allowed + 1):
                valid = True
                new_diffs = 0
                for a in row:
                    if a != current_val:
                        d = abs(a - v)
                        if (used_without_old >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                
                if valid:
                    row[max_i] = v
                    used = used_without_old | new_diffs
                    improved = True
                    break
        
        return rows, get_max(rows)
    
    # 4. Row reconstruction: rebuild a single row given the used mask of others
    def rebuild_row(row_idx, used_others, target_max=None):
        """Rebuild row row_idx to minimize its maximum, given others' differences."""
        if target_max is None:
            target_max = 200  # Generous upper bound
        
        best_row = None
        best_max = float('inf')
        
        # Try multiple random orderings
        for attempt in range(30):
            if elapsed() > timeout:
                break
            
            row = [0]
            success = True
            for pos in range(1, k_plus_1):
                found = False
                for x in range(row[-1] + 1, target_max + 1):
                    valid = True
                    new_diffs = 0
                    for a in row:
                        d = x - a
                        if (used_others >> d) & 1:
                            valid = False
                            break
                        if (new_diffs >> d) & 1:
                            valid = False
                            break
                        new_diffs |= (1 << d)
                    if valid:
                        row.append(x)
                        found = True
                        break
                if not found:
                    success = False
                    break
            
            if success:
                row_max = max(row)
                if row_max < best_max:
                    best_max = row_max
                    best_row = list(row)
        
        return best_row
    
    # 5. Simulated annealing style improvement
    def sa_improve(rows, iterations=10000):
        """Try random perturbations and keep improvements."""
        rows = [list(r) for r in rows]
        used = 0
        for row in rows:
            used |= row_diffs_mask(row)
        
        current_max = get_max(rows)
        best_rows = [list(r) for r in rows]
        best_max = current_max
        
        for it in range(iterations):
            if elapsed() > timeout:
                break
            
            # Pick a random row and random position to perturb
            r = random.randint(0, n_rows - 1)
            i = random.randint(1, len(rows[r]) - 1)  # Don't touch 0
            
            old_val = rows[r][i]
            row = rows[r]
            
            # Calculate old diffs
            old_diffs = 0
            for a in row:
                if a != old_val:
                    d = abs(a - old_val)
                    old_diffs |= (1 << d)
            
            used_without = used & ~old_diffs
            
            # Try a random new value
            delta = random.randint(-5, 5)
            new_val = old_val + delta
            
            # Check bounds
            min_val = row[i-1] + 1 if i > 0 else 1
            max_val_allowed = row[i+1] - 1 if i < len(row) - 1 else 1000
            
            if new_val < min_val or new_val > max_val_allowed:
                continue
            
            # Check validity
            valid = True
            new_diffs = 0
            for a in row:
                if a != old_val:
                    d = abs(a - new_val)
                    if (used_without >> d) & 1:
                        valid = False
                        break
                    if (new_diffs >> d) & 1:
                        valid = False
                        break
                    new_diffs |= (1 << d)
            
            if not valid:
                continue
            
            # Accept if it improves or with some probability
            new_max = max(max(r) for r in rows)
            # Update row
            row[i] = new_val
            used = used_without | new_diffs
            
            new_max_val = get_max(rows)
            if new_max_val < best_max:
                best_max = new_max_val
                best_rows = [list(r) for r in rows]
            
            # With some probability, revert (simulated annealing)
            if random.random() < 0.3:
                row[i] = old_val
                used = used_without | old_diffs
        
        return best_rows, best_max
    
    # Main algorithm
    # Phase 1: Get initial solutions from multiple seeds
    best_rows = None
    best_max = float('inf')
    
    # First, deterministic greedy
    rows = greedy_construct()
    best_rows = rows
    best_max = get_max(rows)
    
    # Try randomized greedy with multiple seeds
    for seed in range(20):
        if elapsed() > timeout - 30:
            break
        rows = randomized_greedy(seed)
        m = get_max(rows)
        if m < best_max:
            best_max = m
            best_rows = rows
    
    # Phase 2: Local search to reduce max
    polished_rows, polished_max = local_search_reduce(best_rows)
    if polished_max < best_max:
        best_max = polished_max
        best_rows = polished_rows
    
    # Phase 3: Row reconstruction
    for attempt in range(10):
        if elapsed() > timeout - 20:
            break
        
        # Compute used mask for all rows except one
        for r in range(n_rows):
            if elapsed() > timeout - 20:
                break
            
            used_others = 0
            for other_r in range(n_rows):
                if other_r != r:
                    used_others |= row_diffs_mask(best_rows[other_r])
            
            new_row = rebuild_row(r, used_others, target_max=best_max - 1)
            if new_row is not None:
                candidate_rows = [list(best_rows[i]) for i in range(n_rows)]
                candidate_rows[r] = new_row
                if is_valid(candidate_rows):
                    m = get_max(candidate_rows)
                    if m < best_max:
                        best_max = m
                        best_rows = candidate_rows
    
    # Phase 4: More local search
    polished_rows, polished_max = local_search_reduce(best_rows, max_iterations=30000)
    if polished_max < best_max:
        best_max = polished_max
        best_rows = polished_rows
    
    # Phase 5: SA improvement
    if elapsed() < timeout - 15:
        sa_rows, sa_max = sa_improve(best_rows, iterations=20000)
        if sa_max < best_max:
            best_max = sa_max
            best_rows = sa_rows
    
    # Final local search
    if elapsed() < timeout - 5:
        polished_rows, polished_max = local_search_reduce(best_rows, max_iterations=20000)
        if polished_max < best_max:
            best_max = polished_max
            best_rows = polished_rows
    
    return best_rows
# EVOLVE-BLOCK-END
