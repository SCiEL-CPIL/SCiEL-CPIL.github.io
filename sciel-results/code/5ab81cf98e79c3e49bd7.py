# EVOLVE-BLOCK-START
import sys
import time
import random

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Improved Greedy Baseline
    def greedy_construct(n_rows, k, randomize=False, seed=0):
        used = set()
        rows = []
        rng = random.Random(seed) if randomize else None
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                    if randomize and rng and rng.random() < 0.1:
                        # Small perturbation: try a slightly larger x next time?
                        # Actually, greedy is deterministic. 
                        # To randomize, we can skip a few valid candidates.
                        pass
            rows.append(row)
        return rows

    # Get a good upper bound using randomized greedy restarts
    best_greedy_rows = None
    best_greedy_max = float('inf')
    for i in range(10):
        # Standard greedy
        r = greedy_construct(n_rows, k)
        m = max(max(row) for row in r)
        if m < best_greedy_max:
            best_greedy_max = m
            best_greedy_rows = r
            
        # Randomized greedy: pick from first few valid candidates
        used = set()
        rows = []
        rng = random.Random(i)
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                valid_candidates = []
                # Look ahead a bit
                for _look in range(5):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= 5:
                            break
                if not valid_candidates:
                    success = False
                    break
                # Pick randomly from valid candidates
                x, nd = rng.choice(valid_candidates)
                row.append(x)
                used |= nd
            rows.append(row)
        
        if success:
            m = max(max(row) for row in rows)
            if m < best_greedy_max:
                best_greedy_max = m
                best_greedy_rows = rows

    if best_greedy_rows is None:
        best_greedy_rows = greedy_construct(n_rows, k)
        best_greedy_max = max(max(row) for row in best_greedy_rows)

    start_time = time.time()
    time_limit = 115.0
    
    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        def build_row(current_row, used_bits, row_idx, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            # Optimization: If the range is large, we might want to sample?
            # For M~134, range is small. Iterate all.
            for x in range(min_x, max_x + 1):
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if not conflict:
                    candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Shuffle candidates
            rng.shuffle(candidates)
            
            for x, bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                if build_row(current_row, used_bits | bits, row_idx, deadline):
                    return True
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            current_row = [0]
            if build_row(current_row, used_bits, row_idx, deadline):
                rows.append(current_row)
                # Update used_bits
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    best_rows = best_greedy_rows
    best_max = best_greedy_max
    
    # Search for better M
    # We iterate M from lb to best_greedy_max
    # Allocate time based on M
    for M in range(lb, best_greedy_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Time budget allocation
        # Give more time to smaller M
        dist = M - lb
        if dist == 0:
            budget = min(10.0, remaining_time)
        elif dist <= 5:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Number of restarts
        # Each restart gets a slice of budget
        num_restarts = max(1, int(budget / 0.5))
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            # Give each restart a small fixed time to ensure diversity
            restart_budget = min(0.5, budget / (num_restarts - seed))
            
            res = solve_csp(n_rows, k, M, restart_budget, seed=seed)
            if res is not None:
                best_rows = res
                best_max = M
                found = True
                break
        
        if found:
            break
            
    # Validate
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
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
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        best_rows = greedy_construct(n_rows, k)
    
    return best_rows
# EVOLVE-BLOCK-END
