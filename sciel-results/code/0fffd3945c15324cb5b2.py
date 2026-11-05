import time
import random

def construct(n_rows, k):
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    # Counting bound
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = 0
    while lb * (lb + 1) // 2 < total_diffs_needed:
        lb += 1
    
    def greedy_construct(n_rows, k):
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
            rows.append(row)
        return rows
    
    greedy_rows = greedy_construct(n_rows, k)
    greedy_max = max(max(r) for r in greedy_rows)
    
    start_time = time.time()
    time_limit = 115.0
    
    def solve_csp(max_entry, deadline, seed):
        random.seed(seed)
        used_bits = 0
        rows = []
        
        def build_row(current_row, used_bits):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            if max_entry - last < remaining:
                return False
            
            candidates = []
            for x in range(last + 1, max_entry + 1):
                if max_entry - x < remaining - 1:
                    break
                
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if conflict:
                    continue
                
                # Heuristic: prefer smaller x, but penalize if it uses very small diffs
                # which might be critical for other rows? 
                # Actually, just prefer smaller x to minimize scope locally.
                # Add slight randomization to order
                score = x + random.random() * 0.1
                candidates.append((score, x, new_diff_bits))
            
            random.shuffle(candidates) # Randomize to escape local minima
            
            for score, x, new_diff_bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                new_used = used_bits | new_diff_bits
                
                if build_row(current_row, new_used):
                    return True
                
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            
            current_row = [0]
            if build_row(current_row, used_bits):
                rows.append(current_row)
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows
    
    best_rows = greedy_rows
    best_max = greedy_max
    
    # Try to find better solution
    # Search range: lb to min(greedy_max, lb + 30)
    max_search = min(greedy_max, lb + 30)
    
    for M in range(lb, max_search + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 5
        if remaining_time < 1.0:
            break
        
        # Budget for this M
        # Give more time to lower M
        budget = min(5.0, remaining_time / (max_search - M + 1))
        if budget < 0.5:
            break
        
        # Try multiple restarts within budget
        attempts = 0
        while time.time() - start_time < start_time + budget:
            result = solve_csp(M, start_time + budget, attempts)
            if result is not None:
                best_rows = result
                best_max = M
                # Found a solution for this M. Since we start from lb, this is likely optimal or near-optimal.
                # We can break out of the M loop if we are confident, but let's just return the first valid one found for M.
                break
            attempts += 1
            
        if best_max == M:
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
