import time

def construct(n_rows, k):
    # Fallback: Greedy algorithm from seed
    def greedy():
        used, rows = set(), []
        for _ in range(n_rows):
            row, x = [0], 0
            while len(row) < k + 1:
                x += 1
                new = {x - a for a in row}
                if not new & used:
                    row.append(x)
                    used |= new
            rows.append(row)
        return rows

    best_rows = greedy()
    best_scope = max(max(r) for r in best_rows)
    
    start_time = time.time()
    time_limit = 100.0
    
    # Try to find a better solution using backtracking
    # We know the optimal is around 112.
    # We search from the lower bound up to best_scope - 1.
    # Lower bound: total_diffs = n_rows * k*(k+1)//2
    # M(M+1)/2 >= total_diffs => M >= ceil((-1 + sqrt(1 + 8*total_diffs))/2)
    total_diffs = n_rows * (k * (k + 1) // 2)
    import math
    lower_bound = math.ceil((-1 + math.sqrt(1 + 8 * total_diffs)) / 2)
    
    for M in range(lower_bound, best_scope):
        if time.time() - start_time > time_limit:
            break
        
        solution = backtrack(n_rows, k, M)
        if solution:
            best_rows = solution
            best_scope = M
            # If we hit the lower bound, we can't do better
            if M == lower_bound:
                break
                
    return best_rows

def backtrack(n_rows, k, M):
    used_diffs = set()
    rows = []
    
    def recurse(row_idx, current_row, local_diffs):
        if len(current_row) == k + 1:
            # Row complete
            used_diffs.update(local_diffs)
            rows.append(current_row)
            
            if row_idx + 1 == n_rows:
                return True
            
            if recurse(row_idx + 1, [0], set()):
                return True
            
            # Backtrack
            rows.pop()
            used_diffs.difference_update(local_diffs)
            return False
        
        last_mark = current_row[-1]
        needed = (k + 1) - len(current_row)
        # We need 'needed' more marks, including the one we are about to pick?
        # No, current_row has len L. We need k+1 total.
        # So we need (k+1) - L more marks.
        # The next mark is the first of these.
        # The largest mark will be at most M.
        # The smallest possible sequence for the remaining 'needed' marks is
        # next_mark, next_mark+1, ..., next_mark+needed-1.
        # So next_mark + needed - 1 <= M
        # next_mark <= M - needed + 1
        max_next = M - needed + 1
        
        for next_mark in range(last_mark + 1, max_next + 1):
            new_local_diffs = set(local_diffs)
            conflict = False
            for prev in current_row:
                d = next_mark - prev
                if d in used_diffs or d in new_local_diffs:
                    conflict = True
                    break
                new_local_diffs.add(d)
            
            if conflict:
                continue
            
            if recurse(row_idx, current_row + [next_mark], new_local_diffs):
                return True
        
        return False

    if recurse(0, [0], set()):
        return rows
    return None
