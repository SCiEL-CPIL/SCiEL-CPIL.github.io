import time
import random

def construct(n_rows, k):
    # The task specifies n_rows=7, k=5.
    # But we should handle general n_rows, k if possible, or just optimize for 7,5.
    # The evaluator passes n_rows and k.
    
    # Lower bound
    lb = n_rows * (k * (k + 1) // 2)
    
    best_solution = None
    best_M = float('inf')
    
    # Time management
    start_time = time.time()
    deadline = start_time + 110
    
    # Try to find a solution with M close to lb
    # We will try M from lb to lb + 10
    for M in range(lb, lb + 11):
        if time.time() > deadline:
            break
            
        # Use backtracking
        solution = solve_for_M(n_rows, k, M, time.time() + 10)
        if solution:
            # Verify solution
            if is_valid(solution, k):
                return solution
                
    # If no solution found, use greedy
    return greedy_construct(n_rows, k)

def is_valid(rows, k):
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

def greedy_construct(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            new = {x - a for a in row}
            if not new & used:
                # Check internal distinctness
                # The greedy logic in seed assumes this?
                # No, seed just checks `new & used`.
                # But `new` contains diffs to ALL previous elements.
                # If the row was a Golomb ruler so far, and new diffs are distinct from old diffs,
                # then the new row is a Golomb ruler.
                # Is it guaranteed that `new` has distinct elements?
                # x - a_i. Since a_i are distinct, x - a_i are distinct.
                # So `new` is a set of distinct values.
                # So the condition `not new & used` ensures global distinctness.
                # And internal distinctness is preserved.
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def solve_for_M(n_rows, k, M, deadline):
    used_diffs = [0] * (M + 1)
    used_count = 0
    rows = []
    
    def backtrack(row_idx, current_row, current_diffs):
        nonlocal used_count
        
        if time.time() > deadline:
            return False
            
        if row_idx == n_rows:
            return True
            
        # Pruning
        remaining_rows = n_rows - row_idx
        min_remaining_diffs = (k * (k + 1) // 2) - len(current_diffs) + (remaining_rows - 1) * (k * (k + 1) // 2)
        if M - used_count < min_remaining_diffs:
            return False

        if len(current_row) == k + 1:
            # Lexicographic check
            if row_idx > 0:
                if current_row < rows[-1]:
                    return False
            
            rows.append(current_row[:])
            if backtrack(row_idx + 1, [0], []):
                return True
            rows.pop()
            return False
            
        last = current_row[-1]
        for x in range(last + 1, M + 1):
            valid = True
            new_diffs = []
            for y in current_row:
                d = x - y
                if d > M:
                    valid = False
                    break
                if used_diffs[d]:
                    valid = False
                    break
                new_diffs.append(d)
                
            if not valid:
                continue
                
            if any(d in current_diffs for d in new_diffs):
                continue
                
            current_row.append(x)
            current_diffs.extend(new_diffs)
            
            for d in new_diffs:
                used_diffs[d] = 1
            used_count += len(new_diffs)
            
            if backtrack(row_idx, current_row, current_diffs):
                return True
                
            for d in new_diffs:
                used_diffs[d] = 0
            used_count -= len(new_diffs)
            del current_diffs[len(current_diffs)-len(new_diffs):]
            current_row.pop()
            
        return False
        
    if backtrack(0, [0], []):
        return rows
    return None
