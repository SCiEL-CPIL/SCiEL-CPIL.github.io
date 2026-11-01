import random
import time

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    if n_rows == 0 or k < 0:
        return []
        
    # Total distinct differences required
    total_diffs = n_rows * k * (k + 1) // 2
    lower_bound = total_diffs
    if lower_bound < k:
        lower_bound = k
        
    # Upper bound
    # Greedy usually gives a decent upper bound
    # Let's compute greedy scope to set a reasonable upper bound
    def greedy_solve():
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

    greedy_sol = greedy_solve()
    upper_bound = max(greedy_sol[r][-1] for r in range(n_rows)) if n_rows > 0 else 0
    
    # We search from lower_bound to upper_bound
    # If we find a solution, we return it.
    # Since we iterate upwards, the first solution is the best (smallest M)
    # for our search strategy.
    
    for M in range(lower_bound, upper_bound + 1):
        if time.time() > deadline:
            break
            
        # Try to solve for this M
        # We use randomized backtracking with symmetry breaking
        found = False
        # Number of attempts per M
        # If M is close to lower bound, it's harder.
        # We can increase attempts for smaller M.
        attempts = 100
        
        for _ in range(attempts):
            if time.time() > deadline:
                break
            sol = solve_backtracking(n_rows, k, M)
            if sol is not None:
                return sol
                
    # If no solution found, return greedy
    return greedy_sol

def solve_backtracking(n_rows, k, M):
    rows = [[0] for _ in range(n_rows)]
    used_diffs = set()
    
    def backtrack_sym(r, c):
        if r == n_rows:
            return True
        if c == k + 1:
            return backtrack_sym(r + 1, 1)
            
        prev = rows[r][c-1]
        # Candidates
        # Optimization: If c==1, we have symmetry constraint
        start_val = prev + 1
        if c == 1 and r > 0:
            start_val = max(start_val, rows[r-1][1] + 1)
            
        if start_val > M:
            return False
            
        candidates = list(range(start_val, M + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            new_diffs = []
            for i in range(c-1):
                d = x - rows[r][i]
                if d in used_diffs:
                    valid = False
                    break
                new_diffs.append(d)
            
            if valid:
                rows[r].append(x)
                used_diffs.update(new_diffs)
                
                if backtrack_sym(r, c + 1):
                    return True
                    
                rows[r].pop()
                for d in new_diffs:
                    used_diffs.discard(d)
                    
        return False

    return backtrack_sym(0, 1)
