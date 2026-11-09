import time
import random
import sys
from itertools import combinations

def greedy_construct(n_rows, k):
    """
    Standard greedy construction to establish a baseline.
    """
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < k + 1:
            x += 1
            diffs = {x - a for a in row}
            if not diffs & used:
                row.append(x)
                used.update(diffs)
        rows.append(row)
    return rows

def is_valid(rows, k):
    """Check if the set of rows is valid (all differences distinct)."""
    all_diffs = set()
    for r in rows:
        if len(r) != k + 1:
            return False
        if r[0] != 0:
            return False
        # Check strictly increasing
        for i in range(1, len(r)):
            if r[i] <= r[i-1]:
                return False
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                d = r[j] - r[i]
                if d in all_diffs:
                    return False
                all_diffs.add(d)
    return True

def backtrack_global(n_rows, k, limit, node_limit=50000):
    """
    Backtracking search over the global set of differences.
    We assign differences to rows.
    """
    used_diffs = set()
    rows = [None] * n_rows
    
    # Pre-compute the number of differences needed per row
    # Each row of size k+1 has k*(k+1)/2 differences
    
    def dfs(row_idx, current_row_idx, current_row, candidates_for_current):
        """
        Recursively fill rows.
        row_idx: which row we are filling
        current_row_idx: index of the next element in the current row (0 is always 0)
        current_row: list of elements so far
        candidates_for_current: list of valid next values
        """
        if node_limit[0] < 0:
            return None
        
        node_limit[0] -= 1
        
        # If we have filled all rows
        if row_idx == n_rows:
            return [r[:] for r in rows]
        
        # If current row is complete
        if current_row_idx == k:  # We need k elements after 0, so indices 0..k
            rows[row_idx] = current_row[:]
            # Move to next row
            return dfs(row_idx + 1, 0, [0], None)
        
        # Otherwise, we need to pick the next value for the current row
        if candidates_for_current is None:
            # Generate candidates for the next position
            last_val = current_row[-1]
            candidates = []
            # We can look ahead a bit, but let's just generate valid ones up to limit
            # Actually, we need to pick a value x > last_val such that:
            # 1. x <= limit
            # 2. All differences x - a for a in current_row are in [1, limit] (implicitly true if x<=limit and a>=0)
            # 3. None of these differences are in used_diffs
            # 4. None of these differences conflict with differences that *will be* used? No, we just check current used_diffs.
            
            # Optimization: We can pre-filter candidates based on used_diffs.
            for x in range(last_val + 1, limit + 1):
                valid = True
                for a in current_row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                if valid:
                    candidates.append(x)
            candidates_for_current = candidates
        
        if not candidates_for_current:
            return None
        
        # Heuristic: try smaller candidates first? Or random?
        # Let's try to bias towards smaller to find solutions with smaller scope quickly?
        # Actually, we are fixing the limit, so any solution is valid.
        # But to allow backtracking to work, ordering matters.
        # Let's try candidates in increasing order.
        
        for x in candidates_for_current:
            # Add x to current row
            current_row.append(x)
            
            # Update used_diffs
            new_diffs = []
            for a in current_row[:-1]:
                d = x - a
                new_diffs.append(d)
                used_diffs.add(d)
            
            # Recursive call
            # Note: we pass the remaining candidates? No, we recompute or pass a subset.
            # Passing a subset is hard because removing x doesn't affect validity of other candidates 
            # unless they share diffs with x? No, x is just one value.
            # However, adding x changes used_diffs, which affects future candidates.
            # But for the *current* step, we just need to pick x.
            # The next step will recompute candidates based on the new current_row and used_diffs.
            
            result = dfs(row_idx, current_row_idx + 1, current_row, None)
            
            if result is not None:
                return result
            
            # Backtrack
            current_row.pop()
            for d in new_diffs:
                used_diffs.remove(d)
                
        return None

    # Start DFS
    return dfs(0, 0, [0], None)

def randomized_backtrack(n_rows, k, limit, seed, node_limit=5000):
    """
    Randomized version of backtracking.
    """
    random.seed(seed)
    used_diffs = set()
    rows = [None] * n_rows
    
    def dfs(row_idx, current_row_idx, current_row):
        if node_limit[0] < 0:
            return None
        
        node_limit[0] -= 1
        
        if row_idx == n_rows:
            return [r[:] for r in rows]
        
        if current_row_idx == k:
            rows[row_idx] = current_row[:]
            return dfs(row_idx + 1, 0, [0])
        
        last_val = current_row[-1]
        candidates = []
        for x in range(last_val + 1, limit + 1):
            valid = True
            for a in current_row:
                d = x - a
                if d in used_diffs:
                    valid = False
                    break
            if valid:
                candidates.append(x)
        
        if not candidates:
            return None
        
        # Randomly shuffle candidates? Or pick randomly from a window?
        # Let's pick randomly from all valid candidates to explore different structures.
        # But this might be slow. Let's pick from the first N valid candidates?
        # No, let's just random.choice from all.
        
        # To make it more efficient, let's limit the search space.
        # If we pick a very large x, we might run out of space for subsequent values.
        # Let's bias towards smaller values?
        # Weight by 1/x? Or just uniform?
        
        # Let's try uniform random from candidates.
        x = random.choice(candidates)
        
        current_row.append(x)
        new_diffs = [x - a for a in current_row[:-1]]
        used_diffs.update(new_diffs)
        
        result = dfs(row_idx, current_row_idx + 1, current_row)
        
        if result is not None:
            return result
        
        current_row.pop()
        used_diffs.difference_update(new_diffs)
        
        return None

    node_limit = [node_limit]
    return dfs(0, 0, [0])

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Leave some margin
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Try to improve with Backtracking
    # Theoretical lower bound
    min_possible = n_rows * k * (k + 1) // 2
    
    # We know a solution exists at best_scope.
    # Try to find one at best_scope - 1, then -2, etc.
    current_limit = best_scope - 1
    
    # Use randomized backtracking to find solutions at specific limits
    seed_counter = 0
    trials_per_limit = 20
    
    while time.time() < deadline and current_limit >= min_possible:
        found = False
        for _ in range(trials_per_limit):
            if time.time() > deadline - 5:
                break
            
            seed = seed_counter
            seed_counter += 1
            
            # Increase node limit as we go? Or keep fixed?
            # Fixed node limit per trial.
            rows = randomized_backtrack(n_rows, k, current_limit, seed, node_limit=20000)
            
            if rows is not None:
                if is_valid(rows, k):
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = rows
                    # Lower the limit to search for even better solutions
                    current_limit = new_scope - 1
                    found = True
                    break
        
        if not found:
            current_limit -= 1
            
    return best_rows

if __name__ == "__main__":
    # Simple test
    rows = construct(4, 7)
    print(rows)
    print("Scope:", max(max(r) for r in rows))
    print("Valid:", is_valid(rows, 7))
