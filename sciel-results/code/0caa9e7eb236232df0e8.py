import time
import random
import sys

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

def backtracking_construct(n_rows, k, limit, time_limit, seed=42):
    """
    Attempts to construct n_rows of k+1 integers, all <= limit, using backtracking.
    Returns a list of rows if successful, None otherwise.
    """
    random.seed(seed)
    used = set()
    rows = []
    start_time = time.time()
    
    # We build rows one by one.
    # For each row, we perform a DFS to find a valid sequence of k+1 numbers.
    
    def dfs_row(row, depth, current_max):
        """
        Recursively build the current row.
        row: list of numbers chosen so far for this row
        depth: number of elements currently in row
        current_max: the last number in row
        """
        if time.time() - start_time > time_limit:
            return False
        
        if depth == k + 1:
            return True
        
        # Generate candidates for the next number
        # We want to minimize the final max, so we prefer smaller numbers.
        # However, we can use a heuristic to prioritize candidates that are "safer".
        # A "safer" candidate is one that doesn't block too many future small numbers.
        
        candidates = []
        for x in range(current_max + 1, limit + 1):
            diffs = {x - a for a in row}
            if not (diffs & used):
                candidates.append(x)
        
        if not candidates:
            return False
            
        # Heuristic: Sort candidates. 
        # Standard greedy takes the first. 
        # To allow backtracking to be effective, we should perhaps shuffle slightly 
        # or use a score. But for minimizing max, smaller is generally better.
        # Let's stick to sorted order (smallest first) but allow the caller 
        # (the main loop) to restart with different seeds if we get stuck, 
        # OR we can randomize the order of candidates slightly to explore different branches.
        
        # Strategy: Try candidates in a slightly randomized order if we've failed 
        # many times, or just strictly smallest first? 
        # Strictly smallest first is deterministic. If it fails, it's because 
        # that specific path is a dead end. Backtracking handles this.
        # However, if we always pick the smallest, we might waste time on a 
        # branch that fails late. 
        # Let's try the smallest few candidates first, but if we want to be 
        # more robust, we can randomize the order of the first 'window' candidates.
        
        # For now, let's try the smallest candidate first. If it leads to a dead end,
        # the recursion will unwind and try the next one.
        
        for x in candidates:
            # Check if adding x is valid (already checked by candidate generation)
            # Save state
            old_used = set(used)
            row.append(x)
            
            # Update used
            for a in row[:-1]:
                used.add(x - a)
                
            if dfs_row(row, depth + 1, x):
                return True
                
            # Backtrack
            used.clear()
            used.update(old_used)
            row.pop()
            
        return False

    # Main loop for rows
    for i in range(n_rows):
        if time.time() - start_time > time_limit:
            return None
        
        row = [0]
        # We need to find a valid row given the current 'used' set.
        # We use the same DFS logic.
        # Note: 'used' is global across rows.
        
        # We need to wrap the row construction in a way that allows 
        # backtracking *within* the row construction, but if the whole row 
        # fails, we might need to backtrack previous rows? 
        # That's extremely expensive. 
        # Instead, we assume that if we can find *any* valid row for the current 
        # 'used' set, we take it. If we can't, we fail.
        # This is "Greedy with Backtracking within Row".
        
        if not dfs_row(row, 1, 0):
            return None
        
        rows.append(row)
        
    return rows

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 120s limit
    
    # 1. Greedy Baseline
    best_rows = greedy_construct(n_rows, k)
    best_scope = max(max(r) for r in best_rows)
    
    # 2. Try to improve using Backtracking
    # We try to find a solution with scope L < best_scope.
    # We iterate L downwards.
    
    # Theoretical lower bound: total differences
    min_possible = n_rows * k * (k + 1) // 2
    
    current_limit = best_scope - 1
    
    # We allocate a time budget for each limit attempt.
    # If we fail to find a solution at L, we move to L-1.
    # We should spend more time on higher L (closer to best_scope) as they are more likely.
    
    seed_counter = 0
    
    while current_limit >= min_possible and time.time() < deadline:
        
        # Determine time budget for this limit
        # If current_limit is close to best_scope, we have high confidence, so we can try harder?
        # Or if we are far down, we should give up quickly?
        # Let's give a fixed small budget per limit, e.g., 5 seconds, 
        # but allow more if we are near the start.
        
        # Actually, a single backtracking run might be fast or slow.
        # Let's run backtracking_construct with a tight time limit.
        
        # If we find a solution, we update best_scope and continue downwards.
        # If we don't, we decrement limit.
        
        # To increase diversity, we can run multiple seeds if time permits.
        
        found = False
        # Try a few seeds for this limit
        max_seeds_for_limit = 5
        
        for seed_idx in range(max_seeds_for_limit):
            if time.time() > deadline - 5:
                break
            
            seed = seed_counter
            seed_counter += 1
            
            # Allocate remaining time minus safety margin
            # But cap it so we don't get stuck on one limit for too long
            # If we have 100s left, don't spend 100s on one limit.
            # Spend at most 10s per limit attempt?
            time_budget = min(10.0, deadline - time.time() - 5)
            
            if time_budget <= 0:
                break
            
            rows = backtracking_construct(n_rows, k, current_limit, time_budget, seed)
            
            if rows is not None:
                # Verify validity
                all_diffs = set()
                valid = True
                for r in rows:
                    if len(r) != k + 1:
                        valid = False
                        break
                    if r[0] != 0:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            d = r[j] - r[i]
                            if d in all_diffs:
                                valid = False
                                break
                            all_diffs.add(d)
                    if not valid:
                        break
                
                if valid:
                    new_scope = max(max(r) for r in rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_rows = rows
                    # We found a solution at this limit. 
                    # We can try to lower the limit further.
                    current_limit = new_scope - 1
                    found = True
                    break
        
        if not found:
            # Failed to find a solution at this limit with the seeds tried.
            # Move to next lower limit.
            current_limit -= 1

    return best_rows

if __name__ == "__main__":
    # Simple test
    rows = construct(4, 7)
    print(rows)
