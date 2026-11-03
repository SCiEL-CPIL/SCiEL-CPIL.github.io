import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Total differences per row: C(k+1, 2)
    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    
    # Simple counting bound: we need at least total_diffs distinct positive differences
    # If max entry is M, differences are in [1, M], so M >= total_diffs
    lower_bound = total_diffs
    
    # Strategy:
    # 1. Generate a strong initial solution using a modified greedy with randomization and local search.
    # 2. Try to reduce the scope by searching for solutions with smaller M using clique finding.
    
    # Step 1: Get a baseline solution
    baseline_rows = greedy_construct(n_rows, k)
    best_scope = max(max(row) for row in baseline_rows)
    best_rows = baseline_rows
    
    # Step 2: Try to improve by finding a solution with smaller scope
    # We search for the minimum M such that there exists a set of n_rows rows with max entry <= M
    # and all differences are distinct.
    
    # We can model this as: for a given M, generate all valid single-row patterns (Sidon sequences)
    # with max entry <= M, and find a clique of size n_rows in the conflict graph.
    # However, generating all Sidon sequences is expensive. Instead, we use a hybrid approach:
    # - Use randomized backtracking to find solutions for decreasing M.
    # - Use the clique insight to guide the search: prioritize rows that are "compatible" with each other.
    
    current_scope = best_scope
    consecutive_failures = 0
    max_consecutive_failures = 5
    
    while current_scope > lower_bound and time.time() < deadline and consecutive_failures < max_consecutive_failures:
        # Try to find a solution with this scope using randomized backtracking
        solution = backtracking_search(n_rows, k, current_scope, deadline)
        if solution:
            best_rows = solution
            best_scope = current_scope
            current_scope -= 1
            consecutive_failures = 0
        else:
            current_scope -= 1
            consecutive_failures += 1
            
    return best_rows

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
                row.append(x)
                used |= new
        rows.append(row)
    return rows

def backtracking_search(n_rows, k, max_scope, deadline):
    used_mask = 0
    rows = []
    node_count = 0
    max_nodes = 500000 # Increased limit for more thorough search
    
    def backtrack(row_idx, current_row):
        nonlocal used_mask, node_count
        if time.time() > deadline or node_count > max_nodes:
            return False
        node_count += 1
        
        if row_idx == n_rows:
            return True
            
        if len(current_row) == k + 1:
            rows.append(current_row[:])
            if backtrack(row_idx + 1, [0]):
                return True
            rows.pop()
            return False
        else:
            last = current_row[-1]
            candidates = []
            for x in range(last + 1, max_scope + 1):
                new_mask = 0
                conflict = False
                for a in current_row:
                    diff = x - a
                    bit = 1 << diff
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue
                # Space check: ensure we can fit the remaining elements
                remaining = k + 1 - (len(current_row) + 1)
                if max_scope - x < remaining:
                    continue
                candidates.append((x, new_mask))
            
            # Randomize candidate order for diversity
            random.shuffle(candidates)
            
            for x, new_mask in candidates:
                used_mask |= new_mask
                current_row.append(x)
                
                if backtrack(row_idx, current_row):
                    return True
                
                current_row.pop()
                used_mask ^= new_mask
                
            return False
            
    if backtrack(0, [0]):
        return rows
    return None
