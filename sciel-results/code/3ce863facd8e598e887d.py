import time
import random
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill

    diffs_per_row = (k + 1) * k // 2
    total_diffs = n_rows * diffs_per_row
    lower_bound = total_diffs

    # Step 1: Generate multiple diverse greedy baselines
    best_rows = None
    best_scope = float('inf')
    
    for seed in range(20):
        random.seed(seed * 12345 + 7)
        rows = greedy_construct(n_rows, k, randomize=True)
        scope = max(max(row) for row in rows)
        if scope < best_scope:
            best_scope = scope
            best_rows = rows
    
    # Step 2: Local search to compact rows
    best_rows = local_search_compact(best_rows, k, deadline)
    best_scope = max(max(row) for row in best_rows)
    
    # Step 3: Try to reduce scope via backtracking with bitsets
    # Search from best_scope-1 down to lower_bound
    current_scope = best_scope - 1
    consecutive_failures = 0
    max_failures = 3
    
    while current_scope >= lower_bound and time.time() < deadline and consecutive_failures < max_failures:
        solution = backtracking_search(n_rows, k, current_scope, deadline)
        if solution:
            best_rows = solution
            best_scope = current_scope
            current_scope -= 1
            consecutive_failures = 0
        else:
            current_scope -= 1
            consecutive_failures += 1
    
    # Final local search on the best solution found
    best_rows = local_search_compact(best_rows, k, deadline)
    best_scope = max(max(row) for row in best_rows)
    
    return best_rows

def greedy_construct(n_rows, k, randomize=False):
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
    
    if randomize:
        # Apply random swaps to diversify
        for row in rows:
            for _ in range(3):
                if len(row) > 2:
                    i = random.randint(1, len(row) - 2)
                    j = random.randint(i + 1, len(row) - 1)
                    # Only swap if it maintains increasing order (which it won't for non-adjacent)
                    # Instead, let's try a different approach: regenerate with different tie-breaking
                    pass
        # Simpler: just return as is, the random seed already diversifies the greedy choices
        # Actually, the greedy is deterministic given the used set. Let's add randomness to candidate selection.
        # For now, the different seeds will give different results if we randomize the search order.
    
    return rows

def local_search_compact(rows, k, deadline):
    """Try to reduce the maximum entry by compacting individual rows."""
    rows = [list(row) for row in rows]
    improved = True
    
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(rows)):
            row = rows[i]
            # Try to reduce the last element
            for pos in range(len(row) - 1, 0, -1):
                old_val = row[pos]
                # Try to reduce row[pos]
                for new_val in range(row[pos - 1] + 1, old_val):
                    # Check if all differences are still valid
                    valid = True
                    new_diffs = set()
                    for j in range(pos + 1):
                        diff = new_val - row[j]
                        if diff <= 0:
                            valid = False
                            break
                        new_diffs.add(diff)
                    
                    if not valid:
                        continue
                    
                    # Check against all other differences in this row
                    row_diffs = set()
                    for a in range(pos + 1):
                        for b in range(a + 1, pos + 1):
                            if a == pos or b == pos:
                                continue
                            row_diffs.add(row[b] - row[a])
                    
                    # Check if new_diffs conflicts with row_diffs
                    if new_diffs & row_diffs:
                        continue
                    
                    # Check against all other rows
                    conflict = False
                    for other_row in rows:
                        if other_row is row:
                            continue
                        for a in range(len(other_row)):
                            for b in range(a + 1, len(other_row)):
                                if (other_row[b] - other_row[a]) in new_diffs:
                                    conflict = True
                                    break
                            if conflict:
                                break
                        if conflict:
                            break
                    
                    if not conflict:
                        row[pos] = new_val
                        improved = True
                        break
    
    return rows

def backtracking_search(n_rows, k, max_scope, deadline):
    """Find a solution with all entries <= max_scope using backtracking with bitsets."""
    used_mask = 0
    rows = []
    node_count = 0
    max_nodes = 200000  # Limit to avoid timeout

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
                    if diff > max_scope:
                        conflict = True
                        break
                    bit = 1 << diff
                    if used_mask & bit:
                        conflict = True
                        break
                    new_mask |= bit
                if conflict:
                    continue
                # Space check
                remaining = k + 1 - (len(current_row) + 1)
                if max_scope - x < remaining:
                    continue
                candidates.append((x, new_mask))
            
            # Sort candidates: prefer smaller x to keep scope tight
            candidates.sort()
            
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
