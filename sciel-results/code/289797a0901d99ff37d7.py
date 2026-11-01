import time
import sys
import random

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 110
    
    # Initial greedy construction
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Randomized Greedy + Local Search
    # We try to find a better solution by randomizing the order or choices
    # and then locally optimizing the scope.
    
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Randomize the search slightly
        # Strategy: Construct rows one by one, but with some randomization in choice of next element
        # or by shuffling the order of rows construction (though rows are indistinguishable, 
        # the set of used diffs matters).
        
        # Actually, a better strategy is to generate a candidate via randomized greedy,
        # then apply local search to minimize the max element.
        
        candidate = _randomized_greedy(n_rows, k, deadline)
        if candidate:
            # Local search to minimize scope
            improved = _local_search_minimize_scope(candidate, n_rows, k, deadline)
            if improved:
                curr_score = max(max(r) for r in improved)
                if curr_score < best_score:
                    best_score = curr_score
                    best_solution = improved
    
    return best_solution

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            for x in range(current_max + 1, current_max + 1000):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    row.append(x)
                    used_diffs.update(new_diffs)
                    current_max = x
                    found = True
                    break
            if not found:
                return None
        rows.append(row)
    return rows

def _randomized_greedy(n_rows, k, deadline):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            if time.time() > deadline:
                return None
            
            # Find all valid candidates in a reasonable range
            # To introduce randomness, we can pick from the first few valid candidates
            # or add noise to the search order.
            
            valid_candidates = []
            # Search a bit further to have options
            limit = current_max + 50 
            for x in range(current_max + 1, limit):
                new_diffs = []
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    valid_candidates.append((x, new_diffs))
                    if len(valid_candidates) >= 10: # Take first 10 valid
                        break
            
            if not valid_candidates:
                # If none found in small window, expand search (greedy fallback)
                for x in range(limit, current_max + 1000):
                    new_diffs = []
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    if valid:
                        valid_candidates.append((x, new_diffs))
                        break
            
            if not valid_candidates:
                return None
            
            # Pick randomly from valid candidates, biased towards smaller values
            # Weighted random choice: smaller x has higher weight
            total_weight = sum(1/(x+1) for x, _ in valid_candidates)
            r = random.random() * total_weight
            cum = 0
            chosen_idx = 0
            for i, (x, nd) in enumerate(valid_candidates):
                cum += 1/(x+1)
                if r <= cum:
                    chosen_idx = i
                    break
            
            x, new_diffs = valid_candidates[chosen_idx]
            row.append(x)
            used_diffs.update(new_diffs)
            current_max = x
            
        rows.append(row)
    return rows

def _local_search_minimize_scope(rows, n_rows, k, deadline):
    # Try to reduce the maximum entry by swapping elements.
    # We iterate through all elements and try to replace them with smaller valid values.
    # This is a heuristic local search.
    
    current_rows = [r[:] for r in rows]
    used_diffs = set()
    for r in current_rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                used_diffs.add(r[j] - r[i])
                
    improved = False
    # Repeat until no improvement or time out
    while time.time() < deadline - 5:
        progress = False
        # Shuffle rows to avoid bias
        random.shuffle(current_rows)
        
        for r_idx in range(n_rows):
            if time.time() > deadline - 5:
                break
            row = current_rows[r_idx]
            # Try to minimize each element from largest to smallest? 
            # Or just iterate. Let's try to reduce the global max.
            # Identify the current global max
            global_max = max(max(r) for r in current_rows)
            
            # For each element in this row (excluding 0), try to replace with smaller valid value
            for pos in range(len(row)-1, 0, -1): # From end to start
                old_val = row[pos]
                if old_val < 1: continue
                
                # Find min possible replacement
                # Must be > row[pos-1]
                # Must not create duplicate diffs with other elements in row
                # Must not create duplicate diffs with other rows
                
                # Collect diffs involving this element
                # Diffs with previous elements in same row: old_val - row[i] for i < pos
                # Diffs with next elements in same row: row[j] - old_val for j > pos
                # Diffs with other rows: all diffs in other rows are in used_diffs, 
                # but we need to remove diffs involving old_val first.
                
                # Remove diffs involving old_val from used_diffs
                diffs_to_remove = set()
                for i in range(pos):
                    diffs_to_remove.add(old_val - row[i])
                for j in range(pos+1, len(row)):
                    diffs_to_remove.add(row[j] - old_val)
                
                # Also need to check if these diffs are used in OTHER rows.
                # If a diff is used in another row, we cannot reuse it.
                # So the available diffs are (used_diffs - diffs_to_remove) is not quite right.
                # used_diffs contains ALL diffs.
                # When we change row[r_idx][pos] to new_val, the diffs involving it change.
                # The new diffs must not be in (used_diffs - diffs_to_remove).
                
                # Let's compute the set of "forbidden" diffs for the new value.
                # Forbidden = (used_diffs - diffs_to_remove)
                # But we also need to ensure new diffs within this row are unique.
                
                forbidden_diffs = used_diffs - diffs_to_remove
                
                # Try values from row[pos-1] + 1 upwards
                # We want to find the smallest valid new_val
                # But we also want to reduce scope, so we try values < old_val first?
                # Actually, if we find ANY valid value < old_val, it's an improvement for this element.
                # To minimize global max, we should prioritize reducing the largest elements.
                
                # Let's try to find the smallest valid value > row[pos-1]
                # If smallest valid < old_val, we update.
                
                found_smaller = False
                for new_val in range(row[pos-1] + 1, old_val):
                    # Check validity
                    valid = True
                    new_diffs = set()
                    for i in range(pos):
                        d = new_val - row[i]
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if not valid: continue
                    
                    for j in range(pos+1, len(row)):
                        d = row[j] - new_val
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                        
                    if valid:
                        # Update used_diffs
                        used_diffs.difference_update(diffs_to_remove)
                        used_diffs.update(new_diffs)
                        row[pos] = new_val
                        progress = True
                        found_smaller = True
                        break
                
                # If we didn't find a smaller one, we don't change it in this pass.
                # We could also try to find a valid one that is not smaller but allows future improvements?
                # For simplicity, only accept if it reduces the value.
                
        if not progress:
            break
            
    return current_rows
