import time
import sys
import random
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115
    
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Phase 1: Local search to minimize scope
    while time.time() < deadline - 25:
        improved = _local_search_minimize_fast(best_solution, n_rows, k, deadline)
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
            else:
                break
    
    # Phase 2: ILS - perturb and local search, with targeted perturbation
    while time.time() < deadline - 15:
        candidate = [r[:] for r in best_solution]
        
        # Targeted perturbation: focus on positions with high values
        n_perturb = random.randint(2, 6)
        for _ in range(n_perturb):
            # 50% chance to target a high-value position, 50% random
            if random.random() < 0.5 and n_rows > 0:
                # Find the row with the maximum last element
                max_row_idx = max(range(n_rows), key=lambda i: candidate[i][-1])
                # Pick a random position in that row (bias towards later positions)
                pos = random.randint(max(1, k // 2), k)
                r_idx = max_row_idx
            else:
                r_idx = random.randint(0, n_rows - 1)
                pos = random.randint(1, k)
            
            new_val = _find_random_valid_value_fast(candidate, r_idx, pos, n_rows, k)
            if new_val is not None and new_val != candidate[r_idx][pos]:
                candidate[r_idx][pos] = new_val
        
        # Local search to minimize scope
        improved = _local_search_minimize_fast(candidate, n_rows, k, deadline)
        
        if improved:
            curr_score = max(max(r) for r in improved)
            if curr_score < best_score:
                best_score = curr_score
                best_solution = improved
            elif random.random() < 0.1:
                # Occasionally accept equal solutions to escape local optima
                best_solution = improved
    
    # Final polish: aggressive local search with multiple restarts
    if time.time() < deadline - 5:
        for _ in range(3):
            if time.time() > deadline - 5:
                break
            final = _local_search_minimize_fast([r[:] for r in best_solution], n_rows, k, deadline)
            if final:
                final_score = max(max(r) for r in final)
                if final_score < best_score:
                    best_solution = final
                    best_score = final_score
                else:
                    break
    
    return best_solution

def _find_random_valid_value_fast(rows, r_idx, pos, n_rows, k):
    # Find a random valid value for rows[r_idx][pos]
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    
    # Compute all diffs in the system
    all_diffs = set()
    for r in rows:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Diffs involving old_val at position pos
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Determine search range - allow both smaller and larger values for perturbation
    lower_bound = row[pos-1] + 1
    upper_bound = 250
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    # Collect valid candidates
    valid_candidates = []
    for new_val in range(lower_bound, upper_bound + 1):
        if new_val == old_val:
            continue
        
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            valid_candidates.append(new_val)
    
    if not valid_candidates:
        return None
    
    # Bias towards values that reduce the maximum
    if valid_candidates:
        # Weight selection towards smaller values
        weights = [1.0 / (v + 1) for v in valid_candidates]
        total_weight = sum(weights)
        r = random.random() * total_weight
        cumulative = 0
        for v, w in zip(valid_candidates, weights):
            cumulative += w
            if r <= cumulative:
                return v
        return random.choice(valid_candidates)
    
    return None

def _local_search_minimize_fast(rows, n_rows, k, deadline):
    # Fast local search with incremental diff tracking
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set
    all_diffs = set()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    # Count how many times each diff is used (for potential future use)
    progress = True
    no_improvement_count = 0
    max_no_improvement = 5
    
    while progress and time.time() < deadline - 5:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 5:
                break
                
            row = best_rows[r_idx]
            old_val = row[pos]
            
            # Compute diffs involving (r_idx, pos)
            diffs_involving = set()
            for i in range(n):
                if i == pos:
                    continue
                if i < pos:
                    diffs_involving.add(old_val - row[i])
                else:
                    diffs_involving.add(row[i] - old_val)
            
            forbidden_diffs = all_diffs - diffs_involving
            
            # Try to find smaller valid value
            lower_bound = row[pos-1] + 1
            upper_bound = old_val - 1
            if pos + 1 < n:
                upper_bound = min(upper_bound, row[pos+1] - 1)
            
            if upper_bound >= lower_bound:
                found = False
                for new_val in range(lower_bound, upper_bound + 1):
                    valid = True
                    new_diffs = set()
                    for i in range(n):
                        if i == pos:
                            continue
                        if i < pos:
                            d = new_val - row[i]
                        else:
                            d = row[i] - new_val
                        
                        if d in forbidden_diffs or d in new_diffs:
                            valid = False
                            break
                        new_diffs.add(d)
                    
                    if valid:
                        # Update: remove old diffs, add new diffs
                        all_diffs.difference_update(diffs_involving)
                        all_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break
                
                if found:
                    no_improvement_count = 0
                    continue
        
        if not progress:
            no_improvement_count += 1
            if no_improvement_count >= max_no_improvement:
                break
    
    return best_rows

def _greedy_construct(n_rows, k):
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1, with a tighter bound
            for x in range(current_max + 1, current_max + 300):
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
                # Try a wider search with some randomization
                for _retry in range(100):
                    x = current_max + 1 + random.randint(0, 50)
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

def _local_search_ilS(rows, n_rows, k, deadline):
    # Deprecated - use _local_search_minimize_fast instead
    return _local_search_minimize_fast(rows, n_rows, k, deadline)

def _repair(rows, n_rows, k):
    # Placeholder for repair functionality
    return None

def _find_smallest_valid_value(rows, r_idx, pos, old_val, n_rows, k):
    # Kept for compatibility but not used in the fast path
    row = rows[r_idx]
    n = len(row)
    
    all_diffs = set()
    for r in rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    forbidden_diffs = all_diffs - diffs_involving_old
    
    lower_bound = row[pos-1] + 1
    upper_bound = old_val - 1
    if pos + 1 < n:
        upper_bound = min(upper_bound, row[pos+1] - 1)
    
    if upper_bound < lower_bound:
        return None
    
    for new_val in range(lower_bound, upper_bound + 1):
        valid = True
        new_diffs = set()
        for i in range(n):
            if i == pos:
                continue
            if i < pos:
                d = new_val - row[i]
            else:
                d = row[i] - new_val
            
            if d in forbidden_diffs or d in new_diffs:
                valid = False
                break
            new_diffs.add(d)
        
        if valid:
            return new_val
    
    return None
