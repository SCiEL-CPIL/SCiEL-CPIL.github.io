import time
import random
import sys
import numpy as np

sys.setrecursionlimit(10000)

def construct(n_rows, k):
    deadline = time.time() + 115
    
    # First, generate a greedy solution as starting point
    best_solution = _greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_score = max(max(r) for r in best_solution)
    
    # Build initial difference set
    all_diffs = set()
    for r in best_solution:
        for i in range(len(r)):
            for j in range(i+1, len(r)):
                all_diffs.add(r[j] - r[i])
    
    # Simulated annealing with valid moves
    temperature = max(10, best_score * 0.1)
    cooling_rate = 0.995
    min_temperature = 1
    
    while time.time() < deadline - 5 and temperature > min_temperature:
        # Perform a random valid move
        candidate = [r[:] for r in best_solution]
        
        # Pick a random row and position
        r_idx = random.randint(0, n_rows - 1)
        pos = random.randint(1, k)
        
        # Find a random valid new value
        new_val = _find_random_valid_value(candidate, r_idx, pos, all_diffs, n_rows, k)
        
        if new_val is not None and new_val != candidate[r_idx][pos]:
            old_val = candidate[r_idx][pos]
            
            # Update difference set
            _update_diffs(candidate, r_idx, pos, old_val, new_val, all_diffs)
            
            # Compute new score
            new_score = max(max(r) for r in candidate)
            
            # Accept or reject based on simulated annealing
            if new_score < best_score:
                best_score = new_score
                best_solution = candidate
            else:
                # Accept worse solution with probability exp(-delta/temperature)
                delta = new_score - best_score
                if delta > 0:
                    prob = np.exp(-delta / temperature)
                    if random.random() < prob:
                        best_solution = candidate
                        best_score = new_score
                else:
                    best_solution = candidate
                    best_score = new_score
        
        # Local search to minimize
        if time.time() < deadline - 10:
            improved = _local_search_minimize(best_solution, all_diffs, n_rows, k, deadline)
            if improved:
                curr_score = max(max(r) for r in improved)
                if curr_score < best_score:
                    best_score = curr_score
                    best_solution = improved
        
        temperature *= cooling_rate
    
    # Final local search pass
    if time.time() < deadline - 2:
        final = _local_search_minimize(best_solution, all_diffs, n_rows, k, deadline)
        if final:
            final_score = max(max(r) for r in final)
            if final_score < best_score:
                best_solution = final
                best_score = final_score
    
    return best_solution

def _update_diffs(rows, r_idx, pos, old_val, new_val, all_diffs):
    """Update the difference set when changing a value in a row."""
    row = rows[r_idx]
    n = len(row)
    
    # Remove old differences involving old_val
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            all_diffs.discard(old_val - row[i])
        else:
            all_diffs.discard(row[i] - old_val)
    
    # Add new differences involving new_val
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            all_diffs.add(new_val - row[i])
        else:
            all_diffs.add(row[i] - new_val)

def _find_random_valid_value(rows, r_idx, pos, all_diffs, n_rows, k):
    """Find a random valid value for rows[r_idx][pos]."""
    row = rows[r_idx]
    old_val = row[pos]
    n = len(row)
    
    # Compute differences involving old_val
    diffs_involving_old = set()
    for i in range(n):
        if i == pos:
            continue
        if i < pos:
            diffs_involving_old.add(old_val - row[i])
        else:
            diffs_involving_old.add(row[i] - old_val)
    
    # Forbidden differences are all diffs except those involving old_val
    forbidden_diffs = all_diffs - diffs_involving_old
    
    # Determine search range
    lower_bound = row[pos-1] + 1
    upper_bound = 300
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
    
    return random.choice(valid_candidates)

def _local_search_minimize(rows, all_diffs, n_rows, k, deadline):
    """Local search to minimize the maximum entry."""
    best_rows = [r[:] for r in rows]
    n = k + 1
    
    # Build initial diff set if not provided
    if all_diffs is None:
        all_diffs = set()
        for r in best_rows:
            for i in range(n):
                for j in range(i+1, n):
                    all_diffs.add(r[j] - r[i])
    
    # Work on a copy of all_diffs
    working_diffs = set(all_diffs)
    
    while time.time() < deadline - 3:
        progress = False
        indices = [(r_idx, pos) for r_idx in range(n_rows) for pos in range(1, k+1)]
        random.shuffle(indices)
        
        for r_idx, pos in indices:
            if time.time() > deadline - 3:
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
            
            forbidden_diffs = working_diffs - diffs_involving
            
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
                        working_diffs.difference_update(diffs_involving)
                        working_diffs.update(new_diffs)
                        best_rows[r_idx][pos] = new_val
                        progress = True
                        found = True
                        break
                
                if found:
                    continue
        
        if not progress:
            break
    
    # Update all_diffs to match best_rows
    all_diffs.clear()
    for r in best_rows:
        for i in range(n):
            for j in range(i+1, n):
                all_diffs.add(r[j] - r[i])
    
    return best_rows

def _greedy_construct(n_rows, k):
    """Greedy construction of initial solution."""
    used_diffs = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        current_max = 0
        while len(row) < k + 1:
            found = False
            # Try values starting from current_max + 1
            for x in range(current_max + 1, current_max + 500):
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
                for _retry in range(50):
                    x = current_max + 1 + random.randint(0, 100)
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
