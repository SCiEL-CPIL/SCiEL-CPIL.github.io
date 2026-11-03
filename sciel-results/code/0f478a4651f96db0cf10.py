import time
import random
import math
import sys
import numpy as np
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 118.0
    
    # Theoretical lower bound: need n_rows * k*(k+1)/2 distinct positive differences
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs - 1  # differences are 1..M, so max entry >= num_diffs - 1
    
    k_plus_1 = k + 1
    
    # 1. Fast greedy baseline
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    best_scope = max(max(row) for row in best_solution)
    
    if best_scope <= lower_bound:
        return best_solution
    
    # 2. SA-based row regeneration for coarse improvement
    if time.time() < deadline - 20.0:
        sa_solution, sa_scope = _sa_row_regeneration(best_solution, n_rows, k, deadline - 15.0)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_solution = sa_solution
            if best_scope <= lower_bound:
                return best_solution
    
    # 3. Fine-grained element-level local search for polish
    if time.time() < deadline - 10.0:
        polished, polish_scope = _element_local_search(best_solution, n_rows, k, deadline - 8.0)
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_solution = polished
            if best_scope <= lower_bound:
                return best_solution
    
    # 4. Targeted backtracking search for further scope reduction
    if time.time() < deadline - 5.0:
        attempt = 0
        while time.time() < deadline - 5.0:
            target_scope = best_scope - 1
            if target_scope < lower_bound:
                break
            
            found = False
            for _ in range(20):
                if time.time() > deadline - 5.0:
                    break
                
                solution = _backtracking_search(n_rows, k, target_scope, deadline - 3.0, attempt)
                if solution:
                    new_scope = max(max(row) for row in solution)
                    if new_scope < best_scope:
                        best_scope = new_scope
                        best_solution = solution
                        found = True
                        # Polish the found solution
                        if time.time() < deadline - 3.0:
                            polished, ps = _element_local_search(best_solution, n_rows, k, deadline - 2.0)
                            if ps < best_scope:
                                best_scope = ps
                                best_solution = polished
                        if best_scope <= lower_bound:
                            return best_solution
                        break
                attempt += 1
            
            if not found and target_scope <= lower_bound + 5:
                break
    
    return best_solution

def greedy_construct(n_rows, k):
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
    return rows

def _sa_row_regeneration(solution, n_rows, k, deadline):
    """
    Simulated annealing with row regeneration.
    Coarse-grained moves: regenerate entire rows to escape local minima.
    """
    k_plus_1 = k + 1
    rows = [list(r) for r in solution]
    current_max = max(max(r) for r in rows)
    best_max = current_max
    best_rows = [list(r) for r in rows]
    
    # Track used differences
    used_diffs = set()
    for r in rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                used_diffs.add(r[j] - r[i])
    
    T = 10.0
    cooling = 0.995
    T_min = 0.1
    
    while time.time() < deadline:
        if T < T_min:
            T = 5.0
            cooling = 0.998
        
        r_idx = random.randint(0, n_rows - 1)
        
        # Remove diffs of row r_idx
        old_row = rows[r_idx]
        old_diffs = set()
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                old_diffs.add(old_row[j] - old_row[i])
        
        used_diffs -= old_diffs
        
        # Regenerate row with randomization
        new_row = _generate_row_sa(used_diffs, k, current_max + 20)
        
        if new_row is None:
            used_diffs |= old_diffs
            continue
        
        new_row_max = max(new_row)
        other_max = max(max(rows[i]) for i in range(n_rows) if i != r_idx)
        new_global_max = max(other_max, new_row_max)
        
        delta = new_global_max - current_max
        
        if delta < 0 or random.random() < math.exp(-delta / T):
            rows[r_idx] = new_row
            new_diffs = set()
            for i in range(k_plus_1):
                for j in range(i + 1, k_plus_1):
                    new_diffs.add(new_row[j] - new_row[i])
            used_diffs |= new_diffs
            current_max = new_global_max
            if current_max < best_max:
                best_max = current_max
                best_rows = [list(r) for r in rows]
        else:
            used_diffs |= old_diffs
        
        T *= cooling
    
    return best_rows, best_max

def _generate_row_sa(used_diffs, k, max_search):
    """Generate a single row given the set of already-used differences."""
    k_plus_1 = k + 1
    row = [0]
    current_row_diffs = set()
    
    for _ in range(k):
        candidates = []
        x = row[-1] + 1
        forbidden = used_diffs | current_row_diffs
        
        while x < max_search and len(candidates) < 8:
            new_diffs = {x - a for a in row}
            if not (new_diffs & forbidden):
                candidates.append(x)
            x += 1
        
        if not candidates:
            return None
        
        # Biased random choice: prefer smaller values
        if random.random() < 0.7:
            next_x = candidates[0]
        else:
            next_x = random.choice(candidates)
        
        row.append(next_x)
        current_row_diffs |= {next_x - a for a in row[:-1]}
    
    return row

def _element_local_search(solution, n_rows, k, deadline):
    """
    Fine-grained element-level local search with conflict counting.
    Moves individual elements to reduce scope.
    """
    k_plus_1 = k + 1
    current_rows = [list(r) for r in solution]
    current_scope = max(max(r) for r in current_rows)
    best_scope = current_scope
    best_rows = [list(r) for r in current_rows]
    
    # Build diff counts for efficient conflict checking
    diff_counts = defaultdict(int)
    conflict_count = 0
    
    def add_diff(d):
        nonlocal conflict_count
        if d in diff_counts:
            if diff_counts[d] == 1:
                conflict_count += 1
            diff_counts[d] += 1
        else:
            diff_counts[d] = 1
    
    def remove_diff(d):
        nonlocal conflict_count
        if d in diff_counts:
            diff_counts[d] -= 1
            if diff_counts[d] == 0:
                del diff_counts[d]
            elif diff_counts[d] == 1:
                conflict_count -= 1
    
    # Initialize diff counts
    for r in current_rows:
        for i in range(k_plus_1):
            for j in range(i + 1, k_plus_1):
                add_diff(r[j] - r[i])
    
    T = 5.0
    T_min = 0.1
    cooling = 0.999
    
    while time.time() < deadline:
        if T < T_min:
            T = 2.0
            cooling = 0.9995
        
        # Pick a random row and position to modify
        r_idx = random.randint(0, n_rows - 1)
        row = current_rows[r_idx]
        p_idx = random.randint(1, k_plus_1 - 1)  # Don't move position 0
        
        old_val = row[p_idx]
        
        # Remove old_val's contributions
        diffs_to_remove = []
        for i in range(k_plus_1):
            if i != p_idx:
                d = abs(row[i] - old_val)
                diffs_to_remove.append(d)
        
        for d in diffs_to_remove:
            remove_diff(d)
        
        # Remove the element
        row.pop(p_idx)
        
        # Try to find a new valid value
        new_val = None
        new_diffs = []
        
        # Search for a good new value
        max_search = min(best_scope + 10, 250)
        found = False
        
        for _ in range(15):
            if found:
                break
            cand = random.randint(1, max_search)
            if cand in row:
                continue
            
            valid = True
            new_diffs_candidate = []
            for a in row:
                d = abs(cand - a)
                if d in diff_counts and diff_counts[d] > 0:
                    valid = False
                    break
                new_diffs_candidate.append(d)
            
            if valid:
                new_val = cand
                new_diffs = new_diffs_candidate
                found = True
        
        if not found:
            # Revert
            row.insert(p_idx, old_val)
            for d in diffs_to_remove:
                add_diff(d)
            continue
        
        # Insert new value at correct position
        insert_pos = 0
        while insert_pos < len(row) and row[insert_pos] < new_val:
            insert_pos += 1
        row.insert(insert_pos, new_val)
        
        for d in new_diffs:
            add_diff(d)
        
        # Check if row is still strictly increasing
        for i in range(len(row) - 1):
            if row[i] >= row[i + 1]:
                # Revert
                row.pop(insert_pos)
                for d in new_diffs:
                    remove_diff(d)
                row.insert(p_idx, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
                break
        else:
            # Calculate new scope
            new_scope = max(max(r) for r in current_rows)
            delta = new_scope - current_scope
            
            if delta < 0 or random.random() < math.exp(-delta / T):
                current_scope = new_scope
                if current_scope < best_scope:
                    best_scope = current_scope
                    best_rows = [list(r) for r in current_rows]
            else:
                # Revert
                row.pop(insert_pos)
                for d in new_diffs:
                    remove_diff(d)
                row.insert(p_idx, old_val)
                for d in diffs_to_remove:
                    add_diff(d)
        
        T *= cooling
    
    return best_rows, best_scope

def _backtracking_search(n_rows, k, max_scope, deadline, seed_offset=0):
    """
    Backtracking search to build n_rows of k+1 increasing integers starting at 0,
    with all pairwise differences distinct across all rows, and max element <= max_scope.
    """
    rng = random.Random(seed_offset * 12345 + 678)
    used_diffs = set()
    rows = []
    k_plus_1 = k + 1
    
    def build_row(pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k_plus_1:
            return list(row_so_far)
        
        # Find valid candidates
        valid_candidates = []
        # Prune upper bound
        max_x = max_scope - (k_plus_1 - pos - 1)
        if max_x <= current_mark:
            return None
            
        for x in range(current_mark + 1, max_x + 1):
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            valid_candidates.append(x)
        
        if not valid_candidates:
            return None
        
        # Randomize order, bias towards smaller values
        n_smallest = min(10, len(valid_candidates))
        smallest = valid_candidates[:n_smallest]
        
        if len(valid_candidates) > n_smallest and rng.random() < 0.2:
            extra = rng.sample(valid_candidates[n_smallest:], min(3, len(valid_candidates) - n_smallest))
            candidates_to_try = smallest + extra
        else:
            candidates_to_try = smallest
        
        rng.shuffle(candidates_to_try)
        
        for x in candidates_to_try:
            new_diffs = {x - a for a in row_so_far}
            
            used_diffs.update(new_diffs)
            res = build_row(pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            used_diffs.difference_update(new_diffs)
        
        return None
    
    for i in range(n_rows):
        row = build_row(1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        used_diffs.update({row[j] - row[i] for i in range(k_plus_1) for j in range(i + 1, k_plus_1)})
    
    return rows
