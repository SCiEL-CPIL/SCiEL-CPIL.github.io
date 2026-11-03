import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # 1. Baseline Greedy Solution
    best_solution = greedy_construct(n_rows, k)
    if not best_solution:
        return []
    
    best_scope = max(max(row) for row in best_solution)
    
    # Calculate theoretical lower bound
    num_diffs = n_rows * k * (k + 1) // 2
    lower_bound = num_diffs
    
    if best_scope <= lower_bound:
        return best_solution

    # 2. Local search over row structures with difference tracking
    # We'll try to improve by swapping elements within rows to reduce max scope
    
    current_solution = [row[:] for row in best_solution]
    current_scope = best_scope
    used_diffs = set()
    for row in current_solution:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    # Try to reduce scope by moving larger values to smaller positions
    # Use a local search approach: try to decrease the largest element in each row
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # For each row, try to reduce the largest element
        for r in range(n_rows):
            if time.time() > deadline - 5:
                break
            
            row = current_solution[r]
            if len(row) <= 1:
                continue
            
            max_val = row[-1]
            if max_val <= lower_bound:
                continue
            
            # Try to replace max_val with a smaller value
            # The new value must be > row[-2] and all new diffs must be unused
            old_diffs = {max_val - a for a in row[:-1]}
            used_diffs -= old_diffs
            
            # Try values from max_val - 1 down to row[-2] + 1
            found = False
            for new_val in range(max_val - 1, row[-2], -1):
                new_diffs = {new_val - a for a in row[:-1]}
                if new_diffs & used_diffs:
                    continue
                
                # Check if this creates a valid row (strictly increasing)
                if new_val <= row[-2]:
                    break
                
                # Accept this change
                row[-1] = new_val
                used_diffs |= new_diffs
                found = True
                
                new_scope = max(max(row) for row in current_solution)
                if new_scope < current_scope:
                    current_scope = new_scope
                    best_scope = new_scope
                    best_solution = [row[:] for row in current_solution]
                
                improved = True
                break
            
            if not found:
                # Restore old state
                used_diffs |= old_diffs
                row[-1] = max_val
        
        # Also try internal swaps: reduce non-last elements
        for r in range(n_rows):
            if time.time() > deadline - 5:
                break
            
            row = current_solution[r]
            if len(row) <= 2:
                continue
            
            # Try to reduce each element (except first)
            for pos in range(1, len(row) - 1):
                if time.time() > deadline - 5:
                    break
                
                val = row[pos]
                prev = row[pos - 1]
                nxt = row[pos + 1] if pos + 1 < len(row) else None
                
                # Diffs involving this element
                old_diffs_in = {val - a for a in row[:pos]}
                old_diffs_out = {b - val for b in row[pos+1:]}
                old_diffs = old_diffs_in | old_diffs_out
                used_diffs -= old_diffs
                
                # Try smaller values
                found = False
                for new_val in range(val - 1, prev, -1):
                    new_diffs_in = {new_val - a for a in row[:pos]}
                    new_diffs_out = {b - new_val for b in row[pos+1:]}
                    new_diffs = new_diffs_in | new_diffs_out
                    
                    if new_diffs & used_diffs:
                        continue
                    
                    # Check strict increase
                    if new_val <= prev:
                        break
                    if nxt is not None and new_val >= nxt:
                        continue
                    
                    # Accept
                    row[pos] = new_val
                    used_diffs |= new_diffs
                    found = True
                    improved = True
                    break
                
                if not found:
                    used_diffs |= old_diffs
                    row[pos] = val
        
        # Try to compact: if there are gaps, shift elements down
        for r in range(n_rows):
            if time.time() > deadline - 5:
                break
            
            row = current_solution[r]
            if len(row) <= 1:
                continue
            
            # Try to shift the whole row down by 1 if possible
            if row[1] > 1:
                # Check if shifting all elements (except 0) down by 1 works
                new_row = [0] + [x - 1 for x in row[1:]]
                new_diffs = set()
                for i in range(len(new_row)):
                    for j in range(i+1, len(new_row)):
                        new_diffs.add(new_row[j] - new_row[i])
                
                # Remove old diffs
                old_diffs = set()
                for i in range(len(row)):
                    for j in range(i+1, len(row)):
                        old_diffs.add(row[j] - row[i])
                
                used_diffs -= old_diffs
                if not (new_diffs & used_diffs):
                    current_solution[r] = new_row
                    used_diffs |= new_diffs
                    improved = True
    
    # Final polish: try randomized local search
    if time.time() < deadline - 10:
        solution = local_search_polish(best_solution, n_rows, k, deadline)
        if solution:
            new_scope = max(max(row) for row in solution)
            if new_scope < best_scope:
                best_solution = solution
                best_scope = new_scope

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

def local_search_polish(solution, n_rows, k, deadline):
    """Randomized local search to improve scope"""
    current = [row[:] for row in solution]
    current_scope = max(max(row) for row in current)
    
    used_diffs = set()
    for row in current:
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                used_diffs.add(row[j] - row[i])
    
    best = [row[:] for row in current]
    best_scope = current_scope
    
    for iteration in range(1000):
        if time.time() > deadline - 5:
            break
        
        # Random perturbation: pick a random row and random position, try to change it
        r = random.randint(0, n_rows - 1)
        row = current[r]
        
        if len(row) <= 1:
            continue
        
        pos = random.randint(1, len(row) - 1)
        val = row[pos]
        prev = row[pos - 1]
        nxt = row[pos + 1] if pos + 1 < len(row) else None
        
        # Remove old diffs
        old_diffs_in = {val - a for a in row[:pos]}
        old_diffs_out = {b - val for b in row[pos+1:]}
        old_diffs = old_diffs_in | old_diffs_out
        used_diffs -= old_diffs
        
        # Try a random new value
        new_val = val + random.randint(-5, 5)
        valid = False
        
        if new_val > prev:
            if nxt is None or new_val < nxt:
                new_diffs_in = {new_val - a for a in row[:pos]}
                new_diffs_out = {b - new_val for b in row[pos+1:]}
                new_diffs = new_diffs_in | new_diffs_out
                
                if not (new_diffs & used_diffs):
                    row[pos] = new_val
                    used_diffs |= new_diffs
                    valid = True
        
        if valid:
            new_scope = max(max(row) for row in current)
            if new_scope < best_scope:
                best_scope = new_scope
                best = [row[:] for row in current]
        else:
            # Restore
            used_diffs |= old_diffs
            row[pos] = val
    
    return best

def randomized_search(n_rows, k, max_scope, deadline):
    """Backtracking search with bitset optimization"""
    used_diffs = set()
    rows = []
    
    def build_row(row_idx, pos, current_mark, row_so_far):
        if time.time() > deadline:
            return None
            
        if pos == k + 1:
            return row_so_far
        
        # Find valid candidates for the next mark
        valid_candidates = []
        
        for x in range(current_mark + 1, max_scope + 1):
            new_diffs = {x - a for a in row_so_far}
            if new_diffs & used_diffs:
                continue
            
            valid_candidates.append(x)
            
            # Early termination: if we have enough candidates, stop
            if len(valid_candidates) >= 20:
                break
        
        if not valid_candidates:
            return None
        
        # Try candidates in order (smallest first for compactness)
        for x in valid_candidates[:min(5, len(valid_candidates))]:
            new_diffs = {x - a for a in row_so_far}
            
            used_diffs.update(new_diffs)
            
            res = build_row(row_idx, pos + 1, x, row_so_far + [x])
            if res is not None:
                return res
            
            used_diffs.difference_update(new_diffs)
            
        return None

    for i in range(n_rows):
        row = build_row(i, 1, 0, [0])
        if row is None:
            return None
        rows.append(row)
        
    return rows
