"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    # Greedy construction: build rows one by one, each row greedily picks
    # the smallest next element whose differences don't conflict with used ones.
    def greedy_construct():
        used = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 1
            while len(row) < k + 1:
                new_diffs = {x - a for a in row}
                if not new_diffs & used:
                    row.append(x)
                    used |= new_diffs
                x += 1
            rows.append(row)
        return rows
    
    def get_scope(rows):
        return max(max(row) for row in rows)
    
    def get_all_diffs(rows):
        diffs = []
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    diffs.append(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        diffs = get_all_diffs(rows)
        return len(diffs) == len(set(diffs))
    
    # Local search: try to reduce scope by moving elements to smaller values
    def local_search(rows):
        best = copy.deepcopy(rows)
        best_scope = get_scope(best)
        improved = True
        while improved:
            improved = False
            for r in range(n_rows):
                for pos in range(1, len(best[r])):  # skip position 0 (must be 0)
                    old_val = best[r][pos]
                    # Try moving to a smaller value
                    for new_val in range(old_val - 1, 0, -1):
                        if new_val in best[r]:
                            continue
                        # Check if new_val is greater than previous element
                        if pos > 0 and new_val <= best[r][pos-1]:
                            continue
                        # Check if new_val is less than next element
                        if pos < len(best[r]) - 1 and new_val >= best[r][pos+1]:
                            continue
                        # Temporarily set and check validity
                        best[r][pos] = new_val
                        if is_valid(best):
                            new_scope = get_scope(best)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                improved = True
                                break
                            else:
                                best[r][pos] = old_val
                        else:
                            best[r][pos] = old_val
        return best
    
    # Kick perturbation: randomly modify a few elements
    def kick(rows, strength=3):
        rows = copy.deepcopy(rows)
        for _ in range(strength):
            r = random.randint(0, n_rows - 1)
            pos = random.randint(1, len(rows[r]) - 1)
            # Try a random new value
            max_val = max(max(row) for row in rows)
            new_val = random.randint(1, max_val + 10)
            # Ensure ordering
            if pos > 0 and new_val <= rows[r][pos-1]:
                new_val = rows[r][pos-1] + 1
            if pos < len(rows[r]) - 1 and new_val >= rows[r][pos+1]:
                new_val = rows[r][pos+1] - 1
            if new_val > 0:
                rows[r][pos] = new_val
        return rows
    
    # Try to repair invalid rows back to valid
    def repair(rows):
        # Rebuild any row that has issues using greedy within constraints
        used = set()
        for row in rows:
            for i in range(len(row)):
                for j in range(i+1, len(row)):
                    used.add(row[j] - row[i])
        
        for r in range(n_rows):
            row = rows[r]
            # Check if this row is valid internally (strictly increasing, starts at 0)
            if row[0] != 0 or any(row[i] >= row[i+1] for i in range(len(row)-1)):
                # Rebuild this row greedily
                row = [0]
                x = 1
                while len(row) < k + 1:
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        row.append(x)
                        used |= new_diffs
                    x += 1
                rows[r] = row
        return rows
    
    # ILS main loop
    current = greedy_construct()
    best = copy.deepcopy(current)
    best_scope = get_scope(best)
    
    while time.time() < deadline:
        # Local search
        current = local_search(current)
        if get_scope(current) < best_scope:
            best = copy.deepcopy(current)
            best_scope = get_scope(best)
        
        # Check deadline before kick
        if time.time() > deadline - 5:
            break
        
        # Kick
        current = kick(current, strength=random.randint(2, 5))
        # Repair if needed
        if not is_valid(current):
            current = repair(current)
            if not is_valid(current):
                # Fallback to greedy
                current = greedy_construct()
                continue
        
        # Local search after kick
        current = local_search(current)
        if get_scope(current) < best_scope:
            best = copy.deepcopy(current)
            best_scope = get_scope(best)
    
    return best
# EVOLVE-BLOCK-END
