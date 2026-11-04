import time
import random
import copy

def construct(n_rows, k):
    deadline = time.time() + 145
    
    def get_diffs(row):
        diffs = set()
        n = len(row)
        for i in range(n):
            for j in range(i+1, n):
                diffs.add(row[j] - row[i])
        return diffs
    
    def is_valid(rows):
        all_diffs = set()
        for row in rows:
            for d in get_diffs(row):
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True
    
    def max_entry(rows):
        return max(max(row) for row in rows)
    
    def greedy_init():
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
    
    rows = greedy_init()
    best_rows = copy.deepcopy(rows)
    best_score = max_entry(rows)
    
    # Precompute initial diff sets
    diff_sets = [get_diffs(row) for row in rows]
    
    # Iterative improvement
    for iteration in range(200):
        if time.time() > deadline - 5:
            break
        
        r = random.randint(0, n_rows - 1)
        row = rows[r]
        
        # Compute diffs used by other rows
        other_diffs = set()
        for i in range(n_rows):
            if i != r:
                other_diffs |= diff_sets[i]
        
        # Try to find a conflict-free replacement for row r
        best_new_row = None
        best_max = float('inf')
        
        for _ in range(80):
            new_row = list(row)
            # Perturb 1-2 positions
            for _ in range(random.randint(1, 2)):
                pos = random.randint(1, k)
                new_row[pos] += random.randint(-6, 6)
            
            # Validate strictly increasing
            if new_row[0] != 0:
                continue
            valid = True
            for i in range(1, len(new_row)):
                if new_row[i] <= new_row[i-1]:
                    valid = False
                    break
            if not valid:
                continue
            
            new_diffs = get_diffs(new_row)
            if new_diffs & other_diffs:
                continue
            
            mx = max(new_row)
            if mx < best_max:
                best_max = mx
                best_new_row = new_row
        
        if best_new_row is not None:
            rows[r] = best_new_row
            diff_sets[r] = get_diffs(best_new_row)
            score = max_entry(rows)
            if score < best_score:
                best_score = score
                best_rows = copy.deepcopy(rows)
    
    return best_rows
