import time
import random
import math
import copy

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115
    
    def greedy_init():
        used_diffs = set()
        rows = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                diffs = {x - a for a in row}
                if not (diffs & used_diffs):
                    row.append(x)
                    used_diffs.update(diffs)
            rows.append(row)
        return rows

    curr_rows = greedy_init()
    
    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i+1, len(r)):
                    d = r[j] - r[i]
                    counts[d] = counts.get(d, 0) + 1
        return counts

    def get_penalty(counts):
        pen = 0
        for c in counts.values():
            if c > 1:
                pen += c - 1
        return pen

    def get_M(rows):
        return max(max(r) for r in rows)

    curr_counts = get_diff_counts(curr_rows)
    curr_penalty = get_penalty(curr_counts)
    curr_M = get_M(curr_rows)
    
    best_rows = copy.deepcopy(curr_rows)
    best_M = curr_M
    best_penalty = curr_penalty
    
    # Track row maxes
    row_maxes = [max(r) for r in curr_rows]
    
    # SA Parameters
    T = 50.0
    cooling = 0.999
    lambda_pen = 100.0
    
    curr_energy = curr_M + lambda_pen * curr_penalty
    
    # Helper to check if a specific value change is valid (for targeted moves)
    def try_move(r_idx, j_idx, new_val):
        """
        Attempts to change curr_rows[r_idx][j_idx] to new_val.
        Returns (accepted, new_M, new_penalty, new_energy)
        Modifies curr_rows, curr_counts, curr_penalty, curr_M, row_maxes in place if accepted.
        Reverts if rejected.
        """
        row = curr_rows[r_idx]
        old_val = row[j_idx]
        
        if new_val == old_val:
            return False, curr_M, curr_penalty, curr_energy
            
        # Check bounds
        low = row[j_idx - 1] + 1
        high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 10 # Allow some expansion
        
        if new_val < low or new_val > high:
            return False, curr_M, curr_penalty, curr_energy
            
        # 1. Remove old diffs
        for i in range(j_idx):
            d = old_val - row[i]
            c = curr_counts[d]
            if c > 1: curr_penalty -= 1
            curr_counts[d] -= 1
            if curr_counts[d] == 0: del curr_counts[d]
                
        for h in range(j_idx + 1, k + 1):
            d = row[h] - old_val
            c = curr_counts[d]
            if c > 1: curr_penalty -= 1
            curr_counts[d] -= 1
            if curr_counts[d] == 0: del curr_counts[d]
        
        # 2. Set new value
        row[j_idx] = new_val
        
        # 3. Add new diffs
        for i in range(j_idx):
            d = new_val - row[i]
            c = curr_counts.get(d, 0)
            if c >= 1: curr_penalty += 1
            curr_counts[d] = c + 1
            
        for h in range(j_idx + 1, k + 1):
            d = row[h] - new_val
            c = curr_counts.get(d, 0)
            if c >= 1: curr_penalty += 1
            curr_counts[d] = c + 1
            
        # 4. Calculate new M
        old_row_max = row_maxes[r_idx]
        new_row_max = max(row)
        
        if n_rows == 1:
            new_M = new_row_max
        else:
            other_maxes = [row_maxes[i] for i in range(n_rows) if i != r_idx]
            other_max = max(other_maxes) if other_maxes else 0
            new_M = max(new_row_max, other_max)
            
        new_energy = new_M + lambda_pen * curr_penalty
        delta_E = new_energy - curr_energy
        
        # Acceptance
        if delta_E < 0 or (T > 0.01 and random.random() < math.exp(-delta_E / T)):
            curr_M = new_M
            row_maxes[r_idx] = new_row_max
            curr_energy = new_energy
            return True, new_M, curr_penalty, new_energy
        else:
            # Revert
            row[j_idx] = old_val
            
            # Remove new diffs
            for i in range(j_idx):
                d = new_val - row[i]
                c = curr_counts[d]
                if c > 1: curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0: del curr_counts[d]
                    
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = curr_counts[d]
                if c > 1: curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0: del curr_counts[d]
            
            # Add old diffs
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts.get(d, 0)
                if c >= 1: curr_penalty += 1
                curr_counts[d] = c + 1
                
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts.get(d, 0)
                if c >= 1: curr_penalty += 1
                curr_counts[d] = c + 1
                
            return False, curr_M, curr_penalty, curr_energy

    # Main SA Loop
    while time.time() < deadline:
        # Random Move
        r_idx = random.randint(0, n_rows - 1)
        row = curr_rows[r_idx]
        j_idx = random.randint(1, k)
        
        low = row[j_idx - 1] + 1
        high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 5
        
        if high < low:
            continue
            
        new_val = random.randint(low, high)
        
        accepted, new_M, new_pen, new_en = try_move(r_idx, j_idx, new_val)
        
        if accepted:
            if new_pen == 0:
                if new_M < best_M:
                    best_M = new_M
                    best_rows = copy.deepcopy(curr_rows)
                    best_penalty = 0
        
        # Cool down
        T *= cooling
        if T < 0.5:
            T = 0.5

    # Final Polish: Try to reduce M specifically
    if best_rows and best_penalty == 0:
        polish_deadline = start_time + 118
        # We work on a copy to be safe, or just mutate best_rows carefully
        # Since best_rows is valid, we can try to reduce its M
        
        # Re-initialize state for best_rows for fast checks?
        # Actually, let's just use the slow check for the few seconds remaining,
        # or re-initialize the fast structures on best_rows.
        
        b_rows = copy.deepcopy(best_rows)
        b_counts = get_diff_counts(b_rows)
        b_pen = get_penalty(b_counts)
        b_M = best_M
        b_row_maxes = [max(r) for r in b_rows]
        
        # If b_pen > 0, something went wrong, skip polish
        if b_pen > 0:
            return best_rows
            
        while time.time() < polish_deadline:
            r_idx = random.randint(0, n_rows - 1)
            row = b_rows[r_idx]
            j_idx = random.randint(1, k)
            
            low = row[j_idx - 1] + 1
            high = row[j_idx + 1] - 1 if j_idx < k else b_M
            
            if high < low:
                continue
                
            # Targeted reduction: try values smaller than current if possible
            # Or just random small changes
            if j_idx == k:
                # Try to shrink the last element
                candidates = list(range(low, row[j_idx]))
                if not candidates: continue
                new_val = random.choice(candidates)
            else:
                # Try small perturbations
                new_val = row[j_idx] + random.choice([-1, 1])
                if new_val < low or new_val > high: continue
            
            if new_val == row[j_idx]:
                continue
                
            # Fast check using b_counts
            old_val = row[j_idx]
            
            # Remove old
            for i in range(j_idx):
                d = old_val - row[i]
                c = b_counts[d]
                if c > 1: b_pen -= 1
                b_counts[d] -= 1
                if b_counts[d] == 0: del b_counts[d]
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = b_counts[d]
                if c > 1: b_pen -= 1
                b_counts[d] -= 1
                if b_counts[d] == 0: del b_counts[d]
            
            # Add new
            row[j_idx] = new_val
            for i in range(j_idx):
                d = new_val - row[i]
                c = b_counts.get(d, 0)
                if c >= 1: b_pen += 1
                b_counts[d] = c + 1
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = b_counts.get(d, 0)
                if c >= 1: b_pen += 1
                b_counts[d] = c + 1
                
            new_row_max = max(row)
            if n_rows == 1:
                new_M = new_row_max
            else:
                other_max = max([b_row_maxes[i] for i in range(n_rows) if i != r_idx]) if n_rows > 1 else 0
                new_M = max(new_row_max, other_max)
                
            if b_pen == 0 and new_M < b_M:
                b_M = new_M
                b_row_maxes[r_idx] = new_row_max
                best_M = b_M
                best_rows = copy.deepcopy(b_rows)
            else:
                # Revert
                row[j_idx] = old_val
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = b_counts[d]
                    if c > 1: b_pen -= 1
                    b_counts[d] -= 1
                    if b_counts[d] == 0: del b_counts[d]
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = b_counts[d]
                    if c > 1: b_pen -= 1
                    b_counts[d] -= 1
                    if b_counts[d] == 0: del b_counts[d]
                    
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = b_counts.get(d, 0)
                    if c >= 1: b_pen += 1
                    b_counts[d] = c + 1
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    c = b_counts.get(d, 0)
                    if c >= 1: b_pen += 1
                    b_counts[d] = c + 1

    return best_rows
