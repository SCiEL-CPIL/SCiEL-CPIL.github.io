import time
import random
import copy
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0
    
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
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

    def get_diff_counts(rows):
        counts = {}
        for r in rows:
            for i in range(len(r)):
                for j in range(i + 1, len(r)):
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

    def sa_optimize(rows, deadline, phase="both"):
        """
        Run SA on the given rows.
        phase: 'conflicts' - minimize penalty only
               'M' - minimize M (penalty should be 0)
               'both' - minimize M + lambda*penalty
        """
        curr_rows = [r[:] for r in rows]
        curr_counts = get_diff_counts(curr_rows)
        curr_penalty = get_penalty(curr_counts)
        curr_M = get_M(curr_rows)
        row_maxes = [max(r) for r in curr_rows]
        
        best_rows = [r[:] for r in curr_rows]
        best_M = curr_M
        best_penalty = curr_penalty
        
        if phase == 'conflicts':
            T = 20.0
            cooling = 0.9995
            lambda_pen = 1000.0
        elif phase == 'M':
            T = 10.0
            cooling = 0.9998
            lambda_pen = 10000.0
        else:
            T = 30.0
            cooling = 0.9995
            lambda_pen = 50.0
        
        curr_energy = curr_M + lambda_pen * curr_penalty
        
        while time.time() < deadline:
            r_idx = random.randint(0, n_rows - 1)
            row = curr_rows[r_idx]
            j_idx = random.randint(1, k)
            old_val = row[j_idx]
            
            low = row[j_idx - 1] + 1
            high = row[j_idx + 1] - 1 if j_idx < k else curr_M + 8
            
            if high < low:
                continue
                
            new_val = random.randint(low, high)
            if new_val == old_val:
                continue
            
            # Remove old diffs
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts[d]
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] -= 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            
            row[j_idx] = new_val
            
            # Add new diffs
            for i in range(j_idx):
                d = new_val - row[i]
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
            for h in range(j_idx + 1, k + 1):
                d = row[h] - new_val
                c = curr_counts.get(d, 0)
                if c >= 1:
                    curr_penalty += 1
                curr_counts[d] = c + 1
            
            # Update M
            old_row_max = row_maxes[r_idx]
            new_row_max = max(row)
            if new_row_max > old_row_max:
                new_M = max(new_row_max, max(row_maxes[:r_idx] + row_maxes[r_idx+1:]) if n_rows > 1 else 0)
            else:
                other_max = max(row_maxes[:r_idx] + row_maxes[r_idx+1:]) if n_rows > 1 else 0
                new_M = max(new_row_max, other_max)
            
            new_energy = new_M + lambda_pen * curr_penalty
            delta_E = new_energy - curr_energy
            
            if delta_E < 0 or (T > 0.5 and random.random() < math.exp(-delta_E / T)):
                curr_energy = new_energy
                curr_M = new_M
                row_maxes[r_idx] = new_row_max
                
                # Track best
                if phase == 'conflicts':
                    if curr_penalty < best_penalty or (curr_penalty == best_penalty and curr_M < best_M):
                        best_penalty = curr_penalty
                        best_M = curr_M
                        best_rows = [r[:] for r in curr_rows]
                elif phase == 'M':
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = [r[:] for r in curr_rows]
                else:
                    if curr_penalty == 0 and curr_M < best_M:
                        best_M = curr_M
                        best_rows = [r[:] for r in curr_rows]
            else:
                row[j_idx] = old_val
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts[d]
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] -= 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                for i in range(j_idx):
                    d = old_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - old_val
                    c = curr_counts.get(d, 0)
                    if c >= 1:
                        curr_penalty += 1
                    curr_counts[d] = c + 1
            
            T *= cooling
            if T < 0.5:
                T = 0.5
        
        return best_rows, best_M, best_penalty

    # Multi-restart strategy
    best_overall_rows = None
    best_overall_M = float('inf')
    
    # Phase 1: Try to get to zero conflicts with different seeds
    phase1_deadline = start_time + 40.0
    n_restarts_phase1 = 3
    
    for restart in range(n_restarts_phase1):
        if time.time() > phase1_deadline:
            break
        init_rows = greedy_init(seed_offset=restart * 1000)
        phase1_sub_deadline = min(phase1_deadline, start_time + 40.0 / n_restarts_phase1 * (restart + 1))
        rows, M, penalty = sa_optimize(init_rows, phase1_sub_deadline, phase="conflicts")
        if penalty == 0 and M < best_overall_M:
            best_overall_M = M
            best_overall_rows = rows
    
    # Phase 2: If we have a valid solution, minimize M
    if best_overall_rows is not None:
        phase2_deadline = start_time + 112.0
        # Multiple SA runs to minimize M
        n_restarts_phase2 = 5
        for restart in range(n_restarts_phase2):
            if time.time() > phase2_deadline:
                break
            sub_deadline = min(phase2_deadline, start_time + 72.0 / n_restarts_phase2 * (restart + 1))
            rows, M, penalty = sa_optimize(best_overall_rows, sub_deadline, phase="M")
            if penalty == 0 and M < best_overall_M:
                best_overall_M = M
                best_overall_rows = rows
    
    # Phase 3: If no valid solution yet, try combined SA
    if best_overall_rows is None:
        phase3_deadline = start_time + 112.0
        init_rows = greedy_init(seed_offset=42)
        rows, M, penalty = sa_optimize(init_rows, phase3_deadline, phase="both")
        if penalty == 0:
            best_overall_M = M
            best_overall_rows = rows
        else:
            # Return the best we have (may be invalid, but better than nothing)
            best_overall_rows = rows
    
    # Final polish: try to reduce M further with targeted moves
    if best_overall_rows is not None:
        polish_deadline = start_time + 114.0
        curr_rows = [r[:] for r in best_overall_rows]
        curr_M = best_overall_M
        curr_counts = get_diff_counts(curr_rows)
        
        while time.time() < polish_deadline:
            r_idx = random.randint(0, n_rows - 1)
            row = curr_rows[r_idx]
            j_idx = random.randint(1, k)
            old_val = row[j_idx]
            
            low = row[j_idx - 1] + 1
            high = row[j_idx + 1] - 1 if j_idx < k else curr_M
            
            if high < low:
                continue
            
            # Try to decrease the value
            if j_idx == k:
                candidates = list(range(low, old_val))
            else:
                candidates = [v for v in [old_val - 2, old_val - 1] if low <= v <= high]
            
            if not candidates:
                continue
            
            new_val = random.choice(candidates)
            if new_val == old_val:
                continue
            
            # Check validity incrementally
            # Remove old diffs
            temp_counts = dict(curr_counts)
            for i in range(j_idx):
                d = old_val - row[i]
                temp_counts[d] -= 1
                if temp_counts[d] == 0:
                    del temp_counts[d]
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                temp_counts[d] -= 1
                if temp_counts[d] == 0:
                    del temp_counts[d]
            
            # Add new diffs
            valid = True
            for i in range(j_idx):
                d = new_val - row[i]
                if d in temp_counts:
                    valid = False
                    break
                temp_counts[d] = temp_counts.get(d, 0) + 1
            if valid:
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    if d in temp_counts:
                        valid = False
                        break
                    temp_counts[d] = temp_counts.get(d, 0) + 1
            
            if valid:
                new_row_max = max(row[:j_idx] + [new_val] + row[j_idx+1:])
                other_max = max(max(r) for r in curr_rows if r is not row) if n_rows > 1 else 0
                new_M = max(new_row_max, other_max)
                
                if new_M < curr_M:
                    row[j_idx] = new_val
                    curr_M = new_M
                    curr_counts = temp_counts
                    best_overall_M = curr_M
                    best_overall_rows = [r[:] for r in curr_rows]
    
    return best_overall_rows
