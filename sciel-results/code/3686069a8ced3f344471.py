import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before the 150s kill

    # Problem parameters
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def greedy_init(rand_prob=0.1, window=20):
        """
        Generates a valid configuration using a randomized greedy approach.
        """
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 200 
                while x <= limit and len(candidates) < window:
                    valid = True
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                    x += 1
                
                if not candidates:
                    while True:
                        x += 1
                        valid = True
                        for a in row:
                            if (x - a) in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                
                if random.random() < rand_prob and len(candidates) > 1:
                    x = random.choice(candidates)
                else:
                    x = candidates[0]
                
                row.append(x)
                for a in row[:-1]:
                    used_diffs.add(x - a)
                
            rows.append(row)
            
        return rows

    def count_conflicts(rows):
        """
        Counts the number of distinct difference collisions.
        """
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def local_search_cdvr(rows, time_limit):
        """
        Optimizes rows using Conflict-Directed Value Removal (CDVR) style local search.
        """
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        
        stagnation_counter = 0
        last_conflict = current_conflicts
        
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        while time.time() < start_time + time_limit:
            if T < T_min:
                if stagnation_counter > 1000:
                    T = 5.0
                    stagnation_counter = 0
                else:
                    break
            
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            pos = random.randint(1, k)
            
            old_val = row[pos]
            
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 10
            
            if lower_bound > upper_bound:
                continue
                
            best_candidate = None
            best_candidate_score = float('inf')
            
            num_samples = 15
            candidates = []
            
            if current_conflicts > 0:
                for _ in range(num_samples):
                    v = random.randint(lower_bound, upper_bound)
                    candidates.append(v)
            else:
                for _ in range(num_samples):
                    if random.random() < 0.6:
                        v = lower_bound + random.randint(0, min(upper_bound - lower_bound, 20))
                    else:
                        v = random.randint(lower_bound, upper_bound)
                    candidates.append(v)
            
            candidates = list(set(candidates))
            if old_val in candidates:
                candidates.remove(old_val)
            
            if not candidates:
                continue
            
            best_v = None
            best_local_score = float('inf')
            
            current_diffs = set()
            for r in current_rows:
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        current_diffs.add(r[j] - r[i])
            
            for i in range(row_len):
                if i == pos:
                    continue
                d1 = abs(old_val - row[i])
                if d1 in current_diffs:
                    current_diffs.discard(d1)
            
            for v in candidates:
                new_diffs = set()
                for i in range(row_len):
                    if i == pos:
                        continue
                    d = abs(v - row[i])
                    new_diffs.add(d)
                
                local_conflicts = sum(1 for d in new_diffs if d in current_diffs)
                
                if current_conflicts == 0:
                    score = local_conflicts * 100 + v
                else:
                    score = local_conflicts
                
                if score < best_local_score:
                    best_local_score = score
                    best_v = v
            
            if best_v is None or best_v == old_val:
                continue
            
            v = best_v
            
            old_val_saved = old_val
            row[pos] = v
            
            new_conflicts = count_conflicts(current_rows)
            new_scope = get_scope(current_rows)
            
            delta_conflict = new_conflicts - current_conflicts
            delta_scope = new_scope - current_scope
            
            accept = False
            if new_conflicts < current_conflicts:
                accept = True
            elif new_conflicts == current_conflicts:
                if new_scope < current_scope:
                    accept = True
                else:
                    if random.random() < math.exp(-delta_scope / T):
                        accept = True
            else:
                if random.random() < math.exp(-delta_conflict / T):
                    accept = True
            
            if accept:
                current_conflicts = new_conflicts
                current_scope = new_scope
                
                if current_conflicts < best_conflicts or (current_conflicts == best_conflicts and current_scope < best_scope):
                    best_conflicts = current_conflicts
                    best_scope = current_scope
                    best_rows = [r[:] for r in current_rows]
                    
                if current_conflicts < last_conflict:
                    last_conflict = current_conflicts
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                    
            else:
                row[pos] = old_val_saved
                stagnation_counter += 1
                
            T *= cooling
            
        return best_rows, best_scope

    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')
    
    num_restarts = 12
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
            
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        
        cdvr_time_limit = 10.0
        optimized_rows, opt_scope = local_search_cdvr(rows, cdvr_time_limit)
        
        opt_conflicts = count_conflicts(optimized_rows)
        
        if opt_conflicts < best_conflicts or (opt_conflicts == best_conflicts and opt_scope < best_scope):
            best_conflicts = opt_conflicts
            best_scope = opt_scope
            best_rows = optimized_rows
            
        if best_conflicts == 0 and best_scope < 125:
            break
            
    if best_rows is not None:
        remaining_time = deadline - time.time()
        if remaining_time > 5:
            final_rows, final_scope = local_search_cdvr(best_rows, remaining_time - 2.0)
            
            final_conflicts = count_conflicts(final_rows)
            
            if final_conflicts < best_conflicts or (final_conflicts == best_conflicts and final_scope < best_scope):
                best_conflicts = final_conflicts
                best_scope = final_scope
                best_rows = final_rows

    if best_rows is None:
        best_rows = greedy_init(0.1, 20)
        
    return best_rows
