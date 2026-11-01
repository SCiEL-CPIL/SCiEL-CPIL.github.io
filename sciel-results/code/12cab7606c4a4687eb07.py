import time
import random
import math
from collections import defaultdict

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    def get_scope(rows):
        return max(max(r) for r in rows)
    
    def validate(rows):
        """Check if rows are valid."""
        used = set()
        for r in rows:
            if len(r) != row_len:
                return False
            if r[0] != 0:
                return False
            for i in range(len(r)):
                if i > 0 and r[i] <= r[i-1]:
                    return False
                for j in range(i):
                    d = r[i] - r[j]
                    if d in used:
                        return False
                    used.add(d)
        return True
    
    def count_conflicts(rows):
        """Count difference collisions."""
        diff_counts = defaultdict(int)
        for r in rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] += 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts
    
    def build_row_with_cp(used_diffs, max_scope, rng):
        """
        Build a single row using constraint propagation with backtracking.
        Tries to find a row of row_len elements starting at 0 with all diffs
        not in used_diffs, keeping max element <= max_scope.
        Returns the row or None if failed.
        """
        row = [0]
        row_diffs = set()
        
        def backtrack(pos, current_max):
            if pos == row_len:
                return True
            
            last = row[-1]
            # Try values from last+1 to max_scope
            # Use a heuristic: try values that create "spread" differences
            candidates = []
            for x in range(last + 1, max_scope + 1):
                new_diffs = set()
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in row_diffs:
                        valid = False
                        break
                    new_diffs.add(d)
                
                if valid:
                    candidates.append((x, new_diffs))
            
            if not candidates:
                return False
            
            # Shuffle candidates for randomization, but bias toward smaller values
            # to keep scope tight
            if rng.random() < 0.3:
                # Random order
                rng.shuffle(candidates)
            else:
                # Sort by value (smaller first) with some randomization
                candidates.sort(key=lambda t: t[0] + rng.random() * 5)
            
            for x, new_diffs in candidates:
                row.append(x)
                old_diffs = set(row_diffs)
                row_diffs.update(new_diffs)
                
                if backtrack(pos + 1, max(row)):
                    return True
                
                row.pop()
                row_diffs = old_diffs
            
            return False
        
        if backtrack(1, 0):
            return row
        return None
    
    def greedy_init(rand_prob=0.1, window=20):
        """Randomized greedy initialization."""
        used_diffs = set()
        rows = []
        
        for _ in range(n_rows):
            row = [0]
            while len(row) < row_len:
                last = row[-1]
                
                candidates = []
                x = last + 1
                limit = 300
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
    
    def targeted_descent(rows, time_limit):
        """
        Targeted descent: repeatedly try to reduce the global maximum.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        while time.time() < start + time_limit:
            # Find global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            if max_c_idx == 0:
                break
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Build set of all diffs
            all_diffs = set()
            for r in best_rows:
                for i in range(row_len):
                    for j in range(i+1, row_len):
                        all_diffs.add(r[j] - r[i])
            
            # Diffs to remove
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, row_len):
                remove_set.add(row[j] - current_val)
            
            fixed_diffs = all_diffs - remove_set
            
            found = False
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                for j in range(max_c_idx + 1, row_len):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, row_len)]
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                row[max_c_idx] = new_val
                found = True
                break
            
            if not found:
                # Random perturbation to escape local minimum
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx-1]
                next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                
                all_diffs = set()
                for r in best_rows:
                    for i in range(row_len):
                        for j in range(i+1, row_len):
                            all_diffs.add(r[j] - r[i])
                
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, row_len):
                    remove_set.add(row[j] - curr_val)
                
                fixed_diffs = all_diffs - remove_set
                
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                
                for _ in range(30):
                    if low > high:
                        break
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    
                    valid = True
                    for i in range(c_idx):
                        d = cand - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    for j in range(c_idx + 1, row_len):
                        d = row[j] - cand
                        if d in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    new_left_diffs = [cand - row[i] for i in range(c_idx)]
                    new_right_diffs = [row[j] - cand for j in range(c_idx + 1, row_len)]
                    all_new = new_left_diffs + new_right_diffs
                    if len(set(all_new)) != len(all_new):
                        continue
                    
                    row[c_idx] = cand
                    break
        
        return best_rows, get_scope(best_rows)
    
    def sa_refine(rows, time_limit):
        """Simulated annealing refinement."""
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        current_conflicts = count_conflicts(current_rows)
        
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        best_conflicts = current_conflicts
        
        T = 10.0
        T_min = 0.01
        cooling = 0.995
        start = time.time()
        
        while time.time() < start + time_limit:
            if T < T_min:
                break
            
            r_idx = random.randint(0, n_rows - 1)
            pos = random.randint(1, k)
            row = current_rows[r_idx]
            old_val = row[pos]
            
            lower_bound = row[pos - 1] + 1
            if pos < k:
                upper_bound = row[pos + 1] - 1
            else:
                upper_bound = current_scope + 10
            
            if lower_bound > upper_bound:
                continue
            
            v = random.randint(lower_bound, upper_bound)
            if v == old_val:
                continue
            
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
            else:
                row[pos] = old_val
            
            T *= cooling
        
        return best_rows, best_scope
    
    def rebuild_row(used_other, max_val, rng_seed=None):
        """Rebuild a single row given differences already used by other rows."""
        if rng_seed is not None:
            local_rng = random.Random(rng_seed)
        else:
            local_rng = random
        
        row = [0]
        row_diffs = set()
        x = 0
        while len(row) < row_len:
            found = False
            search_end = max_val if max_val else 300
            for trial_x in range(x + 1, search_end + 1):
                new = {trial_x - a for a in row}
                if not (new & used_other) and not (new & row_diffs):
                    if local_rng.random() < 0.1 and trial_x < 200:
                        continue
                    row.append(trial_x)
                    row_diffs |= new
                    found = True
                    break
            if not found:
                return None
            x = row[-1]
        return row
    
    def local_polish(rows, max_passes=4):
        """Reduce scope by locally re-optimizing each row."""
        rows = [list(r) for r in rows]
        improved = True
        pass_count = 0
        while improved and time.time() < deadline - 10 and pass_count < max_passes:
            improved = False
            pass_count += 1
            order = sorted(range(n_rows), key=lambda i: -max(rows[i]))
            for i in order:
                if time.time() > deadline - 10:
                    break
                
                used_other = set()
                for j in range(n_rows):
                    if j != i:
                        for a in rows[j]:
                            for b in rows[j]:
                                if a < b:
                                    used_other.add(b - a)
                
                best_row = list(rows[i])
                best_max = max(rows[i])
                
                new_row = rebuild_row(used_other, best_max - 1)
                if new_row is not None and max(new_row) < best_max:
                    best_row = new_row
                    best_max = max(new_row)
                
                for attempt in range(8):
                    jitter_seed = random.randint(0, 1000000)
                    new_row = rebuild_row(used_other, best_max - 1, rng_seed=jitter_seed)
                    if new_row is not None and max(new_row) < best_max:
                        best_row = new_row
                        best_max = max(new_row)
                
                if max(best_row) < max(rows[i]):
                    rows[i] = best_row
                    improved = True
        return rows
    
    def kick(rows, num_kicks=1):
        """Perturb by rebuilding random rows."""
        rows = [list(r) for r in rows]
        indices = random.sample(range(n_rows), min(num_kicks, n_rows))
        for idx in indices:
            used_other = set()
            for j in range(n_rows):
                if j != idx:
                    for a in rows[j]:
                        for b in rows[j]:
                            if a < b:
                                used_other.add(b - a)
            
            success = False
            for attempt in range(10):
                max_val = get_scope(rows) + 20
                new_row = rebuild_row(used_other, max_val)
                if new_row is not None:
                    rows[idx] = new_row
                    success = True
                    break
            if not success:
                new_row = rebuild_row(used_other, get_scope(rows) + 30)
                if new_row is not None:
                    rows[idx] = new_row
        return rows
    
    # --- MAIN SEARCH LOOP ---
    
    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')
    
    # Multi-start with CP-based construction
    num_restarts = 10
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Vary initialization strategy
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)
        
        # Phase 1: Targeted Descent
        t1_rows, t1_scope = targeted_descent(rows, 4.0)
        t1_conflicts = count_conflicts(t1_rows)
        
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = t1_rows
        
        # Phase 2: SA refinement
        if time.time() > deadline - 15:
            break
        
        t2_rows, t2_scope = sa_refine(t1_rows, 4.0)
        t2_conflicts = count_conflicts(t2_rows)
        
        if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
            best_conflicts = t2_conflicts
            best_scope = t2_scope
            best_rows = t2_rows
        
        # Early exit if very good
        if best_conflicts == 0 and best_scope < 120:
            break
    
    # Phase 3: ILS with Row Reconstruction
    if best_rows is not None:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 10:
            num_il_iters = int(remaining_time / 2.5)
            num_il_iters = max(1, num_il_iters)
            
            for it in range(num_il_iters):
                if time.time() > deadline - 5:
                    break
                
                cur = kick(best_rows, num_kicks=random.choice([1, 2]))
                pol_scope = get_scope(cur)
                pol_conflicts = count_conflicts(cur)
                
                polished = local_polish(cur, max_passes=2)
                pol_scope = get_scope(polished)
                pol_conflicts = count_conflicts(polished)
                
                if pol_conflicts < best_conflicts or (pol_conflicts == best_conflicts and pol_scope < best_scope):
                    best_conflicts = pol_conflicts
                    best_scope = pol_scope
                    best_rows = polished
    
    # Final polish
    if best_rows is not None:
        remaining_time = deadline - time.time() - 2.0
        if remaining_time > 5:
            half_time = remaining_time / 2
            
            f1_rows, f1_scope = targeted_descent(best_rows, half_time)
            f1_conflicts = count_conflicts(f1_rows)
            
            if f1_conflicts < best_conflicts or (f1_conflicts == best_conflicts and f1_scope < best_scope):
                best_conflicts = f1_conflicts
                best_scope = f1_scope
                best_rows = f1_rows
            
            if time.time() < deadline - 2.0:
                f2_rows, f2_scope = sa_refine(best_rows, deadline - time.time() - 2.0)
                f2_conflicts = count_conflicts(f2_rows)
                
                if f2_conflicts < best_conflicts or (f2_conflicts == best_conflicts and f2_scope < best_scope):
                    best_conflicts = f2_conflicts
                    best_scope = f2_scope
                    best_rows = f2_rows
    
    if best_rows is None:
        best_rows = greedy_init(0.1, 20)
    
    return best_rows
