import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    # Hard kill is 150s. We finish with a safety margin.
    deadline = start_time + 143.0

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2

    # --- Core Utility Functions ---

    def get_scope(rows):
        return max(max(r) for r in rows)

    def count_conflicts(rows):
        diff_counts = {}
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        conflicts = 0
        for count in diff_counts.values():
            if count > 1:
                conflicts += count - 1
        return conflicts

    def row_diffs_set(row):
        s = set()
        n = len(row)
        for i in range(n):
            ri = row[i]
            for j in range(i + 1, n):
                s.add(row[j] - ri)
        return s

    def get_all_diffs(rows):
        diffs = set()
        for r in rows:
            for i in range(row_len):
                for j in range(i + 1, row_len):
                    diffs.add(r[j] - r[i])
        return diffs

    # --- Initialization ---

    def greedy_init(rand_prob=0.1, window=20):
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
                        if (x - a) in used_diffs:
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

    # --- Local Search: Targeted Descent ---
    # Tries to reduce the maximum element (or any element) by moving it down.

    def local_search_targeted(rows, time_limit):
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)

        while time.time() < start + time_limit:
            # Find global max
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
            
            all_diffs = get_all_diffs(best_rows)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, row_len):
                remove_set.add(row[j] - current_val)
            fixed_diffs = all_diffs - remove_set
            
            found = False
            # Try to move down
            for new_val in range(current_val - 1, prev_val, -1):
                valid = True
                for i in range(max_c_idx):
                    if (new_val - row[i]) in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                for j in range(max_c_idx + 1, row_len):
                    if (row[j] - new_val) in fixed_diffs:
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
                new_scope = get_scope(best_rows)
                if new_scope < best_scope:
                    best_scope = new_scope
                break
            
            if not found:
                # Try random other element moves if max is stuck
                r_idx = random.randint(0, n_rows - 1)
                c_idx = random.randint(1, row_len - 1)
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx - 1]
                next_v = best_rows[r_idx][c_idx + 1] if c_idx < row_len - 1 else curr_val + 100
                
                all_diffs = get_all_diffs(best_rows)
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, row_len):
                    remove_set.add(row[j] - curr_val)
                fixed_diffs = all_diffs - remove_set
                
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len - 1 else curr_val + 50
                if low > high:
                    continue
                
                # Try to move down or random within range
                for _ in range(50):
                    cand = random.randint(low, high)
                    if cand == curr_val:
                        continue
                    valid = True
                    for i in range(c_idx):
                        if (cand - row[i]) in fixed_diffs:
                            valid = False
                            break
                    if not valid:
                        continue
                    for j in range(c_idx + 1, row_len):
                        if (row[j] - cand) in fixed_diffs:
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

        return best_rows, best_scope

    # --- Local Search: Simulated Annealing ---
    # Uses conflicts and scope as energy.

    def local_search_sa(rows, time_limit):
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

    # --- High Precision Polish ---
    # Exhaustive single-element reduction.

    def high_precision_polish(rows, time_limit):
        rows = [list(r) for r in rows]
        start = time.time()
        improved = True
        pass_count = 0
        max_passes = 20
        while improved and pass_count < max_passes and (time.time() - start) < time_limit:
            improved = False
            pass_count += 1
            for r_idx in range(n_rows):
                if (time.time() - start) > time_limit:
                    break
                row = rows[r_idx]
                other_diffs = set()
                for j in range(n_rows):
                    if j != r_idx:
                        other_diffs |= row_diffs_set(rows[j])
                for c_idx in range(row_len - 1, 0, -1):
                    if (time.time() - start) > time_limit:
                        break
                    curr_val = row[c_idx]
                    prev_val = row[c_idx - 1]
                    remove_set = set()
                    for i in range(c_idx):
                        remove_set.add(curr_val - row[i])
                    for j in range(c_idx + 1, row_len):
                        remove_set.add(row[j] - curr_val)
                    fixed_diffs = other_diffs - remove_set
                    same_row_fixed = set()
                    for i in range(row_len):
                        if i == c_idx:
                            continue
                        for j in range(i + 1, row_len):
                            if j == c_idx:
                                continue
                            same_row_fixed.add(row[j] - row[i])
                    all_fixed = fixed_diffs | same_row_fixed
                    found = False
                    for new_val in range(curr_val - 1, prev_val, -1):
                        valid = True
                        for i in range(c_idx):
                            if (new_val - row[i]) in all_fixed:
                                valid = False
                                break
                        if not valid:
                            continue
                        for j in range(c_idx + 1, row_len):
                            if (row[j] - new_val) in all_fixed:
                                valid = False
                                break
                        if not valid:
                            continue
                        new_diffs = set()
                        for i in range(c_idx):
                            new_diffs.add(new_val - row[i])
                        for j in range(c_idx + 1, row_len):
                            new_diffs.add(row[j] - new_val)
                        if len(new_diffs) != len(new_diffs):
                            continue
                        if new_diffs & all_fixed:
                            continue
                        row[c_idx] = new_val
                        found = True
                        improved = True
                        break
                    if found:
                        other_diffs = set()
                        for j in range(n_rows):
                            if j != r_idx:
                                other_diffs |= row_diffs_set(rows[j])
        return rows

    # --- Beam Search (Structural Change from Inspiration) ---
    # Builds rows one by one, maintaining a beam of top partial solutions.
    # This explores a different neighborhood than local search on full rows.

    def beam_search_init(beam_width=50, time_limit=30.0):
        start = time.time()
        # State: (list of rows so far, set of used diffs)
        # We build rows sequentially.
        # To keep it fast, we only keep the top beam_width partial solutions based on max element so far.
        
        # Initial state: no rows, no diffs
        beam = [([], set())]
        
        for r_idx in range(n_rows):
            new_beam = []
            for rows, used_diffs in beam:
                # Generate candidates for the next row
                # We use a greedy-like approach with some randomization to generate diverse candidates
                # but we need to be careful about time.
                
                # Simple heuristic: generate a few random valid rows given used_diffs
                # and keep the best ones (lowest max)
                
                candidates = []
                for _ in range(5): # Try 5 candidates per state
                    row = [0]
                    valid_row = True
                    while len(row) < row_len:
                        last = row[-1]
                        x = last + 1
                        found = False
                        # Search for next element
                        # Limit search depth to avoid infinite loops
                        search_limit = 300 
                        while x <= search_limit:
                            valid = True
                            for a in row:
                                if (x - a) in used_diffs:
                                    valid = False
                                    break
                            if valid:
                                # Check internal uniqueness (implied by construction if we check all previous)
                                # Wait, we need to check if (x-a) is already in row_diffs?
                                # No, row_diffs are unique by construction if we ensure (x-a) is not equal to (row[j]-row[i]) for i<j.
                                # Actually, if we just check against used_diffs (global) and ensure row is strictly increasing,
                                # do we guarantee internal uniqueness?
                                # Let's check: a_j - a_i.
                                # If we pick x such that x-a_i is not in used_diffs, it's not in other rows.
                                # Is it possible that x-a_i == a_j - a_k for i<j<k in the same row?
                                # Yes. So we must check internal uniqueness.
                                
                                # Optimization: maintain a set of diffs for current row
                                # For simplicity in this beam search, we'll just do a full check or a simple set.
                                # Given small k, full check is ok.
                                
                                # Let's optimize: keep a set of current row diffs
                                # This function is a bit slow if not careful.
                                # Let's rewrite the candidate generation to be cleaner.
                                break
                            x += 1
                        if not found:
                            valid_row = False
                            break
                        row.append(x)
                        # We don't add to used_diffs yet, just checking validity against global used_diffs.
                        # We need to check against current row diffs too.
                        # Let's refine the inner loop.
                        
                        # Re-doing the logic for clarity and correctness in the candidate generator
                        # The above loop was pseudo-code. Let's implement a proper `generate_row` helper.
                        
                        pass # Placeholder, will implement generate_row below
                    
                    if valid_row:
                        # Verify row
                        # Check all diffs in row are unique and not in used_diffs
                        row_diffs = row_diffs_set(row)
                        if not (row_diffs & used_diffs) and len(row_diffs) == len(list(row_diffs)):
                            candidates.append((row, max(row)))
                
                # Add candidates to new beam
                for row, max_val in candidates:
                    new_beam.append((rows + [row], used_diffs | row_diffs_set(row)))
            
            if not new_beam:
                break
            
            # Sort by max element of the last row (or global max) to keep best ones
            # Global max is max(max(r) for r in rows)
            new_beam.sort(key=lambda x: max(max(r) for r in x[0]))
            
            # Keep top beam_width
            beam = new_beam[:beam_width]
            
            if time.time() > start + time_limit:
                break
        
        if not beam:
            return None
            
        # Return the best complete solution
        best_rows, _ = beam[0]
        return best_rows

    # Helper for beam search
    def generate_row_candidate(used_diffs, row_len, limit=300):
        row = [0]
        row_diffs = set()
        for _ in range(row_len - 1):
            last = row[-1]
            x = last + 1
            found = False
            while x <= limit:
                valid = True
                for a in row:
                    d = x - a
                    if d in used_diffs or d in row_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for a in row[:-1]:
                        row_diffs.add(x - a)
                    found = True
                    break
                x += 1
            if not found:
                return None
        return row

    def beam_search_init_v2(beam_width=50, time_limit=30.0):
        start = time.time()
        beam = [([], set())]
        
        for r_idx in range(n_rows):
            new_beam = []
            for rows, used_diffs in beam:
                # Generate diverse candidates
                # 1. Greedy (smallest)
                row_greedy = generate_row_candidate(used_diffs, row_len)
                if row_greedy:
                    new_beam.append((rows + [row_greedy], used_diffs | row_diffs_set(row_greedy)))
                
                # 2. Randomized Greedy
                for _ in range(2):
                    row_rand = [0]
                    row_diffs = set()
                    valid_row = True
                    for _ in range(row_len - 1):
                        last = row_rand[-1]
                        x = last + 1
                        found = False
                        candidates = []
                        limit = 200
                        while x <= limit and len(candidates) < 10:
                            valid = True
                            for a in row_rand:
                                d = x - a
                                if d in used_diffs or d in row_diffs:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                            x += 1
                        if not candidates:
                            valid_row = False
                            break
                        # Pick random from candidates
                        x = random.choice(candidates)
                        row_rand.append(x)
                        for a in row_rand[:-1]:
                            row_diffs.add(x - a)
                    
                    if valid_row:
                        new_beam.append((rows + [row_rand], used_diffs | row_diffs_set(row_rand)))
            
            if not new_beam:
                break
            
            # Sort by global max
            new_beam.sort(key=lambda x: max(max(r) for r in x[0]))
            beam = new_beam[:beam_width]
            
            if time.time() > start + time_limit:
                break
        
        if not beam:
            return None
        return beam[0][0]

    # --- Main Search Loop ---
    best_rows = None
    best_scope = float('inf')
    best_conflicts = float('inf')

    # 1. Beam Search Initialization (Structural Change)
    # Try to get a good starting point using beam search
    if time.time() < deadline - 30:
        bs_rows = beam_search_init_v2(beam_width=30, time_limit=10.0)
        if bs_rows:
            bs_scope = get_scope(bs_rows)
            bs_conflicts = count_conflicts(bs_rows)
            if bs_conflicts == 0:
                best_conflicts = bs_conflicts
                best_scope = bs_scope
                best_rows = bs_rows

    # 2. Standard Greedy + Local Search Restarts
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 15:
            break
        rand_prob = 0.05 + 0.05 * i
        window = 10 + 5 * i
        rows = greedy_init(rand_prob, window)
        scope = get_scope(rows)
        conflicts = count_conflicts(rows)

        t1_rows, t1_scope = local_search_targeted(rows, 4.0)
        t1_conflicts = count_conflicts(t1_rows)
        if t1_conflicts < best_conflicts or (t1_conflicts == best_conflicts and t1_scope < best_scope):
            best_conflicts = t1_conflicts
            best_scope = t1_scope
            best_rows = t1_rows

        if time.time() > deadline - 15:
            break
        t2_rows, t2_scope = local_search_sa(t1_rows, 4.0)
        t2_conflicts = count_conflicts(t2_rows)
        if t2_conflicts < best_conflicts or (t2_conflicts == best_conflicts and t2_scope < best_scope):
            best_conflicts = t2_conflicts
            best_scope = t2_scope
            best_rows = t2_rows

        if best_conflicts == 0 and best_scope < 120:
            break

    if best_rows is None:
        # Fallback
        best_rows = greedy_init(0.1, 20)
        best_scope = get_scope(best_rows)
        best_conflicts = count_conflicts(best_rows)

    # 3. Final Polish
    if time.time() < deadline - 3.0:
        polish_time = min(deadline - time.time() - 3.0, 30.0)
        polished = high_precision_polish(best_rows, polish_time)
        pol_conflicts = count_conflicts(polished)
        if pol_conflicts == 0:
            pol_scope = get_scope(polished)
            if pol_scope < best_scope:
                best_scope = pol_scope
                best_rows = polished

    return best_rows
