import time
import random
import sys

def construct(n_rows, k):
    """
    Constructs n_rows rows of k+1 increasing integers starting at 0.
    All positive differences within rows must be distinct over all rows.
    Minimizes the largest entry.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    
    def get_scope(rows):
        return max(max(r) for r in rows)

    def validate(rows):
        """Check if rows are valid (strictly increasing, start at 0, unique diffs)."""
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
                    if d <= 0:
                        return False
                    if d in used:
                        return False
                    used.add(d)
        return True

    def get_diffs(row):
        """Get all positive differences for a single row."""
        return [row[j] - row[i] for i in range(len(row)) for j in range(i+1, len(row))]

    def greedy_init(rand_prob=0.15, window=50):
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
                limit = 1000
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

    def local_search(rows, time_limit):
        """
        Structural refinement: Targeted descent to minimize the global maximum (scope).
        We repeatedly identify the largest element in the entire configuration.
        We attempt to replace it with a smaller valid value.
        If successful, the scope might decrease, or at least the distribution tightens.
        """
        start = time.time()
        best_rows = [r[:] for r in rows]
        best_scope = get_scope(best_rows)
        
        # Precompute all diffs for fast access
        def compute_all_diffs(rows):
            all_diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        all_diffs.add(r[j] - r[i])
            return all_diffs

        while time.time() < start + time_limit:
            # Find the index of the global maximum
            max_val = -1
            max_r_idx = -1
            max_c_idx = -1
            
            for r_idx, r in enumerate(best_rows):
                for c_idx, val in enumerate(r):
                    if val > max_val:
                        max_val = val
                        max_r_idx = r_idx
                        max_c_idx = c_idx
            
            # We don't want to change the first element (0)
            if max_c_idx == 0:
                break
            
            current_val = max_val
            prev_val = best_rows[max_r_idx][max_c_idx - 1]
            
            # Let's build a set of all diffs currently in the system.
            all_diffs = compute_all_diffs(best_rows)
            
            # Diffs to remove (involving target)
            remove_set = set()
            row = best_rows[max_r_idx]
            for i in range(max_c_idx):
                remove_set.add(current_val - row[i])
            for j in range(max_c_idx + 1, len(row)):
                remove_set.add(row[j] - current_val)
            
            # Fixed diffs are all_diffs - remove_set
            fixed_diffs = all_diffs - remove_set
            
            found = False
            # Try values from current_val - 1 down to prev_val + 1
            for new_val in range(current_val - 1, prev_val, -1):
                # Check validity against fixed_diffs
                valid = True
                # Check diffs with left neighbors
                for i in range(max_c_idx):
                    d = new_val - row[i]
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                    
                # Check diffs with right neighbors
                for j in range(max_c_idx + 1, len(row)):
                    d = row[j] - new_val
                    if d in fixed_diffs:
                        valid = False
                        break
                if not valid:
                    continue
                
                new_left_diffs = [new_val - row[i] for i in range(max_c_idx)]
                new_right_diffs = [row[j] - new_val for j in range(max_c_idx + 1, len(row))]
                
                # Check for duplicates within new diffs
                all_new = new_left_diffs + new_right_diffs
                if len(set(all_new)) != len(all_new):
                    continue
                
                # If we pass all checks, this new_val is valid.
                row[max_c_idx] = new_val
                found = True
                if new_val < best_scope:
                    new_scope = get_scope(best_rows)
                    if new_scope < best_scope:
                        best_scope = new_scope
                break
            
            if not found:
                # If we can't reduce the current max, try a random swap on a non-max element to "jolt" the system.
                success_in_step = False
                r_idx = random.randint(0, n_rows-1)
                c_idx = random.randint(1, row_len-1)
                
                curr_val = best_rows[r_idx][c_idx]
                prev_v = best_rows[r_idx][c_idx-1]
                next_v = best_rows[r_idx][c_idx+1] if c_idx < row_len-1 else curr_val + 100
                
                # Re-compute fixed diffs
                all_diffs = compute_all_diffs(best_rows)
                
                remove_set = set()
                row = best_rows[r_idx]
                for i in range(c_idx):
                    remove_set.add(curr_val - row[i])
                for j in range(c_idx + 1, len(row)):
                    remove_set.add(row[j] - curr_val)
                
                fixed_diffs = all_diffs - remove_set
                
                # Try a few random candidates
                low = prev_v + 1
                high = next_v - 1 if c_idx < row_len-1 else curr_val + 50
                
                for _ in range(30):
                    if low > high: break
                    cand = random.randint(low, high)
                    if cand == curr_val: continue
                    
                    valid = True
                    for i in range(c_idx):
                        d = cand - row[i]
                        if d in fixed_diffs:
                            valid = False
                            break
                    if valid:
                        for j in range(c_idx + 1, len(row)):
                            d = row[j] - cand
                            if d in fixed_diffs:
                                valid = False
                                break
                    if not valid:
                        continue
                    
                    new_left_diffs = [cand - row[i] for i in range(c_idx)]
                    new_right_diffs = [row[j] - cand for j in range(c_idx + 1, len(row))]
                    all_new = new_left_diffs + new_right_diffs
                    if len(set(all_new)) != len(all_new):
                        continue
                    
                    row[c_idx] = cand
                    success_in_step = True
                    break
                
                if not success_in_step:
                    # If even random jolts fail, we might be stuck. 
                    # In a full SA, we'd accept worse moves. Here, we just break to avoid infinite loop.
                    # However, to allow escaping, we could implement a simple SA step here.
                    # For now, let's try to increase a random value slightly to open up space? No, that increases scope.
                    # Let's just break out of this specific local search phase and restart with a new greedy init.
                    break
        
        return best_rows, best_scope

    def simulated_annealing(rows, time_limit):
        """
        Simulated Annealing to escape local minima.
        """
        current_rows = [r[:] for r in rows]
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        T = 50.0
        T_min = 0.1
        alpha = 0.995
        
        def compute_all_diffs(rows):
            all_diffs = set()
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        all_diffs.add(r[j] - r[i])
            return all_diffs

        current_diffs = compute_all_diffs(current_rows)
        
        while time.time() < start_time + time_limit:
            # Select a move
            if random.random() < 0.6:
                # Find the index of the global maximum
                max_val = -1
                max_idx = (-1, -1)
                for i in range(n_rows):
                    for j in range(1, len(current_rows[i])):
                        if current_rows[i][j] > max_val:
                            max_val = current_rows[i][j]
                            max_idx = (i, j)
                i, j = max_idx
            else:
                i = random.randint(0, n_rows - 1)
                j = random.randint(1, len(current_rows[i]) - 1)
            
            row = current_rows[i]
            old_val = row[j]
            
            # Determine valid range for new value
            lower_bound = row[j-1] + 1
            upper_bound = old_val - 1
            if j < len(row) - 1:
                upper_bound = min(upper_bound, row[j+1] - 1)
            
            max_possible = row[j+1] - 1 if j < len(row) - 1 else old_val + 10
            min_possible = row[j-1] + 1
            
            if min_possible > max_possible:
                T *= alpha
                continue
                
            search_max = min(max_possible, old_val + 5)
            if search_max < min_possible:
                T *= alpha
                continue
                
            new_val = random.randint(min_possible, search_max)
            
            if new_val == old_val:
                continue
                
            # Calculate old diffs involving index j
            old_diffs = set()
            for t in range(len(row)):
                if t != j:
                    if row[j] > row[t]:
                        old_diffs.add(row[j] - row[t])
                    else:
                        old_diffs.add(row[t] - row[j])
            
            # Calculate new diffs involving index j
            new_diffs = set()
            for t in range(len(row)):
                if t != j:
                    if new_val > row[t]:
                        new_diffs.add(new_val - row[t])
                    else:
                        new_diffs.add(row[t] - new_val)
            
            # Check conflicts
            # The new diffs must not conflict with (current_diffs - old_diffs)
            conflict = False
            for d in new_diffs:
                if d in current_diffs and d not in old_diffs:
                    conflict = True
                    break
            
            if conflict:
                T *= alpha
                continue
                
            # Valid move!
            row[j] = new_val
            current_diffs = (current_diffs - old_diffs) | new_diffs
            
            new_scope = get_scope(current_rows)
            
            if new_scope < current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                delta = new_scope - current_scope
                if delta > 0:
                    prob = 2.718281828 ** (-delta / T)
                    if random.random() > prob:
                        # Reject: Undo move
                        row[j] = old_val
                        current_diffs = (current_diffs - new_diffs) | old_diffs
                        T *= alpha
                        continue
            
            T *= alpha
            if T < T_min:
                T = T_min
                
        return best_rows, best_scope

    # Main Logic
    # 1. Get initial solution
    init_rows = greedy_init()
    
    # 2. Optimize
    time_left = deadline - time.time()
    if time_left > 1:
        # First, do local search to quickly reduce scope
        ls_time = min(30.0, time_left * 0.5)
        ls_rows, ls_scope = local_search(init_rows, ls_time)
        
        time_left = deadline - time.time()
        if time_left > 1:
            # Then, do SA to escape local minima
            sa_rows, sa_scope = simulated_annealing(ls_rows, time_left - 1.0)
            
            if sa_scope < ls_scope:
                best_rows = sa_rows
            else:
                best_rows = ls_rows
        else:
            best_rows = ls_rows
    else:
        best_rows = init_rows
        
    # 3. Final Verification
    if not validate(best_rows):
        best_rows = init_rows

    return best_rows
