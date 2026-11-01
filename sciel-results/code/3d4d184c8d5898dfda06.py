import time
import random
import math

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 138.0  # safety margin before 150s kill

    row_len = k + 1
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row

    def get_scope(rows):
        return max(max(r) for r in rows)

    def full_validate(rows):
        """Return scope if valid, else None."""
        seen = set()
        scope = 0
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    diff = row[j] - ri
                    if diff in seen:
                        return None
                    seen.add(diff)
            scope = max(scope, max(row))
        return scope

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
                limit = last + 150
                while x <= limit:
                    valid = True
                    new_diffs = []
                    for a in row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                        new_diffs.append(d)
                    
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= window:
                            break
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

    def beam_search(max_scope, beam_width=500, time_budget=30.0):
        """
        Beam search over partial configurations.
        State: (rows, used_diffs_set, total_placed)
        We expand by adding one element to one row at a time.
        Prune by scope <= max_scope and keep top beam_width by heuristic.
        """
        if time.time() > deadline - time_budget:
            return None, float('inf')
        
        # Initial state: all rows have [0]
        initial_rows = [[0] for _ in range(n_rows)]
        initial_used = set()
        
        # Beam: list of (heuristic_score, rows, used_diffs)
        # heuristic: current max scope (lower is better)
        beam = [(0, initial_rows, initial_used)]
        
        total_placed = n_rows  # each row has 1 element (the 0)
        target_placed = n_rows * row_len
        
        while total_placed < target_placed:
            if time.time() > deadline - time_budget:
                break
                
            next_beam = []
            
            for score, rows, used_diffs in beam:
                # Determine which row to expand: the one with fewest elements
                # Tie-break: smallest last element
                min_len = row_len
                target_row_idx = -1
                min_last_val = float('inf')
                
                for i in range(n_rows):
                    if len(rows[i]) < min_len:
                        min_len = len(rows[i])
                        target_row_idx = i
                        min_last_val = rows[i][-1]
                    elif len(rows[i]) == min_len:
                        if rows[i][-1] < min_last_val:
                            min_last_val = rows[i][-1]
                            target_row_idx = i
                
                if target_row_idx == -1:
                    # All rows full
                    current_scope = max(max(r) for r in rows)
                    if current_scope < max_scope:
                        next_beam.append((current_scope, rows, used_diffs))
                    continue
                
                i = target_row_idx
                current_row = rows[i]
                last_val = current_row[-1]
                
                # Find valid candidates for next element
                # Search from last_val+1 up to max_scope
                candidates = []
                x = last_val + 1
                while x <= max_scope:
                    valid = True
                    for a in current_row:
                        d = x - a
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= 15:
                            break
                    x += 1
                
                if not candidates:
                    continue
                
                # Create new states for each candidate
                for new_val in candidates:
                    new_rows = [r[:] for r in rows]
                    new_rows[i].append(new_val)
                    
                    new_used = set(used_diffs)
                    for a in new_rows[i]:
                        if a != new_val:
                            new_used.add(new_val - a)
                    
                    new_scope = max(max(r) for r in new_rows)
                    if new_scope < max_scope:
                        next_beam.append((new_scope, new_rows, new_used))
            
            if not next_beam:
                break
            
            # Sort by scope and keep top beam_width
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]
            total_placed += 1
            
            # Check if any state is complete
            for scope, rows, used in beam:
                if all(len(r) == row_len for r in rows):
                    if scope < max_scope:
                        max_scope = scope
        
        # Find best complete solution in beam
        best_rows = None
        best_scope = float('inf')
        for scope, rows, used in beam:
            if all(len(r) == row_len for r in rows):
                if scope < best_scope:
                    best_scope = scope
                    best_rows = rows
        
        return best_rows, best_scope

    def local_search_sa(rows, time_limit):
        """
        Optimizes rows using Simulated Annealing.
        Returns the best rows found.
        """
        current_rows = [r[:] for r in rows]
        
        # Build diff counts
        diff_counts = {}
        for r in current_rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        T = 30.0
        T_min = 0.01
        cooling = 0.998
        
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            if T < T_min:
                break
                
            r_idx = random.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            pos = random.randint(1, k)
            
            current_val = row[pos]
            
            lower_bound = row[pos - 1] + 1
            upper_bound = row[pos + 1] - 1 if pos < k else current_val + 15
            
            if upper_bound <= lower_bound:
                continue
            
            new_val = random.randint(lower_bound, upper_bound)
            
            if new_val == current_val:
                continue
            
            old_diffs = []
            for i in range(pos):
                old_diffs.append(current_val - row[i])
            for j in range(pos + 1, row_len):
                old_diffs.append(row[j] - current_val)
            
            new_diffs = []
            for i in range(pos):
                new_diffs.append(new_val - row[i])
            for j in range(pos + 1, row_len):
                new_diffs.append(row[j] - new_val)
            
            valid = True
            old_diffs_set = set(old_diffs)
            for d in new_diffs:
                if d in old_diffs_set:
                    continue
                if d in diff_counts:
                    valid = False
                    break
            
            if not valid:
                T *= cooling
                continue
            
            row[pos] = new_val
            
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] == 0:
                    del diff_counts[d]
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
            
            new_scope = get_scope(current_rows)
            
            if new_scope <= current_scope:
                current_scope = new_scope
                if new_scope < best_scope:
                    best_scope = new_scope
                    best_rows = [r[:] for r in current_rows]
            else:
                delta = new_scope - current_scope
                if random.random() < math.exp(-delta / T):
                    current_scope = new_scope
                else:
                    row[pos] = current_val
                    for d in old_diffs:
                        diff_counts[d] = diff_counts.get(d, 0) + 1
                    for d in new_diffs:
                        diff_counts[d] -= 1
                        if diff_counts[d] == 0:
                            del diff_counts[d]
            
            T *= cooling
            
        return best_rows, best_scope

    def exact_local_polish(rows, time_limit):
        """
        Systematically try to reduce each element to the smallest valid value.
        """
        current_rows = [r[:] for r in rows]
        
        diff_counts = {}
        for r in current_rows:
            for i in range(row_len):
                for j in range(i+1, row_len):
                    d = r[j] - r[i]
                    diff_counts[d] = diff_counts.get(d, 0) + 1
        
        current_scope = get_scope(current_rows)
        best_rows = [r[:] for r in current_rows]
        best_scope = current_scope
        
        end_time = time.time() + time_limit
        
        improved = True
        while improved and time.time() < end_time:
            improved = False
            for r_idx in range(n_rows):
                if time.time() > end_time:
                    break
                row = current_rows[r_idx]
                for pos in range(k, 0, -1):  # from end to start
                    if time.time() > end_time:
                        break
                    current_val = row[pos]
                    lower_bound = row[pos - 1] + 1
                    
                    # Try to find smallest valid value
                    for new_val in range(lower_bound, current_val):
                        old_diffs = []
                        for i in range(pos):
                            old_diffs.append(current_val - row[i])
                        for j in range(pos + 1, row_len):
                            old_diffs.append(row[j] - current_val)
                        
                        new_diffs = []
                        for i in range(pos):
                            new_diffs.append(new_val - row[i])
                        for j in range(pos + 1, row_len):
                            new_diffs.append(row[j] - new_val)
                        
                        valid = True
                        old_diffs_set = set(old_diffs)
                        for d in new_diffs:
                            if d in old_diffs_set:
                                continue
                            if d in diff_counts:
                                valid = False
                                break
                        
                        if valid:
                            # Apply move
                            row[pos] = new_val
                            for d in old_diffs:
                                diff_counts[d] -= 1
                                if diff_counts[d] == 0:
                                    del diff_counts[d]
                            for d in new_diffs:
                                diff_counts[d] = diff_counts.get(d, 0) + 1
                            
                            new_scope = get_scope(current_rows)
                            if new_scope < best_scope:
                                best_scope = new_scope
                                best_rows = [r[:] for r in current_rows]
                            improved = True
                            break  # Found a reduction, move to next position
        
        return best_rows, best_scope

    # Main strategy:
    # 1. Get a baseline from greedy
    # 2. Use beam search to find a better starting point
    # 3. Polish with SA and exact local search
    
    best_rows = None
    best_scope = float('inf')
    
    # Phase 1: Greedy baseline
    rows = greedy_init(0.1, 20)
    scope = get_scope(rows)
    best_scope = scope
    best_rows = rows
    
    # Phase 2: Beam search to improve
    # Try beam search with a slightly tighter bound
    beam_scope_limit = best_scope - 1
    if time.time() < deadline - 60:
        bs_rows, bs_scope = beam_search(beam_scope_limit, beam_width=300, time_budget=25.0)
        if bs_rows is not None and bs_scope < best_scope:
            best_scope = bs_scope
            best_rows = bs_rows
    
    # Phase 3: SA polish
    if time.time() < deadline - 20:
        sa_time = min(20.0, deadline - time.time() - 10)
        sa_rows, sa_scope = local_search_sa(best_rows, sa_time)
        if sa_scope < best_scope:
            best_scope = sa_scope
            best_rows = sa_rows
    
    # Phase 4: Exact local polish
    if time.time() < deadline - 5:
        polish_time = min(10.0, deadline - time.time() - 2)
        polish_rows, polish_scope = exact_local_polish(best_rows, polish_time)
        if polish_scope < best_scope:
            best_scope = polish_scope
            best_rows = polish_rows
    
    # Final validation
    if best_rows is not None:
        valid_scope = full_validate(best_rows)
        if valid_scope is not None:
            best_scope = valid_scope
        else:
            # Fallback to greedy
            best_rows = greedy_init(0.1, 20)
            best_scope = get_scope(best_rows)
    
    return best_rows
