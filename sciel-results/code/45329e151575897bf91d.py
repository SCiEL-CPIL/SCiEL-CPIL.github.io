# EVOLVE-BLOCK-START
import sys
import time
import random
import math

def construct(n_rows, k):
    """
    Find n_rows rows of k+1 increasing integers starting at 0 such that all
    positive differences within each row are distinct across all rows.
    Minimize the largest entry.
    """
    if n_rows <= 0 or k <= 0:
        return [[0] * (k + 1)] * max(n_rows, 0)
    
    total_diffs_needed = n_rows * k * (k + 1) // 2
    lb = total_diffs_needed
    
    # Improved Greedy Baseline
    def greedy_construct(n_rows, k, randomize=False, seed=0):
        used = set()
        rows = []
        rng = random.Random(seed) if randomize else None
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

    # Get a good upper bound using randomized greedy restarts
    best_greedy_rows = None
    best_greedy_max = float('inf')
    for i in range(20):
        # Standard greedy
        r = greedy_construct(n_rows, k)
        m = max(max(row) for row in r)
        if m < best_greedy_max:
            best_greedy_max = m
            best_greedy_rows = r
            
        # Randomized greedy: pick from first few valid candidates
        used = set()
        rows = []
        rng = random.Random(i)
        success = True
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                valid_candidates = []
                # Look ahead a bit
                for _look in range(10):
                    x += 1
                    new_diffs = {x - a for a in row}
                    if not new_diffs & used:
                        valid_candidates.append((x, new_diffs))
                        if len(valid_candidates) >= 5:
                            break
                if not valid_candidates:
                    success = False
                    break
                # Pick randomly from valid candidates
                x, nd = rng.choice(valid_candidates)
                row.append(x)
                used |= nd
            rows.append(row)
        
        if success:
            m = max(max(row) for row in rows)
            if m < best_greedy_max:
                best_greedy_max = m
                best_greedy_rows = rows

    if best_greedy_rows is None:
        best_greedy_rows = greedy_construct(n_rows, k)
        best_greedy_max = max(max(row) for row in best_greedy_rows)

    start_time = time.time()
    time_limit = 115.0  # Safety margin for 120s limit

    # Helper to compute all positive differences for a row
    def get_row_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    # Helper to check if a set of diffs is valid given global used diffs
    # This is used in CSP
    def is_valid_diffs(new_diffs, used_bits):
        for d in new_diffs:
            if (used_bits >> d) & 1:
                return False
        return True

    def solve_csp(n_rows, k, max_entry, timeout, seed=0):
        deadline = time.time() + timeout
        used_bits = 0
        rows = []
        rng = random.Random(seed)
        
        def build_row(current_row, used_bits, row_idx, deadline):
            if time.time() > deadline:
                return False
            
            if len(current_row) == k + 1:
                return True
            
            last = current_row[-1]
            remaining = k + 1 - len(current_row)
            
            min_x = last + 1
            max_x = max_entry - (remaining - 1)
            
            if min_x > max_x:
                return False
            
            candidates = []
            for x in range(min_x, max_x + 1):
                new_diff_bits = 0
                conflict = False
                for a in current_row:
                    d = x - a
                    if (used_bits >> d) & 1:
                        conflict = True
                        break
                    new_diff_bits |= (1 << d)
                
                if not conflict:
                    candidates.append((x, new_diff_bits))
            
            if not candidates:
                return False
            
            # Heuristic: Shuffle candidates to add randomness
            rng.shuffle(candidates)
            
            for x, bits in candidates:
                if time.time() > deadline:
                    return False
                
                current_row.append(x)
                if build_row(current_row, used_bits | bits, row_idx, deadline):
                    return True
                current_row.pop()
            
            return False
        
        for row_idx in range(n_rows):
            if time.time() > deadline:
                return None
            current_row = [0]
            if build_row(current_row, used_bits, row_idx, deadline):
                rows.append(current_row)
                for i in range(len(current_row)):
                    for j in range(i + 1, len(current_row)):
                        d = current_row[j] - current_row[i]
                        used_bits |= (1 << d)
            else:
                return None
        
        return rows

    best_rows = best_greedy_rows
    best_max = best_greedy_max

    # Phase 1: CSP Search for small M
    # We iterate M from lb to best_greedy_max
    # Allocate time based on M
    for M in range(lb, best_greedy_max + 1):
        elapsed = time.time() - start_time
        remaining_time = time_limit - elapsed - 2
        
        if remaining_time < 0.5:
            break
        
        # Time budget allocation
        dist = M - lb
        if dist == 0:
            budget = min(10.0, remaining_time)
        elif dist <= 5:
            budget = min(5.0, remaining_time)
        else:
            budget = min(2.0, remaining_time)
        
        if budget < 0.1:
            continue
            
        # Number of restarts
        num_restarts = max(1, int(budget / 0.5))
        
        found = False
        for seed in range(num_restarts):
            if time.time() > start_time + time_limit - 2:
                break
            restart_budget = min(0.5, budget / (num_restarts - seed))
            
            res = solve_csp(n_rows, k, M, restart_budget, seed=seed)
            if res is not None:
                best_rows = res
                best_max = M
                found = True
                break
        
        if found:
            break

    # Phase 2: Simulated Annealing Polish
    # Use SA to try to reduce the max entry further, starting from the best valid solution found.
    # This is robust against the "brittleness" of backtracking.
    
    def simulated_annealing_polish(rows, n_rows, k, time_limit, seed=0):
        """
        Refines a valid set of rows to minimize the maximum entry using Simulated Annealing.
        """
        if not rows:
            return None
            
        rng = random.Random(seed)
        
        # Deep copy
        current_rows = [list(r) for r in rows]
        current_max = max(max(r) for r in current_rows)
        
        # Precompute global used diffs for fast checking? 
        # It's better to maintain a set of all used diffs.
        all_used_diffs = set()
        for r in current_rows:
            all_used_diffs |= get_row_diffs(r)
        
        # Initial temperature
        T = current_max * 0.5
        T_min = 0.01
        cooling_rate = 0.95
        max_iters = 50000 # Cap iterations
        
        start_sa = time.time()
        
        for i in range(max_iters):
            if time.time() - start_sa > time_limit:
                break
            if T < T_min:
                break
            
            # Pick a random row
            r_idx = rng.randint(0, n_rows - 1)
            row = current_rows[r_idx]
            
            # Pick a random index to modify (1 to k, since index 0 is 0)
            # We modify one element at a time to maintain increasing order locally
            j_idx = rng.randint(1, k)
            
            old_val = row[j_idx]
            
            # Determine valid range for new value
            min_val = row[j_idx - 1] + 1
            max_val = row[j_idx + 1] - 1 if j_idx < k else current_max + 10 # If last element, can go higher temporarily? 
            # Actually, we want to minimize max, so increasing max is bad. 
            # But SA allows it. Let's allow range up to current_max + 5 to escape local minima.
            if j_idx == k:
                max_val = current_max + 5
            
            if min_val > max_val:
                continue
                
            # Propose new value
            # Bias towards smaller values if we are trying to minimize? 
            # Uniform random in range is standard.
            new_val = rng.randint(min_val, max_val)
            
            if new_val == old_val:
                continue
                
            # Check validity of new row
            new_row = row[:]
            new_row[j_idx] = new_val
            
            # Check strictly increasing
            valid_row = True
            for m in range(1, len(new_row)):
                if new_row[m] <= new_row[m-1]:
                    valid_row = False
                    break
            
            if not valid_row:
                continue
                
            # Check differences
            new_diffs = get_row_diffs(new_row)
            old_diffs = get_row_diffs(row)
            
            # Remove old diffs, add new diffs, check for conflicts with other rows
            # We need to ensure new_diffs does not intersect with (all_used_diffs - old_diffs)
            # And that new_diffs itself has no duplicates (handled by set, but get_row_diffs returns set so no dupes within row)
            # Wait, get_row_diffs returns a set, so internal duplicates are impossible for distinct integers.
            
            # Conflict check:
            # The set of diffs used by OTHER rows is all_used_diffs - old_diffs.
            # If new_diffs intersects with that, it's invalid.
            # Also, if new_diffs has any element in old_diffs, that's fine, it's just reused within the same row? 
            # No, differences within a row must be distinct? 
            # "all differences between two entries of the same row are distinct over all rows"
            # This implies:
            # 1. Diffs within row 1 are distinct from each other.
            # 2. Diffs within row 2 are distinct from each other.
            # 3. Diffs in row 1 are distinct from diffs in row 2.
            # So yes, new_diffs must be disjoint from (all_used_diffs - old_diffs).
            
            other_diffs = all_used_diffs - old_diffs
            if new_diffs & other_diffs:
                continue
            
            # Calculate new max
            new_max = max(max(r) for r in current_rows if r is not new_row)
            if new_max < new_val:
                 new_max = new_val
            # Actually, just calculate max of the whole set
            # Since only one row changed, new_max = max(max(other rows), max(new_row))
            
            # Compute cost delta
            delta_cost = new_max - current_max
            
            # Acceptance probability
            if delta_cost <= 0:
                # Always accept improvement
                # Apply move
                row[:] = new_row
                current_max = new_max
                # Update all_used_diffs
                all_used_diffs -= old_diffs
                all_used_diffs |= new_diffs
            else:
                # Probabilistic acceptance
                p = math.exp(-delta_cost / T)
                if rng.random() < p:
                    # Apply move
                    row[:] = new_row
                    current_max = new_max
                    all_used_diffs -= old_diffs
                    all_used_diffs |= new_diffs
            
            # Cool down
            T *= cooling_rate
        
        return current_rows

    # Run SA on the best solution found
    remaining_time = time_limit - (time.time() - start_time) - 1
    if remaining_time > 1.0:
        # Try a few SA restarts with different seeds
        best_sa_rows = best_rows
        best_sa_max = best_max
        
        num_sa_restarts = max(1, int(remaining_time / 2.0))
        for i in range(num_sa_restarts):
            sa_budget = remaining_time / (num_sa_restarts - i) - 0.1
            if sa_budget < 0.5:
                continue
            sa_res = simulated_annealing_polish(best_rows, n_rows, k, sa_budget, seed=100+i)
            if sa_res:
                sa_max = max(max(r) for r in sa_res)
                if sa_max < best_sa_max:
                    best_sa_max = sa_max
                    best_sa_rows = sa_res
        
        if best_sa_max < best_max:
            best_rows = best_sa_rows
            best_max = best_sa_max

    # Final Validation
    def validate(rows, n_rows, k):
        if len(rows) != n_rows:
            return False
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
            for i in range(1, len(row)):
                if row[i] <= row[i-1]:
                    return False
        return True
    
    if not validate(best_rows, n_rows, k):
        # Fallback to greedy
        best_rows = greedy_construct(n_rows, k)

    return best_rows
# EVOLVE-BLOCK-END
