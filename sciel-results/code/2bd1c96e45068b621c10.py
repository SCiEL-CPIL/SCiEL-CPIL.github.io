import random
import time
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # k+1 elements means k gaps
    num_gaps = k
    diffs_per_row = k * (k + 1) // 2
    
    def time_left():
        return deadline - time.time()

    def get_diffs_from_gaps(gaps):
        """
        Calculate all pairwise differences from a list of gaps.
        gaps[i] = a_{i+1} - a_i
        Elements: a_0=0, a_1=gaps[0], a_2=gaps[0]+gaps[1], ...
        """
        diffs = set()
        # a_j - a_i = sum(gaps[i:j])
        # We can compute this efficiently
        n = len(gaps)
        # Prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + gaps[i]
        
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                d = prefix[j] - prefix[i]
                diffs.add(d)
        return diffs

    def get_scope_from_gaps(gaps):
        return sum(gaps)

    def count_duplicates(all_diffs_list):
        """
        all_diffs_list is a list of sets (one per row).
        Returns the number of differences that appear in more than one set.
        """
        counter = {}
        for diffs in all_diffs_list:
            for d in diffs:
                counter[d] = counter.get(d, 0) + 1
        
        dup_count = 0
        for d, count in counter.items():
            if count > 1:
                dup_count += count - 1 # Or just count the number of conflicting diffs
        return dup_count

    def evaluate_solution(rows_gaps):
        """
        Returns (dup_count, max_scope)
        """
        all_diffs = []
        max_scope = 0
        for gaps in rows_gaps:
            diffs = get_diffs_from_gaps(gaps)
            all_diffs.append(diffs)
            scope = get_scope_from_gaps(gaps)
            if scope > max_scope:
                max_scope = scope
        
        dup_count = count_duplicates(all_diffs)
        return dup_count, max_scope

    def greedy_init():
        """
        Create an initial set of rows using a greedy approach.
        We want to avoid duplicates.
        """
        rows_gaps = []
        used_diffs = set()
        
        for r in range(n_rows):
            # Try to find gaps that avoid used_diffs
            # Start with small gaps
            best_gaps = None
            best_scope = float('inf')
            
            # Simple heuristic: try random small gaps
            for _ in range(50):
                gaps = [1] * num_gaps
                # Randomize a bit
                for i in range(num_gaps):
                    gaps[i] = random.randint(1, 5)
                
                diffs = get_diffs_from_gaps(gaps)
                if diffs & used_diffs:
                    continue
                
                scope = get_scope_from_gaps(gaps)
                if scope < best_scope:
                    best_scope = scope
                    best_gaps = gaps
                    # If we find a very good one, we can stop
                    if scope < 10:
                        break
            
            if best_gaps is None:
                # Fallback: increase gaps until no conflict
                gaps = [1] * num_gaps
                while True:
                    diffs = get_diffs_from_gaps(gaps)
                    if not (diffs & used_diffs):
                        break
                    # Increase a random gap
                    idx = random.randint(0, num_gaps - 1)
                    gaps[idx] += 1
                best_gaps = gaps
            
            rows_gaps.append(best_gaps)
            used_diffs.update(get_diffs_from_gaps(best_gaps))
            
        return rows_gaps

    # 1. Initialize
    current_rows = greedy_init()
    current_dup, current_scope = evaluate_solution(current_rows)
    
    # If initial is invalid, we need to start SA from there.
    # If initial is valid, we can start SA to reduce scope.
    
    best_rows = [list(r) for r in current_rows]
    best_dup, best_scope = current_dup, current_scope
    
    # SA Parameters
    # Cost function: C = dup_count * 1000 + scope
    # We want to minimize C.
    # If dup_count > 0, we prioritize removing duplicates.
    # If dup_count == 0, we prioritize reducing scope.
    
    T_start = 50.0
    T_end = 0.1
    cooling_rate = 0.999
    
    # We will run SA for a fixed number of steps or until time runs out.
    # Since n_rows and k are small (likely k=8, n_rows=4), the state space is manageable.
    # Each step involves perturbing one gap in one row.
    
    steps_per_second_target = 50000 # Rough estimate
    # We'll just loop until time is up, checking time every 1000 steps.
    
    step = 0
    T = T_start
    
    # To speed up evaluation, we can maintain the set of diffs for each row
    # and a global counter of diff frequencies.
    
    # Let's implement a faster incremental evaluation for SA.
    
    row_diffs = [get_diffs_from_gaps(gaps) for gaps in current_rows]
    diff_counter = {}
    for diffs in row_diffs:
        for d in diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1
    
    def get_current_dup_count():
        count = 0
        for d, c in diff_counter.items():
            if c > 1:
                count += c - 1
        return count

    def get_current_max_scope():
        return max(get_scope_from_gaps(gaps) for gaps in current_rows)

    # Initial State
    current_dup = get_current_dup_count()
    current_scope = get_current_max_scope()
    current_cost = current_dup * 1000 + current_scope
    
    best_dup = current_dup
    best_scope = current_scope
    best_cost = current_cost
    best_rows = [list(r) for r in current_rows]
    
    # SA Loop
    while time_left() > 2.0:
        # Pick a random row and a random gap
        row_idx = random.randint(0, n_rows - 1)
        gap_idx = random.randint(0, num_gaps - 1)
        
        # Store old state for rollback
        old_gap_val = current_rows[row_idx][gap_idx]
        old_row_diffs = row_diffs[row_idx]
        old_scope = get_scope_from_gaps(current_rows[row_idx])
        
        # Perturb
        # Change gap by +/- 1 or random small amount
        delta = random.choice([-2, -1, 1, 2])
        new_gap_val = old_gap_val + delta
        if new_gap_val < 1:
            new_gap_val = 1
            
        # Apply change
        current_rows[row_idx][gap_idx] = new_gap_val
        new_scope = old_scope + (new_gap_val - old_gap_val)
        
        # Recalculate diffs for this row
        # Optimization: Instead of recalculating all diffs, we can update them.
        # But for simplicity and correctness first, recalculate.
        # For k=8, 36 diffs, this is fast.
        new_row_diffs = get_diffs_from_gaps(current_rows[row_idx])
        
        # Update diff_counter
        # Remove old diffs
        for d in old_row_diffs:
            diff_counter[d] -= 1
            if diff_counter[d] == 0:
                del diff_counter[d]
        
        # Add new diffs
        for d in new_row_diffs:
            diff_counter[d] = diff_counter.get(d, 0) + 1
        
        # Calculate new cost
        new_dup = get_current_dup_count()
        # Max scope needs to be checked against other rows
        # We only changed one row's scope.
        # If new_scope < old_scope, max_scope might decrease.
        # If new_scope > old_scope, max_scope might increase.
        # To be safe, calculate max scope.
        new_max_scope = 0
        for i in range(n_rows):
            s = get_scope_from_gaps(current_rows[i])
            if s > new_max_scope:
                new_max_scope = s
        
        new_cost = new_dup * 1000 + new_max_scope
        
        delta_cost = new_cost - current_cost
        
        # Metropolis Criterion
        if delta_cost <= 0 or random.random() < math.exp(-delta_cost / T):
            # Accept
            current_dup = new_dup
            current_scope = new_max_scope
            current_cost = new_cost
            row_diffs[row_idx] = new_row_diffs
            
            # Update best
            if current_cost < best_cost:
                best_cost = current_cost
                best_dup = current_dup
                best_scope = current_scope
                best_rows = [list(r) for r in current_rows]
                
            # If we have a valid solution, we might want to cool down faster
            # or focus more on reducing scope.
            # The cost function already handles this by weighting dup_count heavily.
            
        else:
            # Reject
            # Rollback
            current_rows[row_idx][gap_idx] = old_gap_val
            row_diffs[row_idx] = old_row_diffs
            
            # Restore diff_counter
            for d in new_row_diffs:
                diff_counter[d] -= 1
                if diff_counter[d] == 0:
                    del diff_counter[d]
            
            for d in old_row_diffs:
                diff_counter[d] = diff_counter.get(d, 0) + 1
        
        # Cool down
        T *= cooling_rate
        if T < T_end:
            T = T_end
        
        step += 1
        if step % 1000 == 0:
            # Optional: Restart if stuck? 
            # For now, just continue.
            pass

    # Final Validation
    def validate(rows_gaps):
        all_diffs = set()
        for gaps in rows_gaps:
            if any(g < 1 for g in gaps):
                return False
            diffs = get_diffs_from_gaps(gaps)
            for d in diffs:
                if d in all_diffs:
                    return False
                all_diffs.add(d)
        return True

    # Convert best_rows (gaps) to elements
    def gaps_to_elements(gaps):
        elems = [0]
        for g in gaps:
            elems.append(elems[-1] + g)
        return elems

    # Check if best solution is valid
    if validate(best_rows):
        final_rows = [gaps_to_elements(gaps) for gaps in best_rows]
        return final_rows
    else:
        # Fallback: Return the current state if valid, else a valid greedy one
        if validate(current_rows):
            final_rows = [gaps_to_elements(gaps) for gaps in current_rows]
            return final_rows
        
        # If SA failed to find a valid solution, use the greedy one from init
        # But greedy_init might also be invalid if it was a fallback.
        # Let's generate a guaranteed valid solution with large scope.
        
        def guaranteed_valid_solution():
            rows_gaps = []
            used_diffs = set()
            for r in range(n_rows):
                gaps = [1] * num_gaps
                while True:
                    diffs = get_diffs_from_gaps(gaps)
                    if not (diffs & used_diffs):
                        break
                    # Increase all gaps to jump over conflicts
                    gaps = [g + 1 for g in gaps]
                rows_gaps.append(gaps)
                used_diffs.update(diffs)
            return rows_gaps

        safe_rows_gaps = guaranteed_valid_solution()
        final_rows = [gaps_to_elements(gaps) for gaps in safe_rows_gaps]
        return final_rows

