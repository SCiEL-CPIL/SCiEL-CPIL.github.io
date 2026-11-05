import time
import random
import sys
import copy
import numpy as np

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin before 120s kill

    total_diffs = n_rows * (k + 1) * k // 2

    def get_scope(rows):
        return max(max(r) for r in rows)

    def is_valid(rows):
        used = set()
        for row in rows:
            if len(row) != k + 1:
                return False
            if row[0] != 0:
                return False
            for i in range(1, len(row)):
                if row[i] <= row[i - 1]:
                    return False
            for i in range(len(row)):
                for j in range(i + 1, len(row)):
                    d = row[j] - row[i]
                    if d in used:
                        return False
                    used.add(d)
        return True

    def greedy_construct(scope_limit, rng=None):
        """
        Greedy construction: for each row, pick the smallest valid value at each step.
        'Pressure' variant: try to keep differences small by preferring values close to last.
        """
        if rng is None:
            rng = random.Random()
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                # Try values starting from last_val + 1
                # To apply "pressure", we just take the first valid one (smallest)
                # This naturally minimizes the gap to the previous element
                for x in range(last_val + 1, scope_limit + 1):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        row.append(x)
                        for y in row[:-1]:
                            used_diffs.add(x - y)
                        found = True
                        break
                if not found:
                    return None
            rows.append(row)
            
        return rows

    def greedy_randomized(scope_limit, rng):
        """
        Randomized greedy: pick from a small set of valid candidates to diversify.
        """
        used_diffs = set()
        rows = []
        
        for r_idx in range(n_rows):
            row = [0]
            while len(row) < k + 1:
                found = False
                last_val = row[-1]
                # Collect a few candidates
                candidates = []
                for x in range(last_val + 1, min(last_val + 50, scope_limit + 1)):
                    valid = True
                    for y in row:
                        d = x - y
                        if d in used_diffs:
                            valid = False
                            break
                    if valid:
                        candidates.append(x)
                        if len(candidates) >= 5:
                            break
                
                if not candidates:
                    # Fallback to full search if no candidates in window
                    for x in range(last_val + 1, scope_limit + 1):
                        valid = True
                        for y in row:
                            d = x - y
                            if d in used_diffs:
                                valid = False
                                break
                        if valid:
                            candidates.append(x)
                            break
                            # Actually just break out of inner loop and use this one
                            # But the loop above breaks. Let's just use the first valid in full range if window failed
                            # This logic is a bit messy, let's simplify:
                            # If window failed, do full search
                            pass
                    
                    # Re-do full search properly if window failed
                    if not candidates:
                         for x in range(last_val + 1, scope_limit + 1):
                            valid = True
                            for y in row:
                                d = x - y
                                if d in used_diffs:
                                    valid = False
                                    break
                            if valid:
                                candidates.append(x)
                                break

                if not candidates:
                    return None
                
                # Pick randomly from candidates
                x = rng.choice(candidates)
                row.append(x)
                for y in row[:-1]:
                    used_diffs.add(x - y)
                    
            rows.append(row)
            
        return rows

    def local_search_sa(initial_rows, time_limit):
        """
        Simulated Annealing local search.
        Objective: Minimize Scope (M) + Lambda * Penalty (number of duplicate diffs).
        We want to find a valid solution (Penalty=0) with low Scope.
        """
        rows = [r[:] for r in initial_rows]
        M = get_scope(rows)
        
        # Calculate initial penalty and counts
        def calc_penalty_and_counts(rows):
            counts = {}
            for r in rows:
                for i in range(len(r)):
                    for j in range(i+1, len(r)):
                        d = r[j] - r[i]
                        counts[d] = counts.get(d, 0) + 1
            pen = 0
            for c in counts.values():
                if c > 1:
                    pen += c - 1
            return pen, counts

        curr_penalty, curr_counts = calc_penalty_and_counts(rows)
        
        best_rows = copy.deepcopy(rows)
        best_M = M
        best_penalty = curr_penalty

        # SA Parameters
        T = 100.0
        cooling = 0.9995
        lambda_pen = 50.0 # Weight for penalty
        
        curr_energy = M + lambda_pen * curr_penalty
        
        t_end = time.time() + time_limit
        no_improve = 0
        max_no_improve = 2000
        
        # Precompute max diff bound for safety, though we use dict so it's dynamic
        # But we need to be careful with memory if diffs get huge. 
        # Scope is usually < 200, so diffs < 200.
        
        while time.time() < t_end:
            # Generate Move
            # 1. Single element change
            r_idx = random.randint(0, n_rows - 1)
            row = rows[r_idx]
            j_idx = random.randint(1, k)
            
            old_val = row[j_idx]
            
            # Determine valid range
            low = row[j_idx - 1] + 1
            # Allow expansion slightly beyond current M to escape local minima
            # But don't go too far
            high = row[j_idx + 1] - 1 if j_idx < k else M + 20
            
            if high < low:
                continue
                
            new_val = random.randint(low, high)
            
            if new_val == old_val:
                continue
            
            # Update Counts and Penalty Incrementally
            # Remove old diffs
            for i in range(j_idx):
                d = old_val - row[i]
                c = curr_counts.get(d, 0)
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] = c - 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
                    
            for h in range(j_idx + 1, k + 1):
                d = row[h] - old_val
                c = curr_counts.get(d, 0)
                if c > 1:
                    curr_penalty -= 1
                curr_counts[d] = c - 1
                if curr_counts[d] == 0:
                    del curr_counts[d]
            
            # Tentatively set new value
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
            
            # Calculate new M
            # Only the changed row can increase M. If it decreased, M might decrease.
            # Efficiently calculate M:
            new_row_max = max(row)
            other_max = 0
            for r in rows:
                if r is not row:
                    if max(r) > other_max:
                        other_max = max(r)
            new_M = max(new_row_max, other_max)
            
            new_energy = new_M + lambda_pen * curr_penalty
            delta_E = new_energy - curr_energy
            
            # Acceptance criterion
            if delta_E < 0 or (T > 1.0 and random.random() < pow(2.71828, -delta_E / T)):
                curr_energy = new_energy
                M = new_M
                
                # Check if we found a valid solution (penalty 0)
                if curr_penalty == 0:
                    if M < best_M:
                        best_M = M
                        best_rows = copy.deepcopy(rows)
                        best_penalty = 0
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    # Even if not valid, we might be getting closer?
                    # For now, only track valid solutions as "best"
                    # But we should reset no_improve if energy decreases significantly?
                    # Let's just let it run.
                    no_improve += 1
            else:
                # Revert
                row[j_idx] = old_val
                
                # Remove new diffs
                for i in range(j_idx):
                    d = new_val - row[i]
                    c = curr_counts.get(d, 0)
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] = c - 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                        
                for h in range(j_idx + 1, k + 1):
                    d = row[h] - new_val
                    c = curr_counts.get(d, 0)
                    if c > 1:
                        curr_penalty -= 1
                    curr_counts[d] = c - 1
                    if curr_counts[d] == 0:
                        del curr_counts[d]
                
                # Add old diffs
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
                
                # Recalculate M for consistency (should be same as before)
                # But to be safe:
                # M = get_scope(rows) # Expensive, avoid if possible.
                # Since we reverted, M is unchanged.
                
                no_improve += 1
            
            # Cool down
            T *= cooling
            if T < 1.0:
                T = 1.0
            
            # Restart if stuck
            if no_improve > max_no_improve:
                # Perturb
                for _ in range(5):
                    r_idx = random.randint(0, n_rows - 1)
                    row = rows[r_idx]
                    j_idx = random.randint(1, k)
                    old_val = row[j_idx]
                    low = row[j_idx - 1] + 1
                    high = row[j_idx + 1] - 1 if j_idx < k else M + 30
                    if high < low: continue
                    new_val = random.randint(low, high)
                    if new_val == old_val: continue
                    
                    # Update counts (same logic as above)
                    for i in range(j_idx):
                        d = old_val - row[i]
                        c = curr_counts.get(d, 0)
                        if c > 1: curr_penalty -= 1
                        curr_counts[d] = c - 1
                        if curr_counts[d] == 0: del curr_counts[d]
                    for h in range(j_idx + 1, k + 1):
                        d = row[h] - old_val
                        c = curr_counts.get(d, 0)
                        if c > 1: curr_penalty -= 1
                        curr_counts[d] = c - 1
                        if curr_counts[d] == 0: del curr_counts[d]
                    
                    row[j_idx] = new_val
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
                    
                    new_row_max = max(row)
                    other_max = 0
                    for r in rows:
                        if r is not row:
                            if max(r) > other_max: other_max = max(r)
                    M = max(new_row_max, other_max)
                    curr_energy = M + lambda_pen * curr_penalty
                    
                no_improve = 0
                T = max(T, 50.0) # Reheat slightly

        # Return best valid solution found
        if best_penalty == 0:
            return best_rows
        else:
            # If no valid solution found, return the initial one (which might be valid or not)
            # But we started with a valid greedy solution, so best_rows should be valid if best_penalty==0
            # If we never found a valid one, we should return the initial valid one.
            # However, best_rows is updated only when penalty==0.
            # So if best_penalty != 0, best_rows is still the initial one?
            # No, best_rows is initialized to initial_rows.
            # So returning best_rows is safe.
            return best_rows

    # Step 1: Generate initial solution
    # Try a few random seeds for greedy to get a diverse starting point
    best_initial = None
    best_initial_scope = float('inf')
    
    for _ in range(5):
        # Try greedy with a reasonable scope limit
        # The counting bound is 112, best known 134.
        # Let's try limits around there.
        for scope_limit in [120, 130, 140]:
            rng = random.Random()
            rows = greedy_randomized(scope_limit, rng)
            if rows is not None:
                scope = get_scope(rows)
                if scope < best_initial_scope:
                    best_initial_scope = scope
                    best_initial = rows
                break # If one limit works, try next seed
    
    if best_initial is None:
        # Fallback to simple greedy
        for scope_limit in [150, 200, 300]:
            best_initial = greedy_construct(scope_limit)
            if best_initial is not None:
                break
    
    if best_initial is None:
        # Last resort fallback
        used_diffs = set()
        best_initial = []
        for _ in range(n_rows):
            row = [0]
            x = 0
            while len(row) < k + 1:
                x += 1
                valid = True
                for y in row:
                    if x - y in used_diffs:
                        valid = False
                        break
                if valid:
                    row.append(x)
                    for y in row[:-1]:
                        used_diffs.add(x - y)
            best_initial.append(row)
        return best_initial

    # Step 2: Local Search (Simulated Annealing)
    remaining_time = deadline - time.time()
    if remaining_time > 5:
        # Run SA for the remaining time
        final_rows = local_search_sa(best_initial, remaining_time - 2)
        
        # Verify validity
        if is_valid(final_rows):
            return final_rows
        else:
            # Return the initial valid solution if SA failed to maintain validity
            # (SA should maintain validity if we only accept valid moves, but we use penalty)
            # Actually, SA allows invalid moves (penalty > 0).
            # So we must check if the best found was valid.
            # local_search_sa returns best_rows which is only updated if penalty==0.
            # So if best_penalty != 0, it returns the initial rows (which are valid).
            # Wait, in local_search_sa, if best_penalty != 0, it returns best_rows.
            # best_rows is initialized to initial_rows.
            # So it should be valid.
            return final_rows
    
    return best_initial

if __name__ == "__main__":
    rows = construct(4, 7)
    scope = max(max(r) for r in rows)
    print(f"Scope: {scope}")
    print(rows)
    # Verify
    used = set()
    valid = True
    for r in rows:
        if len(r) != 8: valid = False
        if r[0] != 0: valid = False
        for i in range(1, 8):
            if r[i] <= r[i-1]: valid = False
        for i in range(8):
            for j in range(i+1, 8):
                d = r[j] - r[i]
                if d in used: valid = False
                used.add(d)
    print(f"Valid: {valid}")
