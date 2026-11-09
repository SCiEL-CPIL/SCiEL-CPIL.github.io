import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    def check_constraints(arr):
        """
        Returns (is_sidon, covered_diffs_set)
        arr: sorted list of distinct integers
        """
        n = len(arr)
        if n < 2:
            return True, set()
        
        diffs = set()
        is_sidon = True
        
        # For efficiency, we compute all differences and check uniqueness
        # n is at most 32, so n*(n-1)/2 <= 496 diffs.
        
        diff_list = []
        for i in range(n):
            ai = arr[i]
            for j in range(i+1, n):
                d = arr[j] - ai
                diff_list.append(d)
                if 1 <= d <= 30:
                    diffs.add(d)
        
        if len(diff_list) != len(set(diff_list)):
            is_sidon = False
            
        return is_sidon, diffs

    def get_missing_diffs(covered):
        return [d for d in range(1, 31) if d not in covered]

    def greedy_construct():
        """
        Fast greedy construction to get a valid initial solution.
        """
        a = [0]
        covered = set()
        used_diffs = set()
        
        # We need to cover 1..30
        # We will iteratively add elements to cover the smallest missing difference
        # To keep numbers small, we prefer adding small numbers.
        
        for _ in range(32): # Max size 32
            if len(covered) == 30:
                break
                
            missing = get_missing_diffs(covered)
            if not missing:
                break
                
            target_d = min(missing)
            
            # Try to add a single element x such that x +/- a_i = target_d
            # Or x - y = target_d for some existing y? No, single element addition only creates diffs with existing elements.
            # If we add x, new diffs are |x - a_i| for all a_i in a.
            # We want one of these to be target_d.
            # So x = a_i + target_d or x = a_i - target_d.
            
            candidates = set()
            for a_i in a:
                candidates.add(a_i + target_d)
                candidates.add(a_i - target_d)
            
            # Filter valid candidates
            valid_candidates = []
            for x in candidates:
                if x < 0 or x in a:
                    continue
                
                # Check if adding x violates Sidon
                # New diffs: |x - a_i| for all a_i in a
                # These must not be in used_diffs
                # Also, the new diffs themselves must be distinct (they are, since a_i are distinct)
                
                new_diffs = []
                valid = True
                for a_i in a:
                    d = abs(x - a_i)
                    if d in used_diffs:
                        valid = False
                        break
                    new_diffs.append(d)
                
                if valid:
                    # Check if new_diffs have duplicates among themselves?
                    # |x - a_i| = |x - a_j| implies x - a_i = a_j - x => 2x = a_i + a_j.
                    # If such a_i, a_j exist, then diff is duplicated.
                    # But since a is Sidon, a_i + a_j is unique for i<j? No, Sidon means differences are unique.
                    # Actually, if |x-a_i| = |x-a_j|, then a_i and a_j are symmetric around x.
                    # This creates a duplicate difference value.
                    # So we must check if new_diffs has duplicates.
                    if len(new_diffs) != len(set(new_diffs)):
                        valid = False
                    
                if valid:
                    valid_candidates.append(x)
            
            # If we found valid single-element candidates, pick the smallest one to minimize max?
            # Or pick the one that covers the most new diffs?
            # For now, pick smallest to keep diameter small.
            if valid_candidates:
                x = min(valid_candidates)
                a.append(x)
                a.sort()
                
                # Update used_diffs and covered
                # Recompute is simpler and safer for small n
                is_sidon, new_covered = check_constraints(a)
                if is_sidon:
                    covered = new_covered
                    # Update used_diffs
                    # We can recompute used_diffs from a
                    for i in range(len(a)):
                        for j in range(i+1, len(a)):
                            used_diffs.add(a[j] - a[i])
                else:
                    # Should not happen if check was correct, but rollback
                    a.pop()
                    a.sort()
            else:
                # No single element works. Try adding a pair (x, x+target_d)?
                # Or just give up and return what we have?
                # Greedy might fail to find a solution if it gets stuck.
                # Let's try a broader search for x in a limited range if single fails.
                # But for the initial seed, let's just return what we have.
                # If we don't have all 30, the SA phase will fix it.
                pass
                
        return a

    def sa_construct():
        """
        Simulated Annealing to improve the solution.
        Objective: Minimize max(arr) subject to Sidon and covering 1..30.
        """
        # Start with a valid solution if possible, else a random valid-ish one
        current_arr = greedy_construct()
        
        # Ensure we have a valid starting point for SA.
        # If greedy failed to cover all 30, we need to force a valid state.
        # Let's check if current_arr is valid.
        is_sidon, covered = check_constraints(current_arr)
        
        if not is_sidon or len(covered) < 30:
            # Fallback: Try to construct a valid one by brute force small search?
            # Or just start with a known valid set if we can find one quickly?
            # Let's try to fix it by adding random elements until valid.
            # This is risky. Let's assume greedy works for 1..30.
            # If not, we return the best we have (which might be invalid -> score 0).
            # To be safe, let's try a few random restarts of greedy.
            for _ in range(10):
                trial = greedy_construct()
                is_s, cov = check_constraints(trial)
                if is_s and len(cov) == 30:
                    current_arr = trial
                    break
        
        if len(current_arr) == 0:
            current_arr = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
            # This is not Sidon.
            # Let's just use a simple valid Sidon set that covers 1..30?
            # [0, 1, 4, 10, 12, 17, 21, 25, 30, 36, 42, 49, 57, 66, 75, 85, 96, 108, 121, 135, 150, 166, 183, 201, 220, 240, 261, 283, 306, 330, 355, 381]
            # This is too big.
            # Let's rely on the greedy. If it fails, we return a dummy valid set?
            # Actually, if greedy fails, we can try to repair it.
            pass

        # Get initial state
        current_max = max(current_arr)
        current_is_sidon, current_covered = check_constraints(current_arr)
        
        # If not valid, we must make it valid before optimizing max.
        # Penalty method: Score = max(arr) + 1000 * (missing_diffs_count) + 1000 * (sidon_violations)
        
        def get_score(arr):
            if not arr:
                return float('inf')
            n = len(arr)
            if n > 32:
                return float('inf')
            if 0 not in arr:
                return float('inf')
            
            diffs = set()
            is_sidon = True
            diff_list = []
            for i in range(n):
                ai = arr[i]
                for j in range(i+1, n):
                    d = arr[j] - ai
                    diff_list.append(d)
                    if 1 <= d <= 30:
                        diffs.add(d)
            
            if len(diff_list) != len(set(diff_list)):
                is_sidon = False
            
            missing = 30 - len(diffs)
            violations = 0 if is_sidon else 1
            
            # Heavier penalty for invalidity
            score = max(arr) + 10000 * missing + 10000 * violations
            return score

        current_score = get_score(current_arr)
        best_arr = current_arr[:]
        best_score = current_score
        
        # SA Parameters
        T = 100.0
        cooling_rate = 0.999
        min_T = 0.1
        
        while T > min_T:
            if time.time() > deadline:
                break
            
            # Generate neighbor
            new_arr = current_arr[:]
            
            # Random move
            move_type = random.randint(0, 2)
            
            if move_type == 0 and len(new_arr) > 1:
                # Remove a random element (except 0)
                idx = random.randint(1, len(new_arr)-1)
                new_arr.pop(idx)
                
            elif move_type == 1 and len(new_arr) < 32:
                # Add a random element
                # Bias towards small numbers
                max_val = max(new_arr) if new_arr else 0
                # Try to add near existing elements or small numbers
                if random.random() < 0.5:
                    # Near existing
                    base = random.choice(new_arr)
                    offset = random.randint(-10, 10)
                    x = base + offset
                else:
                    # Small random
                    x = random.randint(1, max_val + 10)
                
                if x > 0 and x not in new_arr:
                    new_arr.append(x)
                    new_arr.sort()
                else:
                    # Try again or skip
                    new_arr = current_arr[:]
                    
            else:
                # Shift a random element
                if len(new_arr) > 1:
                    idx = random.randint(1, len(new_arr)-1)
                    old_val = new_arr[idx]
                    new_val = old_val + random.randint(-5, 5)
                    if new_val > 0 and new_val not in new_arr:
                        new_arr[idx] = new_val
                        new_arr.sort()
                    else:
                        new_arr = current_arr[:]

            # Evaluate
            new_score = get_score(new_arr)
            
            delta = new_score - current_score
            
            if delta < 0 or random.random() < np.exp(-delta / T):
                current_arr = new_arr
                current_score = new_score
                
                if current_score < best_score:
                    best_score = current_score
                    best_arr = current_arr[:]
            
            T *= cooling_rate
        
        return best_arr

    # Run SA
    best_solution = sa_construct()
    
    # Final validation
    is_sidon, covered = check_constraints(best_solution)
    if not is_sidon or len(covered) < 30:
        # Fallback to greedy if SA failed to maintain validity
        greedy_sol = greedy_construct()
        g_is_sidon, g_covered = check_constraints(greedy_sol)
        if g_is_sidon and len(g_covered) == 30:
            best_solution = greedy_sol
    
    return best_solution
# EVOLVE-BLOCK-END
