import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s limit, leaving 5s margin
    
    max_val = 4096
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(x, current_set, sum_set):
        new_sums = [3 * x]
        for y in current_set:
            new_sums.append(2 * x + y)
            new_sums.append(x + 2 * y)
        for y, z in combinations(current_set, 2):
            new_sums.append(x + y + z)
        
        new_sums_set = set(new_sums)
        if len(new_sums_set) != len(new_sums):
            return False, None
        if not new_sums_set.isdisjoint(sum_set):
            return False, None
        return True, new_sums_set

    def run_greedy(candidates):
        current_set = []
        sum_set = set()
        
        for x in candidates:
            valid, new_sums = is_valid_addition(x, current_set, sum_set)
            if valid:
                current_set.append(x)
                sum_set.update(new_sums)
                
        return current_set, sum_set

    # 1. Deterministic Increasing Greedy
    s, ss = run_greedy(range(1, max_val + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = s
        best_sum_set = ss

    # 2. Randomized Greedy
    rng = random.Random(42)
    # Run random passes until time is low
    for i in range(100):
        if time.time() > deadline - 15.0: # Leave 15s for local search
            break
            
        candidates = list(range(1, max_val + 1))
        rng.shuffle(candidates)
        
        s, ss = run_greedy(candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            best_sum_set = ss

    # 3. Local Search (Random Walk with Add Attempts)
    if time.time() < deadline - 2.0:
        S = list(best_set)
        sum_set = best_sum_set
        size = len(S)
        
        # Helper to remove element
        def remove_element(x, S, sum_set):
            to_remove = set()
            to_remove.add(3*x)
            for y in S:
                if y == x: continue
                to_remove.add(2*x + y)
                to_remove.add(x + 2*y)
            for y, z in combinations(S, 2):
                if y == x or z == x: continue
                to_remove.add(x + y + z)
            sum_set.difference_update(to_remove)
            S.remove(x)

        # Helper to add element
        def add_element(x, S, sum_set):
            new_sums = [3 * x]
            for y in S:
                new_sums.append(2 * x + y)
                new_sums.append(x + 2 * y)
            for y, z in combinations(S, 2):
                new_sums.append(x + y + z)
            
            new_sums_set = set(new_sums)
            if len(new_sums_set) != len(new_sums):
                return False
            if not new_sums_set.isdisjoint(sum_set):
                return False
            
            sum_set.update(new_sums_set)
            S.append(x)
            return True

        # Random Walk
        # We will perform swaps that preserve validity, and periodically try to add.
        # If we add, we record the best size.
        
        current_size = len(S)
        best_walk_set = list(S)
        best_walk_sum_set = set(sum_set)
        
        # To speed up, we can precompute the set of elements in S
        S_set = set(S)
        
        # Number of steps
        steps = 0
        max_steps = 20000 # Adjust based on speed
        
        while time.time() < deadline - 1.0:
            # Try to add a random element
            if rng.random() < 0.2: # 20% chance to try add
                c = rng.randint(1, max_val)
                if c not in S_set:
                    # Backup
                    S_backup = list(S)
                    sum_set_backup = set(sum_set)
                    S_set_backup = set(S_set)
                    
                    if add_element(c, S, sum_set):
                        S_set.add(c)
                        current_size += 1
                        if current_size > best_size:
                            best_size = current_size
                            best_set = list(S)
                            best_sum_set = set(sum_set)
                            best_walk_set = list(S)
                            best_walk_sum_set = set(sum_set)
                        # Reset walk? Or continue? Continue.
                    else:
                        # Revert
                        S = S_backup
                        sum_set = sum_set_backup
                        S_set = S_set_backup
            else:
                # Try a swap
                if len(S) == 0:
                    break
                a = rng.choice(S)
                b = rng.randint(1, max_val)
                if b in S_set:
                    continue
                
                # Backup
                S_backup = list(S)
                sum_set_backup = set(sum_set)
                S_set_backup = set(S_set)
                
                # Remove a
                remove_element(a, S, sum_set)
                S_set.remove(a)
                
                # Add b
                if add_element(b, S, sum_set):
                    S_set.add(b)
                    # Accept swap
                    steps += 1
                else:
                    # Revert
                    S = S_backup
                    sum_set = sum_set_backup
                    S_set = S_set_backup

    return best_set
