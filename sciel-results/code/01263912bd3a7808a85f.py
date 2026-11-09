import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 110  # Safety margin for 120s limit

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(subset):
        # Check if a set is sum-free
        s = set(subset)
        for a in s:
            for b in s:
                if a + b in s:
                    return False
        return True

    def max_sum_free_subset(T, rng, restarts=200):
        """
        Find a large sum-free subset of T using greedy with random restarts.
        """
        if not T:
            return set()
        
        best_size = 0
        best_set = set()
        
        # If T is very small, we can be more exhaustive or just do many restarts
        # Greedy: add elements in random order if valid
        
        for _ in range(restarts):
            current_set = set()
            shuffled_T = list(T)
            rng.shuffle(shuffled_T)
            
            for x in shuffled_T:
                # Check if adding x keeps it sum-free
                # We need to check:
                # 1. x + y != z for y, z in current_set
                # 2. x + x != y for y in current_set
                # 3. x + y != x (implies y=0, but 0 is not in S)
                
                valid = True
                # Optimization: just check pairs involving x
                for y in current_set:
                    if x + y in current_set:
                        valid = False
                        break
                    if x + y == x: # y=0, impossible
                        valid = False
                        break
                if valid and x + x in current_set:
                    valid = False
                
                if valid:
                    current_set.add(x)
            
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
        
        return best_set

    def solve_instance(S):
        O_all = {x for x in S if x % 2 != 0}
        E_all = {x for x in S if x % 2 == 0}
        
        if not O_all:
            # If no odds, we just need max sum-free subset of E_all
            # But E_all is even numbers. Sum of two evens is even.
            # We can scale down by 2? No, just solve directly.
            return max_sum_free_subset(E_all, random.Random(42), restarts=500)
            
        # Start with all odds
        O_curr = set(O_all)
        best_solution = set(O_curr)
        
        # Helper to compute F_O for a given O
        def get_F_O(O):
            # E_safe: e in E_all such that e not in {o1+o2 | o1,o2 in O}
            sums_O = set()
            for o1 in O:
                for o2 in O:
                    sums_O.add(o1 + o2)
            
            # F_O: e in E_safe such that for all o in O, o+e not in O
            F_O = set()
            for e in E_all:
                if e in sums_O:
                    continue
                valid_cross = True
                for o in O:
                    if o + e in O:
                        valid_cross = False
                        break
                if valid_cross:
                    F_O.add(e)
            return F_O

        # Initial solution
        F_O_curr = get_F_O(O_curr)
        E_curr = max_sum_free_subset(F_O_curr, random.Random(42), restarts=100)
        curr_solution = O_curr | E_curr
        if len(curr_solution) > len(best_solution):
            best_solution = curr_solution

        # Local search: try removing odds to allow more evens
        improved = True
        while improved and (time.time() - start_time) < timeout:
            improved = False
            # Try removing each odd
            for o in list(O_curr):
                O_trial = O_curr - {o}
                F_O_trial = get_F_O(O_trial)
                E_trial = max_sum_free_subset(F_O_trial, random.Random(42), restarts=50)
                trial_solution = O_trial | E_trial
                
                if len(trial_solution) > len(best_solution):
                    best_solution = trial_solution
                    O_curr = O_trial
                    E_curr = E_trial
                    improved = True
                    # Don't break, keep trying to remove more? 
                    # Or break and restart loop to re-evaluate all?
                    # Restarting is safer to ensure we don't miss better combinations.
                    break
        
        return best_solution

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        # Use a dedicated RNG for the solver to ensure reproducibility if needed, 
        # but random.Random(42) inside is fine.
        subset = solve_instance(S)
        results.append(sorted(list(subset)))
        
    return results
