"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    deadline = time.time() + 115
    random.seed(42)
    
    # Objective: minimize max element subject to:
    # 1. Set contains 0
    # 2. All positive differences are distinct (Sidon)
    # 3. All d in 1..30 appear as differences
    # 4. At most 32 elements, all in 0..10^6
    
    # Simulated annealing over a fixed-size candidate set
    # We work with a set of size n (start with 10, which gives 45 diffs, enough for 30)
    
    best_result = None
    best_max = float('inf')
    
    def evaluate(elements):
        """Return (valid, max_val, num_conflicts, num_missing) for a sorted list with elements[0]=0."""
        n = len(elements)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(elements[j] - elements[i])
        
        # Check Sidon (all diffs distinct)
        diff_set = set(diffs)
        num_conflicts = len(diffs) - len(diff_set)
        
        # Check coverage of 1..30
        num_missing = sum(1 for d in range(1, 31) if d not in diff_set)
        
        max_val = elements[-1]
        
        # Valid if no conflicts and no missing
        valid = (num_conflicts == 0 and num_missing == 0)
        
        return valid, max_val, num_conflicts, num_missing
    
    def cost(elements):
        """Penalty-based cost: large if invalid, otherwise max_val."""
        n = len(elements)
        diffs = []
        for i in range(n):
            for j in range(i + 1, n):
                diffs.append(elements[j] - elements[i])
        
        diff_set = set(diffs)
        num_conflicts = len(diffs) - len(diff_set)
        num_missing = sum(1 for d in range(1, 31) if d not in diff_set)
        
        # Heavy penalty for violations
        penalty = num_conflicts * 10000 + num_missing * 5000
        return elements[-1] + penalty
    
    def make_initial(n, max_bound):
        """Create a random initial set of size n containing 0."""
        # Start with 0, then add random elements
        elems = [0]
        for _ in range(n - 1):
            x = random.randint(1, max_bound)
            while x in elems:
                x = random.randint(1, max_bound)
            elems.append(x)
        elems.sort()
        return elems
    
    def perturb(elements, max_bound):
        """Make a small random change to the set."""
        elems = list(elements)
        op = random.random()
        
        if op < 0.6:
            # Move one element (not 0)
            idx = random.randint(1, len(elems) - 1)
            delta = random.randint(-50, 50)
            new_val = elems[idx] + delta
            if new_val > 0 and new_val < max_bound and new_val not in elems:
                elems[idx] = new_val
                elems.sort()
        elif op < 0.8:
            # Replace one element (not 0) with a new random one
            idx = random.randint(1, len(elems) - 1)
            new_val = random.randint(1, max_bound)
            if new_val not in elems:
                elems[idx] = new_val
                elems.sort()
        else:
            # Swap two non-zero elements
            if len(elems) >= 3:
                i, j = random.sample(range(1, len(elems)), 2)
                elems[i], elems[j] = elems[j], elems[i]
                elems.sort()
        
        return elems
    
    # Try different sizes
    for n in [9, 10, 11, 12]:
        if time.time() > deadline - 5:
            break
        
        # Upper bound for search: start generous, tighten as we find solutions
        # Lower bound: need max >= 30 (to cover diff 30)
        # For n elements, min max is roughly n^2/2 for Golomb rulers
        upper_bound = n * n * 2  # Generous start
        
        # Multiple restarts with annealing
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            elems = make_initial(n, upper_bound)
            current_cost = cost(elems)
            
            T = 10000.0
            T_min = 1.0
            cooling = 0.995
            
            steps = 0
            max_steps = 5000
            
            while T > T_min and steps < max_steps:
                if time.time() > deadline - 5:
                    break
                
                new_elems = perturb(elems, upper_bound)
                new_cost = cost(new_elems)
                
                delta_cost = new_cost - current_cost
                
                if delta_cost < 0 or random.random() < pow(2.718281828, -delta_cost / T):
                    elems = new_elems
                    current_cost = new_cost
                    
                    # Check if we have a valid solution
                    valid, max_val, nc, nm = evaluate(elems)
                    if valid and max_val < best_max:
                        best_max = max_val
                        best_result = list(elems)
                        # Tighten upper bound
                        upper_bound = max_val - 1
                        if upper_bound < 30:
                            upper_bound = 30
                
                T *= cooling
                steps += 1
    
    if best_result is not None:
        return best_result
    
    # Fallback: deterministic greedy
    a = [0]
    for d in range(1, 31):
        found = False
        for x in range(1, 100000):
            trial = sorted(a + [x])
            diffs = [trial[j] - trial[i] for i in range(len(trial)) for j in range(i+1, len(trial))]
            if len(set(diffs)) == len(diffs) and d in diffs:
                a = trial
                found = True
                break
        if not found:
            # Force: add x such that x - 0 = d if possible, else x - some element
            x = d
            while x in a or any((x - ai) in [a[j]-a[i] for i in range(len(a)) for j in range(i+1,len(a))] for ai in a):
                x += 1
            a = sorted(a + [x])
    
    return a
# EVOLVE-BLOCK-END
