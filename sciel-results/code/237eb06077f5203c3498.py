"""New program for Sidon set with differences 1..30 using Deterministic Backtracking and Simulated Annealing."""
import random
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a Sidon set containing 0 with all differences 1..30.
    Minimize the largest element.
    
    Strategy:
    1. Use deterministic backtracking to find a valid Sidon set covering 1..30 quickly.
    2. Use Simulated Annealing to minimize the largest element.
    """
    target_diffs = set(range(1, 31))
    
    def is_valid_sidon(s):
        """Check if set s is a Sidon set."""
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                d = abs(s[j] - s[i])
                if d in diffs:
                    return False
                diffs.add(d)
        return True

    def get_missing(s):
        """Get missing differences from 1..30."""
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(abs(s[j] - s[i]))
        return target_diffs - diffs

    def backtrack(current_s, missing, depth=0):
        """
        Recursively add the smallest valid element to current_s until missing is empty.
        Returns a valid set or None.
        """
        if not missing:
            return list(current_s)
        
        if len(current_s) >= 32:
            return None
            
        if depth > 50: # Prevent infinite recursion if stuck
            return None

        # Try candidates from 1 upwards
        # Optimization: Only try x that covers at least one missing diff
        max_val = max(current_s) if current_s else 0
        
        # We can limit the search space for x
        # If max_val is 0, x can be 1..30
        # If max_val > 0, x can be 1..max_val+30
        # Let's just scan 1..200 for the first few steps
        
        candidates = []
        for x in range(1, 200):
            # Check if x covers any missing diff
            covers = False
            for s_i in current_s:
                d = abs(x - s_i)
                if d in missing:
                    covers = True
                    break
            if not covers:
                continue
                
            # Check Sidon property
            trial = sorted(current_s + [x])
            if is_valid_sidon(trial):
                candidates.append(x)
                
        if not candidates:
            # If no candidate in 1..200, try higher?
            # For this problem, 200 should be enough for initialization.
            return None
            
        # Try the smallest candidate first
        x = candidates[0]
        trial = sorted(current_s + [x])
        new_missing = get_missing(trial)
        
        res = backtrack(trial, new_missing, depth + 1)
        if res:
            return res
            
        # If smallest failed, try next few
        for x in candidates[1:5]:
            trial = sorted(current_s + [x])
            new_missing = get_missing(trial)
            res = backtrack(trial, new_missing, depth + 1)
            if res:
                return res
                
        return None

    # 1. Find initial solution
    start_time = time.time()
    initial_s = backtrack([0], set(target_diffs))
    
    if initial_s is None:
        # Fallback: simple greedy
        s = [0]
        missing = set(target_diffs)
        while missing and len(s) < 32:
            found = False
            for x in range(1, 500):
                trial = sorted(s + [x])
                if is_valid_sidon(trial):
                    new_missing = get_missing(trial)
                    if len(new_missing) < len(missing):
                        s = trial
                        missing = new_missing
                        found = True
                        break
            if not found:
                break
        initial_s = s

    if not initial_s or get_missing(initial_s):
        return sorted(initial_s) if initial_s else [0]

    # 2. Simulated Annealing to minimize max element
    current_s = sorted(initial_s)
    best_s = current_s
    best_max = max(current_s)
    
    T = 100.0
    cooling_rate = 0.99
    
    # Run SA for a fixed number of iterations or time
    max_iters = 50000
    for i in range(max_iters):
        if time.time() - start_time > 110: # Leave 10s buffer
            break
            
        # Pick a random element (not 0)
        idx = random.randint(1, len(current_s) - 1)
        old_val = current_s[idx]
        
        # Try to move to a new position
        # Bias towards smaller values to help shrink
        # Range: 1 to current_max
        new_val = random.randint(1, max(1, max(current_s)))
        
        if new_val == old_val or new_val in current_s:
            continue
            
        new_s = sorted([v for v in current_s if v != old_val] + [new_val])
        
        # Check validity
        if not is_valid_sidon(new_s):
            continue
        if get_missing(new_s):
            continue
            
        new_max = max(new_s)
        
        # Acceptance criteria
        if new_max < best_max:
            best_s = new_s
            best_max = new_max
            current_s = new_s
        elif new_max < max(current_s):
            current_s = new_s
        else:
            # Accept worse solution with probability exp(-delta / T)
            delta = new_max - max(current_s)
            if random.random() < (2.71828 ** (-delta / T)):
                current_s = new_s
                
        T *= cooling_rate
        
    return sorted(best_s)
# EVOLVE-BLOCK-END
