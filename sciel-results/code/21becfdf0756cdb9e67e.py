"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def max_sidon_subset(s):
    """Compute the largest Sidon subset of s using backtracking with pruning."""
    s = sorted(s)
    n = len(s)
    best = 0
    
    def backtrack(start, current_len, sums):
        nonlocal best
        if current_len > best:
            best = current_len
        if start >= n:
            return
        # Pruning
        if current_len + (n - start) <= best:
            return
        for i in range(start, n):
            x = s[i]
            # Check x+x
            if x + x in sums:
                continue
            # Check x+y for all y in current (we don't track current, just sums)
            # Actually we need to check against existing sums
            valid = True
            # We need to know which elements are in current to check x+y
            # This approach doesn't work without tracking current. Let me restructure.
            pass
    
    # Better: track current list
    best = 0
    def backtrack(start, current, sums):
        nonlocal best
        if len(current) > best:
            best = len(current)
        if start >= n:
            return
        if len(current) + (n - start) <= best:
            return
        for i in range(start, n):
            x = s[i]
            # Check all sums x+y for y in current, and x+x
            ok = True
            for y in current:
                if x + y in sums:
                    ok = False
                    break
            if ok and x + x in sums:
                ok = False
            if not ok:
                continue
            new_sums = sums.copy()
            new_sums.add(x + x)
            for y in current:
                new_sums.add(x + y)
            backtrack(i + 1, current + [x], new_sums)
    
    backtrack(0, [], set())
    return best

def construct():
    random.seed(123)
    
    # Start with a promising initial set
    # Use clusters to create many sum collisions
    initial = [0, 1, 2, 3, 20, 21, 22, 23, 40, 41, 42, 43]
    
    best_set = initial[:]
    best_val = max_sidon_subset(best_set)
    
    start_time = time.time()
    timeout = 50
    
    current = initial[:]
    current_val = best_val
    
    while time.time() - start_time < timeout:
        candidate = current[:]
        move_type = random.random()
        
        if move_type < 0.4:
            # Swap two elements
            i, j = random.sample(range(12), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]
        elif move_type < 0.7:
            # Change one element
            i = random.randint(0, 11)
            old = candidate[i]
            new = random.randint(0, 128)
            if new != old and new not in candidate:
                candidate[i] = new
            else:
                continue
        else:
            # Move one element to a nearby value
            i = random.randint(0, 11)
            old = candidate[i]
            delta = random.randint(-5, 5)
            new = old + delta
            if 0 <= new <= 128 and new != old and new not in candidate:
                candidate[i] = new
            else:
                continue
        
        if len(set(candidate)) != 12:
            continue
        
        val = max_sidon_subset(candidate)
        
        if val < current_val:
            current = candidate
            current_val = val
            if val < best_val:
                best_val = val
                best_set = candidate[:]
        elif val == current_val:
            if random.random() < 0.2:
                current = candidate
    
    return best_set
# EVOLVE-BLOCK-END
