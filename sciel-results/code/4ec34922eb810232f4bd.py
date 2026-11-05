import time
import random
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst, 1)

def _is_valid_subset(selected):
    if len(selected) < 2:
        return True
    
    lcm_val = _lcm_list(selected)
    vals = [lcm_val // x for x in selected]
    n = len(selected)
    
    for i in range(n):
        target = vals[i]
        others = [vals[j] for j in range(n) if j != i]
        m = len(others)
        if m == 0:
            continue
        
        half = m // 2
        left = others[:half]
        right = others[half:]
        
        left_sums = {0}
        for mask in range(1, 1 << len(left)):
            s = 0
            for b in range(len(left)):
                if mask & (1 << b):
                    s += left[b]
            left_sums.add(s)
            
        right_sums = {0}
        for mask in range(1, 1 << len(right)):
            s = 0
            for b in range(len(right)):
                if mask & (1 << b):
                    s += right[b]
            right_sums.add(s)
            
        if len(left_sums) < len(right_sums):
            for l in left_sums:
                if target - l in right_sums:
                    return False
        else:
            for r in right_sums:
                if target - r in left_sums:
                    return False
                    
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    all_elements = list(range(1, 25))
    
    # Greedy initialization
    def greedy_fill(order):
        s = []
        for x in order:
            trial = s + [x]
            if _is_valid_subset(trial):
                s = trial
        return s
    
    best_set = []
    
    # Try sorted
    s1 = greedy_fill(all_elements)
    if len(s1) > len(best_set):
        best_set = s1
        
    # Try reverse sorted
    s2 = greedy_fill(sorted(all_elements, reverse=True))
    if len(s2) > len(best_set):
        best_set = s2
        
    # Try random orders
    for _ in range(20):
        if time.time() > deadline - 10:
            break
        random.shuffle(all_elements)
        s = greedy_fill(all_elements)
        if len(s) > len(best_set):
            best_set = s[:]
            
    current = best_set[:]
    best_size = len(current)
    
    # Local Search
    def try_add(x):
        nonlocal current
        if x in current:
            return False
        trial = current + [x]
        if _is_valid_subset(trial):
            current = trial
            return True
        return False

    def try_swap(x_out, x_in):
        nonlocal current
        if x_out not in current or x_in in current:
            return False
        trial = [x for x in current if x != x_out] + [x_in]
        if _is_valid_subset(trial):
            current = trial
            return True
        return False

    iterations = 0
    stuck_count = 0
    
    while time.time() < deadline:
        iterations += 1
        
        available = [x for x in all_elements if x not in current]
        
        # 1. Try to add a random element
        if available:
            x = random.choice(available)
            if try_add(x):
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
                stuck_count = 0
                continue
            else:
                stuck_count += 1
                
        # 2. If we couldn't add, try to swap
        if current and available:
            x_out = random.choice(current)
            # Try a few swaps
            success = False
            for _ in range(10):
                if time.time() > deadline:
                    break
                x_in = random.choice(available)
                if try_swap(x_out, x_in):
                    # Swap succeeded. Check if we can add now
                    new_available = [x for x in all_elements if x not in current]
                    if new_available:
                        x_add = random.choice(new_available)
                        if try_add(x_add):
                            if len(current) > best_size:
                                best_size = len(current)
                                best_set = current[:]
                            stuck_count = 0
                            success = True
                            break
                    # If swap succeeded but no add, we are at same size.
                    # We might accept this swap to escape local optima?
                    # Let's say we accept it with some probability or if stuck_count is high.
                    if stuck_count > 20:
                        stuck_count = 0
                        success = True
                        break
                    # Otherwise, revert? 
                    # Actually, try_swap already modified current.
                    # If we don't accept, we should revert.
                    # But we don't have the old state easily.
                    # Let's just keep the swap if it doesn't hurt.
                    # But we need to be careful not to cycle.
                    # For simplicity, let's only keep the swap if it leads to an add or if we are very stuck.
                    # To revert, we need to swap back.
                    if not success:
                        # Revert swap
                        try_swap(x_in, x_out)
                        
            if success:
                continue
                
        # 3. If stuck, restart or perturb
        if stuck_count > 50:
            # Random restart
            random_set = []
            random.shuffle(all_elements)
            for x in all_elements:
                trial = random_set + [x]
                if _is_valid_subset(trial):
                    random_set = trial
            if len(random_set) > len(current):
                current = random_set
                if len(current) > best_size:
                    best_size = len(current)
                    best_set = current[:]
            stuck_count = 0
            
    return best_set
