import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = 192
    max_size = 48
    
    # Helper to check if adding x to S is valid
    def is_valid_addition(x, S):
        # S is a list/set of current elements
        # 1. Check if x divides sum of any 3 distinct elements in S
        #    i.e., no a,b,c in S distinct such that (a+b+c) % x == 0
        s_list = list(S)
        L = len(s_list)
        
        # Optimization: Precompute residues? 
        # For check 1, we just iterate triples.
        # If L is small, this is fine.
        for i in range(L):
            a = s_list[i]
            for j in range(i+1, L):
                b = s_list[j]
                for k in range(j+1, L):
                    c = s_list[k]
                    if (a + b + c) % x == 0:
                        return False
        
        # 2. Check if any y in S divides (x + a + b) for distinct a,b in S\{y}
        #    Note: a,b must be distinct from x (x is new) and distinct from each other.
        #    Also a,b must be in S.
        #    The sum is x + a + b. The divisor is y in S.
        #    We need a, b distinct from y? 
        #    "distinct OTHER selected integers".
        #    If y is the divisor, the summands are x, a, b.
        #    They must be distinct from y.
        #    So a != y, b != y, a != b.
        
        for y in s_list:
            # Find pairs a, b in S such that a != y, b != y, a != b
            # and (x + a + b) % y == 0
            # We can iterate pairs in S
            for i in range(L):
                a = s_list[i]
                if a == y:
                    continue
                for j in range(i+1, L):
                    b = s_list[j]
                    if b == y:
                        continue
                    if (x + a + b) % y == 0:
                        return False
        return True

    # Greedy Initialization
    def greedy_initialization(order):
        S = set()
        for x in order:
            if len(S) >= max_size:
                break
            if is_valid_addition(x, S):
                S.add(x)
        return S

    # Try different orders
    orders = []
    orders.append(list(range(n, 0, -1))) # Largest first
    orders.append(list(range(1, n+1)))   # Smallest first
    
    # Random orders
    random.seed(42)
    for _ in range(10):
        o = list(range(1, n+1))
        random.shuffle(o)
        orders.append(o)
        
    best_set = set()
    best_size = 0
    
    for o in orders:
        if time.time() > deadline - 10:
            break
        S = greedy_initialization(o)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()
            
    # Local Search to improve best_set
    # Try to add more elements if possible
    # Or swap to enable adding more
    
    current_set = best_set.copy()
    current_size = len(current_set)
    
    # Pre-compute all candidates
    all_candidates = set(range(1, n+1))
    
    # Simple Hill Climbing: Try to add random elements not in set
    # If stuck, try swaps
    
    no_improve_count = 0
    max_no_improve = 1000
    
    while time.time() < deadline - 2:
        if len(current_set) >= max_size:
            break
            
        # Try to add a random element
        added = False
        for _ in range(50): # Try 50 random adds per step
            if time.time() > deadline - 2:
                break
            candidates = list(all_candidates - current_set)
            if not candidates:
                break
            x = random.choice(candidates)
            if is_valid_addition(x, current_set):
                current_set.add(x)
                added = True
                break
                
        if added:
            no_improve_count = 0
            continue
            
        # If couldn't add, try a swap
        # Remove one, add one (size stays same, but structure changes)
        # Or remove one, add two? (Size increases by 1)
        # Let's try remove 1, add 2.
        
        # Pick a random element to remove
        # We want to remove an element that is "causing" conflicts.
        # Hard to know. Just pick random.
        
        swap_improved = False
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            if not current_set:
                break
            y = random.choice(list(current_set))
            
            # Try to add two new elements x1, x2
            candidates = list(all_candidates - current_set)
            if len(candidates) < 2:
                continue
            
            # Shuffle candidates to pick random pairs
            random.shuffle(candidates)
            # Try first few pairs
            for i in range(min(10, len(candidates))):
                x1 = candidates[i]
                # Need x2 distinct from x1
                # Find x2
                x2_found = False
                for j in range(i+1, min(i+11, len(candidates))):
                    x2 = candidates[j]
                    new_set = current_set.copy()
                    new_set.remove(y)
                    new_set.add(x1)
                    new_set.add(x2)
                    
                    # Check validity of new_set
                    # This is O(|S|^3) roughly
                    # We can check is_valid_addition(x1, new_set - {x1}) ? No, new_set contains x1.
                    # We need to check if new_set is valid.
                    # We can check:
                    # 1. Is x1 valid with (new_set - {x1})?
                    # 2. Is x2 valid with (new_set - {x2})?
                    # 3. Are there conflicts between x1, x2 and others?
                    # Actually, is_valid_addition(x, S) checks if x can be added to S.
                    # So check is_valid_addition(x1, new_set - {x1}) and is_valid_addition(x2, new_set - {x2}).
                    # This covers all cases involving x1 or x2 as the new element.
                    # Does it cover cases where x1 and x2 interact?
                    # If x1 is divisor, sum involves x2? Yes, if x2 is in S.
                    # is_valid_addition(x1, S') where S' contains x2.
                    # Check 1: x1 divides sum of 3 in S'. One sum could be x2 + a + b.
                    # Check 2: y in S' divides x1 + a + b.
                    # So yes, checking both additions to the rest of the set covers it.
                    
                    S_rem_1 = new_set - {x1}
                    S_rem_2 = new_set - {x2}
                    
                    if is_valid_addition(x1, S_rem_1) and is_valid_addition(x2, S_rem_2):
                        # Valid swap!
                        current_set = new_set
                        swap_improved = True
                        x2_found = True
                        break
                if x2_found:
                    break
                    
        if swap_improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Restart with a new random greedy?
                # Or break?
                # Let's break to save time, we have a decent set.
                break

    return list(current_set)
