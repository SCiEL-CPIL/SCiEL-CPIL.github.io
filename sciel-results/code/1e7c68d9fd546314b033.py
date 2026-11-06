import time
import random

def _has_4ap_in_set(S):
    """
    Check if set S contains a nonconstant 4-term arithmetic progression.
    Returns True if a 4-AP exists, False otherwise.
    """
    if len(S) < 4:
        return False
    
    # For efficiency, we can iterate over pairs or use the property of APs.
    # A set S has a 4-AP if there exist a, b, c, d in S such that a+d = b, b+d=c, c+d=e.
    # i.e., a, a+d, a+2d, a+3d in S.
    
    # Simple O(N^2) check might be too slow if called frequently in local search.
    # We need a fast "can_add" check for the local search steps.
    # This function is mainly for final verification or initial checks.
    
    s_list = sorted(S)
    s_set = S
    n = len(s_list)
    
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            d = s_list[j] - a
            if d == 0:
                continue
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if b in s_set and c in s_set and e in s_set:
                return True
    return False

def _can_add(S, x):
    """
    Check if adding x to set S creates a 4-term AP.
    Returns True if x can be added (i.e., no 4-AP is formed), False otherwise.
    """
    if x in S:
        return True # Already there, no change
        
    # x is the 1st term: x, x+d, x+2d, x+3d
    # x is the 2nd term: x-d, x, x+d, x+2d
    # x is the 3rd term: x-2d, x-d, x, x+d
    # x is the 4th term: x-3d, x-2d, x-d, x

    # To check efficiently, we iterate over potential d values.
    # d must be such that the other terms are in S (or within range).
    
    # Case 1: x is 1st term (a)
    # Need y1, y2, y3 in S such that y1 = x+d, y2 = x+2d, y3 = x+3d
    # This implies d = y1 - x.
    # Iterate y1 in S where y1 > x:
    #   d = y1 - x
    #   if x+2d in S and x+3d in S: return False
    
    # Case 2: x is 2nd term (b)
    # Need y0, y2, y3 in S such that y0 = x-d, y2 = x+d, y3 = x+2d
    # d = x - y0 = y2 - x.
    # Iterate y0 in S where y0 < x:
    #   d = x - y0
    #   if x+d in S and x+2d in S: return False
    
    # Case 3: x is 3rd term (c)
    # Need y0, y1, y3 in S such that y0 = x-2d, y1 = x-d, y3 = x+d
    # d = x - y1 = y3 - x.
    # Iterate y1 in S where y1 < x:
    #   d = x - y1
    #   if x-2d in S and x+d in S: return False
    
    # Case 4: x is 4th term (d)
    # Need y0, y1, y2 in S such that y0 = x-3d, y1 = x-2d, y2 = x-d
    # d = x - y2.
    # Iterate y2 in S where y2 < x:
    #   d = x - y2
    #   if x-3d in S and x-2d in S: return False

    # Optimization: Instead of iterating all S for each case, we can just check specific d's.
    # But S is a set, so lookup is O(1).
    # The number of iterations is proportional to |S|.
    # Total complexity O(|S|) per check.
    
    # Case 1: x is a
    for y1 in S:
        if y1 <= x: continue
        d = y1 - x
        if (x + 2*d) in S and (x + 3*d) in S:
            return False
            
    # Case 2: x is b
    for y0 in S:
        if y0 >= x: continue
        d = x - y0
        if (x + d) in S and (x + 2*d) in S:
            return False
            
    # Case 3: x is c
    for y1 in S:
        if y1 >= x: continue
        d = x - y1
        if (x - 2*d) in S and (x + d) in S:
            return False
            
    # Case 4: x is d
    for y2 in S:
        if y2 >= x: continue
        d = x - y2
        if (x - 3*d) in S and (x - 2*d) in S:
            return False
            
    return True

def _ternary_01_set(n):
    """Generate set of numbers in [1, n] with base-3 digits only 0 and 1."""
    s = set()
    for x in range(1, n + 1):
        temp = x
        valid = True
        while temp > 0:
            if temp % 3 == 2:
                valid = False
                break
            temp //= 3
        if valid:
            s.add(x)
    return s

def construct():
    deadline = time.time() + 85  # Safety margin
    
    # 1. Initialize with Base-3 {0,1} set
    current_set = _ternary_01_set(512)
    
    # 2. Greedy Expansion
    # Try to add all other numbers in increasing order
    for x in range(1, 513):
        if x not in current_set:
            if _can_add(current_set, x):
                current_set.add(x)
                
    best_set = set(current_set)
    best_size = len(best_set)
    
    # 3. Local Search (Hill Climbing with Swaps)
    # We try to improve the set by swapping elements or adding if possible.
    # Since we did greedy add, it's hard to add more.
    # We use swaps to rearrange the set to potentially allow future adds or just to find a local max.
    # Actually, if we are at a local maximum for "add", we need to "remove" to "add".
    # A swap (remove x, add y) keeps size same.
    # To increase size, we need a sequence of swaps that leads to a state where an add is possible.
    
    # Strategy:
    # Loop until deadline.
    # 1. Try to add a random element. If success, update best.
    # 2. If fail, try a random swap. If valid, accept with some probability (simulated annealing-like) or just accept if it leads to a "better" structure?
    # Simpler: Just try random swaps. Occasionally, check if we can add.
    
    all_numbers = list(range(1, 513))
    random.shuffle(all_numbers)
    
    while time.time() < deadline:
        # Try to add a random element not in set
        # Pick a random candidate from all_numbers
        # To make it efficient, we can maintain a list of candidates, but re-shuffling is fine.
        
        # Try Add
        added = False
        for y in random.sample(all_numbers, min(100, len(all_numbers))):
            if y not in current_set:
                if _can_add(current_set, y):
                    current_set.add(y)
                    added = True
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(best_set)
                    break
        
        if added:
            continue
            
        # Try Swap
        # Pick x in current_set, y not in current_set
        # If _can_add(current_set - {x}, y) is True, then swap is valid.
        # Note: _can_add checks if adding y to S creates AP.
        # Here S is current_set - {x}.
        
        # To avoid O(N^2) in the swap check (which involves creating a new set and calling _can_add),
        # we can optimize:
        # Removing x might break APs, allowing y to be added.
        
        # Let's just do a simple swap attempt.
        # Sample a few swaps
        for _ in range(50):
            if not current_set: break
            x = random.choice(list(current_set))
            y = random.choice(all_numbers)
            if y in current_set or y == x:
                continue
                
            # Check if (current_set - {x}) + {y} is valid
            # This is equivalent to:
            # 1. y can be added to (current_set - {x})
            # 2. No AP is formed that involves y and elements of current_set - {x}
            
            # Efficient check:
            # Temporarily remove x
            current_set.remove(x)
            
            if _can_add(current_set, y):
                # Valid swap
                current_set.add(y)
                # We accepted the swap. 
                # We don't know if it's "better" in terms of future adds, 
                # but it keeps size same. 
                # We accept it to explore the state space.
                # If we find a better size later, we update best_set.
            else:
                # Invalid swap, revert
                current_set.add(x)
        
        # Periodically, if we are stuck, we might try a "2-swap" or just continue.
        # The random swaps act as a random walk in the space of valid sets of fixed size.
        # If we hit a state where an add is possible, we take it.

    return sorted(best_set)

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 85  # Safety margin
    
    # 1. Initialize with Base-3 {0,1} set
    current_set = _ternary_01_set(512)
    
    # 2. Greedy Expansion
    # Try to add all other numbers in increasing order
    for x in range(1, 513):
        if x not in current_set:
            if _can_add(current_set, x):
                current_set.add(x)
                
    best_set = set(current_set)
    best_size = len(best_set)
    
    # 3. Local Search (Hill Climbing with Swaps)
    all_numbers = list(range(1, 513))
    
    # Pre-compute list for random choices to avoid list(set) overhead in tight loop
    # We'll just use random.sample or random.choice on lists.
    
    while time.time() < deadline:
        # Try to add a random element not in set
        added = False
        # Sample a batch of candidates
        candidates = random.sample(all_numbers, 100)
        for y in candidates:
            if y not in current_set:
                if _can_add(current_set, y):
                    current_set.add(y)
                    added = True
                    if len(current_set) > best_size:
                        best_set = set(current_set)
                        best_size = len(best_set)
                    break
        
        if added:
            continue
            
        # Try Swap
        # Pick x in current_set, y not in current_set
        # Sample a few swaps
        for _ in range(50):
            if not current_set: break
            x = random.choice(list(current_set))
            y = random.choice(all_numbers)
            if y in current_set or y == x:
                continue
                
            # Check if (current_set - {x}) + {y} is valid
            current_set.remove(x)
            
            if _can_add(current_set, y):
                current_set.add(y)
                # Accepted swap
            else:
                current_set.add(x)
                # Reverted swap
                
    return sorted(best_set)
# EVOLVE-BLOCK-END
