import time
import random

N = 1000
MAX_SUM = 2 * N
TIME_LIMIT = 100

def construct():
    start_time = time.time()
    
    # Use a list of booleans for membership, and a list for the set
    in_set = [False] * (N + 1)
    S = []
    # sum_counts[s] = number of pairs (a,b) in S with a<=b and a+b=s
    sum_counts = [0] * (MAX_SUM + 1)
    
    def elapsed():
        return time.time() - start_time

    def can_add(x):
        # Check if adding x keeps all counts <= 2
        if sum_counts[2 * x] >= 2:
            return False
        
        # Local variable lookup is faster
        sc = sum_counts
        for y in S:
            if sc[x + y] >= 2:
                return False
        return True

    def add(x):
        in_set[x] = True
        S.append(x)
        sum_counts[2 * x] += 1
        sc = sum_counts
        for y in S:
            if y != x:
                sc[x + y] += 1

    def remove(x):
        in_set[x] = False
        S.remove(x) # O(N)
        sum_counts[2 * x] -= 1
        sc = sum_counts
        for y in S:
            if y != x:
                sc[x + y] -= 1

    def swap(x_out, x_in):
        # Check validity of S \ {x_out} U {x_in}
        # We can simulate by temporarily removing x_out, checking x_in, then reverting if invalid.
        # Or just calculate the delta.
        
        # Let's do a fast check:
        # 1. Remove x_out contributions from a temporary view? 
        # Actually, easiest is:
        #   Temporarily remove x_out
        #   Check if x_in can be added
        #   If yes, add x_in (commit swap)
        #   If no, restore x_out
        
        # But `remove` and `add` modify state. 
        # To avoid deep copies, we can just do it in place and revert.
        
        # Save state? No, just do it.
        remove(x_out)
        
        if can_add(x_in):
            add(x_in)
            return True
        else:
            add(x_out)
            return False

    # Phase 1: Greedy Randomized Construction
    # Try to add elements in random order
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    # We will iterate through candidates multiple times if we want, 
    # but a single pass with random order is a good start.
    # Then we do local search.
    
    for x in candidates:
        if elapsed() > TIME_LIMIT - 10:
            break
        if not in_set[x]:
            if can_add(x):
                add(x)
                
    # Phase 2: Local Search (Hill Climbing)
    # Optimized for speed: fewer attempts per iteration, stricter time checks.
    
    while elapsed() < TIME_LIMIT - 2:
        # Try to add
        added = False
        # Try a smaller number of random additions to keep iteration time low
        for _ in range(10):
            if elapsed() > TIME_LIMIT - 2:
                break
            x = random.randint(1, N)
            if not in_set[x]:
                if can_add(x):
                    add(x)
                    added = True
                    break
        
        if added:
            continue
            
        # If no addition, try swaps
        # Pick random x_out in S, x_in not in S
        swapped = False
        for _ in range(10):
            if elapsed() > TIME_LIMIT - 2:
                break
            if not S:
                break
            x_out = random.choice(S)
            x_in = random.randint(1, N)
            if in_set[x_in]:
                continue
            if swap(x_out, x_in):
                swapped = True
                break
        
        if not swapped and not added:
            # Small sleep to avoid busy waiting if stuck, though time check handles it
            # No sleep, just loop again
            pass
            
    return S
