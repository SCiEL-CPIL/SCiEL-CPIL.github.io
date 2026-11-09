import time
import random

N = 1000
MAX_SUM = 2 * N
TIME_LIMIT = 140

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
        # New pairs: (x, x) and (x, y) for y in S
        if sum_counts[2 * x] >= 2:
            return False
        
        for y in S:
            s = x + y
            if sum_counts[s] >= 2:
                return False
        return True

    def add(x):
        in_set[x] = True
        S.append(x)
        sum_counts[2 * x] += 1
        for y in S:
            if y == x:
                continue
            sum_counts[x + y] += 1

    def remove(x):
        in_set[x] = False
        S.remove(x)
        sum_counts[2 * x] -= 1
        for y in S:
            if y == x:
                continue
            sum_counts[x + y] -= 1

    def swap(x_out, x_in):
        # Check validity of S \ {x_out} U {x_in}
        # Temporarily remove x_out, check x_in, revert if invalid
        remove(x_out)
        
        if can_add(x_in):
            add(x_in)
            return True
        else:
            add(x_out)
            return False

    # Phase 1: Greedy Randomized Construction
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    for x in candidates:
        if elapsed() > TIME_LIMIT - 10:
            break
        if not in_set[x]:
            if can_add(x):
                add(x)
                
    # Phase 2: Local Search (Hill Climbing)
    # Try to improve size or enable further growth via swaps
    
    while elapsed() < TIME_LIMIT - 5:
        # Try to add
        added = False
        # Try a few random additions
        for _ in range(100):
            if elapsed() > TIME_LIMIT - 5:
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
        for _ in range(100):
            if elapsed() > TIME_LIMIT - 5:
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
            # If completely stuck, we can just break or continue trying
            # Continuing is better as random fluctuations might open a path
            pass

    return S
