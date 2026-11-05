import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 85.0
    
    base_set = set(range(91, 181))
    candidates = list(range(1, 91))
    
    best_set = set(base_set)
    best_size = len(best_set)
    
    def can_add(x, current_sa):
        # current_sa is a set of elements in [1, 90] already added
        # We are adding x.
        # Check if 1/a = 1/x + 1/c for a, c in current_sa
        # This implies c = ax / (a - x)
        # We need a > x for c > 0.
        for a in current_sa:
            if a <= x:
                continue
            den = a - x
            num = a * x
            if num % den == 0:
                c = num // den
                # Check if c is in current_sa and distinct from a and x
                if c in current_sa and c != a:
                    return False
        return True

    # 1. Descending greedy
    sa_desc = set()
    for x in sorted(candidates, reverse=True):
        if can_add(x, sa_desc):
            sa_desc.add(x)
    set_desc = base_set | sa_desc
    if len(set_desc) > best_size:
        best_set = set_desc
        best_size = len(set_desc)
        
    # 2. Random restarts
    num_restarts = 500
    for i in range(num_restarts):
        if time.time() - start_time > DEADLINE:
            break
            
        random.shuffle(candidates)
        sa_rand = set()
        for x in candidates:
            if can_add(x, sa_rand):
                sa_rand.add(x)
        
        set_rand = base_set | sa_rand
        if len(set_rand) > best_size:
            best_set = set_rand
            best_size = len(set_rand)
            
    return list(best_set)
