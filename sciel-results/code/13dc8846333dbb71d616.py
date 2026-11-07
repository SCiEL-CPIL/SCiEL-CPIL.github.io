"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

N = 128

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin
    
    # Fast validity: check if set S has no three distinct a,b,c with a+b,a+c,b+c all in S
    def is_valid_full(S):
        S_set = set(S)
        elems = sorted(S_set)
        n = len(elems)
        for i in range(n):
            a = elems[i]
            for j in range(i+1, n):
                b = elems[j]
                ab = a + b
                if ab not in S_set:
                    continue
                for k in range(j+1, n):
                    c = elems[j+1] if k == j+1 else elems[k]
                    if c == a or c == b:
                        continue
                    if (a+c) in S_set and (b+c) in S_set:
                        return False
        return True
    
    def can_add(S_set, x):
        """Check if adding x to S_set creates any violating triple."""
        # Only triples involving x can be new violations
        # Triple (x, a, b) where a,b in S_set, a != b, a != x, b != x
        # Violates if (x+a) in S_set and (x+b) in S_set and (a+b) in S_set
        elems = [e for e in S_set if e != x]
        n = len(elems)
        for i in range(n):
            a = elems[i]
            xa = x + a
            if xa not in S_set:
                continue
            for j in range(i+1, n):
                b = elems[j]
                xb = x + b
                if xb not in S_set:
                    continue
                ab = a + b
                if ab in S_set:
                    return False
        return True
    
    # Start with upper half
    best_set = set(range(65, 129))
    best_size = len(best_set)
    
    # Try to add elements from lower half
    for x in range(1, 65):
        if time.time() > deadline - 10:
            break
        if can_add(best_set, x):
            best_set.add(x)
            best_size = len(best_set)
    
    # Local search: try to add more elements
    changed = True
    while changed and time.time() < deadline - 5:
        changed = False
        for x in range(1, 129):
            if x in best_set:
                continue
            if can_add(best_set, x):
                best_set.add(x)
                best_size = len(best_set)
                changed = True
    
    # Try removing elements to allow adding more
    for _ in range(10):
        if time.time() > deadline - 5:
            break
        improved = False
        for x in sorted(best_set):
            if time.time() > deadline - 5:
                break
            trial_set = best_set - {x}
            # Try to add elements
            added = 0
            for y in range(1, 129):
                if y in trial_set:
                    continue
                if can_add(trial_set, y):
                    trial_set.add(y)
                    added += 1
            if len(trial_set) > len(best_set):
                best_set = trial_set
                best_size = len(best_set)
                improved = True
        if not improved:
            break
    
    return list(best_set)
# EVOLVE-BLOCK-END
