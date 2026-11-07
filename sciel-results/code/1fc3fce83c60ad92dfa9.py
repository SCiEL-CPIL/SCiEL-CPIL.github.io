import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80
    
    S = [0] * 129
    for i in range(65, 129):
        S[i] = 1
    
    # Precompute active pairs: pairs (a,b) with a<b, a,b in S, a+b in S
    def compute_active_pairs():
        active = set()
        for a in range(1, 129):
            if S[a] == 0:
                continue
            for b in range(a+1, 129):
                if S[b] == 0:
                    continue
                s = a + b
                if s <= 128 and S[s] == 1:
                    active.add((a, b))
        return active
    
    def can_add(x, active_pairs):
        # Check if adding x creates a forbidden triple
        # A forbidden triple is {a,b,c} such that a+b, a+c, b+c are in S.
        # If we add x, the new triples are {a,b,x} for a,b in S.
        # For {a,b,x} to be forbidden, we need:
        # 1. a+b in S
        # 2. a+x in S
        # 3. b+x in S
        # We only need to check pairs (a,b) where a+b is in S.
        # These are exactly the pairs in active_pairs.
        for (a, b) in active_pairs:
            if a + x <= 128 and S[a+x] == 1 and b + x <= 128 and S[b+x] == 1:
                return False
        return True

    def add_to_set(x, active_pairs):
        S[x] = 1
        # 1. Add pairs involving x as an element
        for a in range(1, 129):
            if a == x or S[a] == 0:
                continue
            s = a + x
            if s <= 128 and S[s] == 1:
                if a < x:
                    active_pairs.add((a, x))
                else:
                    active_pairs.add((x, a))
        # 2. Add pairs where x is the sum
        # We need to find a,b in S such that a+b=x
        # Since a,b < x, we can iterate a from 1 to x-1
        for a in range(1, x):
            if S[a] == 0:
                continue
            b = x - a
            if b > a and b < 129 and S[b] == 1:
                active_pairs.add((a, b))

    def remove_from_set(x, active_pairs):
        # 1. Remove pairs involving x as an element
        # We need to remove (a,x) where a in S and a+x in S
        for a in range(1, 129):
            if a == x or S[a] == 0:
                continue
            s = a + x
            if s <= 128 and S[s] == 1:
                if a < x:
                    if (a, x) in active_pairs:
                        active_pairs.remove((a, x))
                else:
                    if (x, a) in active_pairs:
                        active_pairs.remove((x, a))
        
        # 2. Remove pairs where x is the sum
        # We need to remove (a,b) where a+b=x
        for a in range(1, x):
            if S[a] == 0:
                continue
            b = x - a
            if b > a and b < 129 and S[b] == 1:
                if (a, b) in active_pairs:
                    active_pairs.remove((a, b))
        
        S[x] = 0

    active_pairs = compute_active_pairs()
    
    # Greedy add
    smalls = list(range(1, 65))
    random.shuffle(smalls)
    for x in smalls:
        if time.time() > deadline - 30:
            break
        if can_add(x, active_pairs):
            add_to_set(x, active_pairs)
    
    best_size = sum(S)
    best_S = S[:]
    best_active = active_pairs.copy()
    
    # Local search
    for it in range(200000):
        if time.time() > deadline:
            break
        
        r = random.random()
        if r < 0.4:
            x = random.randint(1, 64)
            if S[x] == 0 and can_add(x, active_pairs):
                add_to_set(x, active_pairs)
                if sum(S) > best_size:
                    best_size = sum(S)
                    best_S = S[:]
                    best_active = active_pairs.copy()
        elif r < 0.7:
            smalls_in_S = [i for i in range(1, 65) if S[i] == 1]
            if smalls_in_S:
                x = random.choice(smalls_in_S)
                remove_from_set(x, active_pairs)
        else:
            smalls_in_S = [i for i in range(1, 65) if S[i] == 1]
            if smalls_in_S:
                x = random.choice(smalls_in_S)
                remove_from_set(x, active_pairs)
                y = random.randint(1, 64)
                if S[y] == 0 and can_add(y, active_pairs):
                    add_to_set(y, active_pairs)
                    if sum(S) > best_size:
                        best_size = sum(S)
                        best_S = S[:]
                        best_active = active_pairs.copy()
                else:
                    # Revert
                    add_to_set(x, active_pairs)
    
    result = [i for i in range(1, 129) if best_S[i] == 1]
    return result
