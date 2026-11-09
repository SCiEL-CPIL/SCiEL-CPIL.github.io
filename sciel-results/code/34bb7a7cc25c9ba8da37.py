import time
import random

N = 256

def is_valid_sidon_maximal(S):
    # Verification for safety
    sums = set()
    for i, a in enumerate(S):
        for j in range(i, len(S)):
            b = S[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    
    # Check maximality
    S_set = set(S)
    for x in range(1, N + 1):
        if x in S_set:
            continue
        # Try adding x
        new_sums = set(sums)
        valid = True
        for a in S:
            s1 = x + a
            if s1 in new_sums:
                valid = False
                break
            new_sums.add(s1)
        if valid:
            s2 = 2 * x
            if s2 in new_sums:
                valid = False
        if valid:
            return False # x could be added, so not maximal
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 25.0
    
    best_set = None
    best_size = N + 1
    
    # Initial deterministic trial (ascending)
    perm = list(range(1, N + 1))
    S, sums = greedy_construct(perm)
    if len(S) < best_size:
        best_size = len(S)
        best_set = S[:]
        
    while time.time() < deadline:
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        S, sums = greedy_construct(perm)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = S[:]
            
    return best_set

def greedy_construct(perm):
    S = []
    sums = set()
    for x in perm:
        # Check if x can be added
        # 1. 2x not in sums
        if 2 * x in sums:
            continue
        # 2. x + a not in sums for all a in S
        can_add = True
        for a in S:
            if x + a in sums:
                can_add = False
                break
        
        if can_add:
            S.append(x)
            sums.add(2 * x)
            for a in S[:-1]: # Exclude x itself from the loop for sums update? 
                # Wait, the check was for existing S.
                # Now we add new sums.
                # The new sums are x+a for a in OLD S, and 2x.
                # We already checked they don't collide.
                # We just need to add them.
                sums.add(x + a)
                
    return S, sums
