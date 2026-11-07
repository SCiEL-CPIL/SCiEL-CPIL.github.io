import time
import random
from math import gcd, isqrt

P = {'n': 256}

def is_valid(S):
    S_list = list(S)
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(n):
            if j == i: continue
            b = S_list[j]
            g = gcd(b, a)
            k = a // g
            for k_idx in range(n):
                if k_idx == i or k_idx == j: continue
                c = S_list[k_idx]
                if c % k == 0:
                    return False
    return True

def construct():
    n = P['n']
    primes = [a for a in range(2, n+1) if all(a%d for d in range(2, isqrt(a)+1))]
    S = set(primes)
    candidates = set(range(2, n+1)) - S
    
    start_time = time.time()
    deadline = start_time + 50
    
    # Phase 1: Greedy Add (1-for-0)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for c in list(candidates):
            S_new = S | {c}
            if is_valid(S_new):
                S = S_new
                candidates.remove(c)
                changed = True
                # Restart to check for new additions enabled by this one? 
                # Actually, adding an element makes constraints stricter, 
                # so it can only remove possibilities, not add them. 
                # So we can just continue.
                break
        if changed:
            continue # Restart inner loop? No, just continue to next candidate?
            # If we add c, other candidates might still be addable.
            # So we should not break the outer while, but continue the for loop?
            # The for loop iterates over a snapshot of candidates.
            # If we remove c, the loop continues with the next candidate.
            # This is fine.
            
    # Phase 2: Local Search (1-for-1)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(S):
            if time.time() > deadline:
                break
            for c in list(candidates):
                S_new = (S - {a}) | {c}
                if is_valid(S_new):
                    S = S_new
                    candidates = candidates - {c} | {a}
                    improved = True
                    break
            if improved:
                break
                
    # Phase 3: Random Search to escape local optima
    if time.time() < deadline - 5:
        best_S = S
        best_size = len(S)
        current_S = S
        current_candidates = candidates
        
        for _ in range(10000):
            if time.time() > deadline:
                break
            a = random.choice(list(current_S))
            c = random.choice(list(current_candidates))
            S_new = (current_S - {a}) | {c}
            if is_valid(S_new):
                current_S = S_new
                current_candidates = current_candidates - {c} | {a}
                if len(current_S) > best_size:
                    best_S = current_S
                    best_size = len(current_S)
        S = best_S
        # Update candidates to match best_S for return? 
        # We just return list(S)
        
    return list(S)
