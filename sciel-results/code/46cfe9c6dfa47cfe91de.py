"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    deadline = time.time() + 80.0
    N = 256
    
    def has_3ap(S, x):
        s = set(S)
        for a in S:
            if x > a:
                # x = 2b - a => b = (a+x)/2
                if (a + x) % 2 == 0:
                    b = (a + x) // 2
                    if b in s and b != a:
                        return True
            else:
                # a = 2b - x => b = (x+a)/2
                if (x + a) % 2 == 0:
                    b = (x + a) // 2
                    if b in s and b != x:
                        return True
        return False
    
    def can_add(S, x):
        return not has_3ap(S, x)
    
    def recip_sum(S):
        total = Fraction(0, 1)
        for x in S:
            total += Fraction(1, x)
        return total
    
    # Greedy: add in order of decreasing reciprocal (i.e., increasing x)
    # But we want to maximize sum of 1/x, so we should prefer small numbers.
    # Greedy: try adding smallest available first that doesn't create 3-AP
    S = set()
    for x in range(1, N + 1):
        if can_add(S, x):
            S.add(x)
    
    best_S = set(S)
    best_score = recip_sum(best_S)
    
    # Local search: try swaps
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = sorted(best_S)
        for i, a in enumerate(S_list):
            if time.time() > deadline - 2:
                break
            for x in range(1, N + 1):
                if x in best_S:
                    continue
                # Try removing a and adding x
                new_S = best_S - {a}
                if can_add(new_S, x):
                    new_S.add(x)
                    new_score = recip_sum(new_S)
                    if new_score > best_score:
                        best_S = new_S
                        best_score = new_score
                        improved = True
                        break
            if improved:
                break
    
    # Local search: try adding elements after removals
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if time.time() > deadline - 2:
                break
            if x not in best_S:
                if can_add(best_S, x):
                    best_S.add(x)
                    best_score = recip_sum(best_S)
                    improved = True
    
    # Try single removals to see if we can add more
    for a in list(best_S):
        if time.time() > deadline - 2:
            break
        trial = best_S - {a}
        for x in range(1, N + 1):
            if x in trial:
                continue
            if can_add(trial, x):
                new_S = trial | {x}
                new_score = recip_sum(new_S)
                if new_score > best_score:
                    best_S = new_S
                    best_score = new_score
                    break
    
    return sorted(best_S)
# EVOLVE-BLOCK-END
