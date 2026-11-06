import time
import random

def is_3ap_free(S):
    S = set(S)
    for a in S:
        for b in S:
            if a == b: continue
            c = 2*b - a
            if c in S and c != a and c != b:
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50.0 # 10s margin
    
    # Initial greedy set
    S = []
    for x in range(1, 257):
        if all(2*b-a not in S and 2*b-a != x for a in S for b in S if a < b):
             # Check if x forms AP with any pair in S
             # Pairs (a,b) in S. x is 3rd term: x = 2b - a => 2b = x+a
             # x is 1st term: a = 2x - b => 2x = a+b
             # x is 2nd term: 2x = a+b
             pass 
    # Better check:
    def can_add(x, S):
        for a in S:
            for b in S:
                if a == b: continue
                # x, a, b form AP?
                # x is middle: 2x = a+b
                if 2*x == a+b: return False
                # x is first: x + b = 2a => b = 2a - x
                if 2*a - x in S and 2*a - x != x and 2*a - x != a: return False
                # x is last: a + x = 2b => a = 2b - x
                if 2*b - x in S and 2*b - x != x and 2*b - x != b: return False
        return True

    # Re-implement greedy properly
    S_greedy = []
    for x in range(1, 257):
        if can_add(x, S_greedy):
            S_greedy.append(x)
    
    best_S = set(S_greedy)
    best_score = sum(1.0/x for x in best_S)
    
    # Local Search
    # We want to swap x in S with y not in S such that 1/y > 1/x (y < x)
    # and the new set is valid.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        S_list = list(best_S)
        for i, x in enumerate(S_list):
            if time.time() > deadline: break
            # Try to replace x with y < x
            for y in range(1, x):
                if y in best_S: continue
                # Check if best_S - {x} + {y} is valid
                # Remove x, check if y can be added to rest
                S_rest = best_S - {x}
                if can_add(y, S_rest):
                    new_S = S_rest | {y}
                    new_score = best_score - 1.0/x + 1.0/y
                    if new_score > best_score:
                        best_S = new_S
                        best_score = new_score
                        improved = True
                        # Restart inner loop? Or just continue?
                        # Continuing might miss better swaps from the new state immediately, 
                        # but restarting is safer for convergence.
                        # However, restarting the outer loop is expensive.
                        # Let's break and restart the while loop.
                        break
            if improved:
                break
        if not improved:
             # Try adding? No, adding larger elements hurts.
             # Try removing? No, removing hurts.
             # Maybe try 2-for-1?
             pass

    return list(best_S)
