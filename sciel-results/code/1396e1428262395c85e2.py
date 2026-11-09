import time
import random
from itertools import combinations

def _check_addition(x, S):
    """
    Check if x can be added to S.
    S is a list of integers.
    Condition: No element in S U {x} divides the sum of any 3 distinct OTHER elements.
    Since S is valid, we only need to check triples involving x.
    1. x divides sum of 3 distinct elements from S.
    2. Any a in S divides sum of x and 2 distinct elements from S \ {a}.
    """
    if len(S) < 3:
        return True
    
    # 1. Check if x divides sum of any 3 distinct elements from S
    # Sum of 3 elements from S
    for a, b, c in combinations(S, 3):
        if (a + b + c) % x == 0:
            return False
            
    # 2. Check if any a in S divides (x + b + c) where b, c are distinct from a and x
    # Note: b, c must be distinct from a. x is distinct from all in S.
    for a in S:
        # We need to pick 2 distinct elements b, c from S \ {a}
        # Sum is x + b + c
        # Check if a divides x + b + c
        # Optimization: Precompute sums? No, K is small.
        # Iterate b, c from S excluding a
        # To avoid double counting and ensure distinctness, use indices or set difference
        # S is a list.
        # Create a list of S without a? Or just check indices.
        # Simpler:
        # for b in S:
        #   if b == a: continue
        #   for c in S:
        #     if c == a or c == b: continue
        #     if (x + b + c) % a == 0: return False
        
        # This is O(K^3) total for all a.
        # Let's just do it.
        for b in S:
            if b == a:
                continue
            for c in S:
                if c == a or c == b:
                    continue
                if (x + b + c) % a == 0:
                    return False
    return True

def _greedy(order):
    S = []
    for x in order:
        if _check_addition(x, S):
            S.append(x)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50s limit
    
    best_set = []
    
    # 1. Deterministic Greedy (Descending)
    order = list(range(192, 0, -1))
    S = _greedy(order)
    if len(S) > len(best_set):
        best_set = S
    if len(best_set) == 48:
        return best_set
        
    # 2. Randomized Greedy
    # Strategy: Mix of random shuffles and biased shuffles
    for i in range(1000): # Max iterations, limited by time
        if time.time() > deadline:
            break
            
        # Randomize order
        # Sometimes prefer larger numbers
        # 50%: Full random shuffle
        # 50%: Random shuffle of top half, then bottom half
        if random.random() < 0.5:
            order = list(range(1, 193))
            random.shuffle(order)
        else:
            top = list(range(96, 193))
            bot = list(range(1, 96))
            random.shuffle(top)
            random.shuffle(bot)
            order = top + bot
            
        S = _greedy(order)
        if len(S) > len(best_set):
            best_set = S
            if len(best_set) == 48:
                return best_set
                
    return best_set
