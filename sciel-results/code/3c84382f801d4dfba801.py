import sys
from itertools import combinations
import random

def construct():
    candidates = list(range(1, 193))
    
    def is_valid(x, S):
        # S is a list of selected integers
        n = len(S)
        if n < 3:
            # If n < 3, we can't form a sum of 3 others for x.
            # But we must check if x creates conflict with existing y's?
            # Condition: No y in S divides sum of x and 2 others.
            # If n < 2, we can't form sum of x and 2 others.
            # So if n < 2, it's valid.
            # If n == 2, we check y1, y2.
            pass
        
        # Check 1: Does x divide sum of any 3 distinct in S?
        if n >= 3:
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Check 2: Does any y in S divide sum of x and any 2 distinct others in S \ {y}?
        for y in S:
            others = [v for v in S if v != y]
            if len(others) < 2:
                continue
            for a, b in combinations(others, 2):
                if (x + a + b) % y == 0:
                    return False
        
        return True

    def greedy(order):
        S = []
        for x in order:
            if is_valid(x, S):
                S.append(x)
        return S

    # Initialize with best greedy result
    best_set = []
    
    orders = [
        candidates,
        candidates[::-1],
        sorted(candidates, key=lambda x: -x),
        sorted(candidates, key=lambda x: x),
        # Try a few random permutations with fixed seed
    ]
    
    for i in range(5):
        random.seed(i)
        random.shuffle(candidates)
        orders.append(candidates[:])
    
    for order in orders:
        S = greedy(order)
        if len(S) > len(best_set):
            best_set = S
    
    best_size = len(best_set)
    
    # Branch and Bound
    # Sort candidates to process most constrained first?
    # Small numbers are more constrained.
    search_order = list(range(1, 193))
    
    # To speed up B&B, we can use a recursive function with pruning.
    # We need to avoid deep recursion if possible, but 192 is fine.
    
    # We'll use a global best
    global_best_size = best_size
    global_best_set = best_set[:]
    
    # Optimization: Pre-compute nothing complex, rely on is_valid.
    # But is_valid is O(N^3). 
    # If N grows to 40, 40^3 = 64000.
    # If we visit 10000 nodes, 640M ops. Too slow in Python.
    
    # We need a faster validity check or better pruning.
    
    # Let's try a different approach for B&B:
    # Use bitsets for S?
    # And precompute conflicts?
    
    # Actually, let's try to improve the greedy significantly.
    # The "No element divides sum of 3 others" constraint is quite loose for large numbers.
    # Most conflicts come from small numbers.
    # If we just pick large numbers, they are safe.
    
    # Let's try a greedy that picks the largest valid number first.
    # This is candidates[::-1].
    
    # Let's try to run a local search on the best greedy result.
    def local_search(S):
        S = S[:]
        improved = True
        while improved:
            improved = False
            # Try to add any number from 1..192 not in S
            for x in range(1, 193):
                if x not in S:
                    if is_valid(x, S):
                        S.append(x)
                        improved = True
                        break
            # If we couldn't add, try swapping?
            # Or removing one to allow adding two?
            # Simple hill climbing: if stuck, try removing one element and see if we can add more.
            if not improved:
                # Try removing one element
                for i in range(len(S)):
                    orig = S[:]
                    S.pop(i)
                    # Try to add
                    added = False
                    for x in range(1, 193):
                        if x not in S:
                            if is_valid(x, S):
                                S.append(x)
                                added = True
                                break
                    if added and len(S) > len(orig):
                        improved = True
                        break
                    elif added and len(S) == len(orig):
                        # Keep the new one? Maybe it allows more later.
                        # For now, revert if no size gain
                        S = orig
                    else:
                        S = orig
        return S

    best_set = local_search(best_set)
    best_size = len(best_set)
    
    # If time permits, run B&B?
    # Given 60s limit, and Python speed, B&B might time out.
    # Let's stick to Greedy + Local Search which is robust.
    
    return best_set
