"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin for 90s limit
    
    n = 192
    k = 3
    
    # Precompute all valid triples of distinct integers from 1..n
    # A triple (a, b, c) is valid if a, b, c are distinct and in [1, n]
    # We store the sum and the elements
    # Actually, for incremental check, we need to know:
    # If we add x, which existing pairs (a, b) in S form a triple (a, b, x) such that some y in S divides sum(a,b,x)?
    # Or if we add x, does x divide sum(a,b,c) for any triple (a,b,c) in S?
    
    # Let's use a different incremental strategy:
    # Maintain S.
    # To check if x can be added to S:
    # 1. Does x divide sum(a,b,c) for any distinct a,b,c in S?
    # 2. For any pair (a,b) in S, does any y in S (including x? No, x is new) divide sum(a,b,x)?
    #    Wait, the condition is: No selected integer y divides sum of any 3 distinct OTHER selected integers.
    #    So if we add x, we must check:
    #    A) For every y in S (old elements), does y divide sum(a,b,x) for any pair (a,b) in S \ {y}?
    #       Note: a,b,x must be distinct from y. Since x is new, a,b must be in S \ {y, x}. Since x not in S, a,b in S \ {y}.
    #    B) Does x divide sum(a,b,c) for any triple (a,b,c) in S?
    
    # This is still O(|S|^3) for A and O(|S|^3) for B if done naively.
    # However, |S| is at most 48. 48^3 = 110k operations. This is very fast.
    # We can do full re-validation in ~1ms.
    # So we can do thousands of restarts.
    
    def is_valid_set(S):
        # S is a list of selected integers
        # Check all y in S
        for y in S:
            others = [v for v in S if v != y]
            for a, b, c in itertools.combinations(others, 3):
                s = a + b + c
                if s % y == 0:
                    return False
        return True

    def greedy_randomized(order):
        S = []
        for x in order:
            if x in S:
                continue
            # Check if adding x keeps validity
            # Optimization: Only check constraints involving x
            # 1. Does x divide sum(a,b,c) for any a,b,c in S?
            valid = True
            for a, b, c in itertools.combinations(S, 3):
                if (a + b + c) % x == 0:
                    valid = False
                    break
            if not valid:
                continue
            
            # 2. For any y in S, does y divide sum(a,b,x) for any a,b in S \ {y}?
            #    Note: a,b must be distinct from y and x.
            for y in S:
                others = [v for v in S if v != y]
                for a, b in itertools.combinations(others, 2):
                    if (a + b + x) % y == 0:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                S.append(x)
        return S

    def local_search(S):
        # Try to add elements not in S
        # Or swap elements to allow larger sets?
        # Simple addition first
        S = list(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x in S:
                    continue
                # Check validity of adding x
                # Use the same logic as greedy
                valid = True
                for a, b, c in itertools.combinations(S, 3):
                    if (a + b + c) % x == 0:
                        valid = False
                        break
                if not valid:
                    continue
                
                for y in S:
                    others = [v for v in S if v != y]
                    for a, b in itertools.combinations(others, 2):
                        if (a + b + x) % y == 0:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    S.append(x)
                    improved = True
                    break # Restart loop as S changed
        return S

    best_S = []
    best_len = 0
    
    # Try a few deterministic orders first
    # 1. Descending
    S = greedy_randomized(range(n, 0, -1))
    S = local_search(S)
    if len(S) > best_len:
        best_len = len(S)
        best_S = S
    
    # 2. Ascending
    S = greedy_randomized(range(1, n + 1))
    S = local_search(S)
    if len(S) > best_len:
        best_len = len(S)
        best_S = S

    # Randomized restarts
    while time.time() < deadline:
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        # Sometimes bias towards larger numbers?
        # Let's just shuffle for now.
        S = greedy_randomized(order)
        
        # Local search to fill gaps
        S = local_search(S)
        
        if len(S) > best_len:
            best_len = len(S)
            best_S = S
            # If we hit the theoretical max or a very high number, we might stop early?
            # Max is 48.
            if best_len >= 48:
                break
        
        # Occasionally try swap-based local search?
        # Given time, simple addition is usually sufficient if greedy is good.
        
    return best_S

# EVOLVE-BLOCK-END
