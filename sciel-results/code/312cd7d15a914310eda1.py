import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    def is_valid_addition(x, S):
        """
        Check if adding x to S keeps it valid.
        S is a list of elements.
        For every a<b<=c in the new set, a must NOT divide b+c.
        New violations must involve x.
        1. x is a: exists b,c in S with x<b<=c and x|(b+c)
        2. x is b: exists a,c in S with a<x<=c and a|(x+c)
        3. x is c: exists a,b in S with a<b<=x and a|(b+x)
        """
        if not S:
            return True

        # Optimization: if x is larger than all elements in S, it can only be 'c'.
        # If x is smaller than all elements, it can only be 'a'.
        # But checking all cases is fast enough for n=256.
        
        # Case 1: x is 'a'
        # We need b, c in S such that x < b <= c and (b+c) % x == 0
        for b in S:
            if b <= x:
                continue
            for c in S:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    return False
        
        # Case 2: x is 'b'
        # We need a, c in S such that a < x <= c and (x+c) % a == 0
        for a in S:
            if a >= x:
                continue
            for c in S:
                if c < x:
                    continue
                if (x + c) % a == 0:
                    return False
        
        # Case 3: x is 'c'
        # We need a, b in S such that a < b <= x and (b+x) % a == 0
        for a in S:
            if a >= x:
                continue
            for b in S:
                if b <= a or b > x:
                    continue
                if (b + x) % a == 0:
                    return False
                    
        return True

    def is_valid_set(S):
        """Check if S is valid."""
        S_sorted = sorted(S)
        m = len(S_sorted)
        for i in range(m):
            a = S_sorted[i]
            for j in range(i+1, m):
                b = S_sorted[j]
                for k in range(j, m):
                    c = S_sorted[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def greedy_construct(order):
        """Construct a set by adding elements in the given order if valid."""
        S = []
        for x in order:
            if is_valid_addition(x, S):
                S.append(x)
        return S

    def local_search_improve(S):
        """Try to add any missing elements to S."""
        S_set = set(S)
        improved = True
        while improved:
            improved = False
            for x in range(1, n + 1):
                if x not in S_set:
                    if is_valid_addition(x, S):
                        S.append(x)
                        S_set.add(x)
                        improved = True
        return S

    # Strategy:
    # 1. Greedy descending (large numbers first) - usually yields large sets.
    # 2. Greedy ascending (small numbers first) - might yield different structures.
    # 3. Random restarts with greedy.
    # 4. Local search improvement on the best found.
    
    best_set = []
    best_size = 0

    # Initial strategies
    # Descending order
    S1 = greedy_construct(range(n, 0, -1))
    S1 = local_search_improve(S1)
    if len(S1) > best_size:
        best_size = len(S1)
        best_set = S1[:]

    # Ascending order
    S2 = greedy_construct(range(1, n + 1))
    S2 = local_search_improve(S2)
    if len(S2) > best_size:
        best_size = len(S2)
        best_set = S2[:]

    # Random restarts
    random.seed(42)
    for attempt in range(500):
        if time.time() > deadline - 5:
            break
        
        # Shuffle the order
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S_try = greedy_construct(order)
        S_try = local_search_improve(S_try)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]

    # Final local search on best_set
    if time.time() < deadline - 5:
        best_set = local_search_improve(best_set)

    return sorted(best_set)
