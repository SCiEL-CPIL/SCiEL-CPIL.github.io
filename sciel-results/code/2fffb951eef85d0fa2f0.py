import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    n = 256

    # Precompute divisibility relationships for speed
    # For a fixed a, which numbers x satisfy x % a == 0?
    # We need to check if (b+c) % a == 0.
    # Let's precompute for each a in 1..n, the list of numbers in 1..2n that are divisible by a.
    # Actually, b+c ranges from 1+2 to 255+256 = 511.
    # So we need divisibility up to 512.
    
    # Precompute divisors or multiples?
    # It's faster to check (b+c) % a == 0 directly if we don't have too many checks.
    # But with local search, we might check many times.
    # Let's precompute a lookup table:
    # div_lookup[a][s] = True if a divides s.
    # s ranges from 2 to 512.
    # a ranges from 1 to 256.
    
    max_sum = 2 * n
    div_lookup = [[False] * (max_sum + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for multiple in range(a, max_sum + 1, a):
            div_lookup[a][multiple] = True

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
        
        # Case 1: x is 'a'
        # We need b, c in S such that x < b <= c and x | (b+c)
        # b+c ranges from (x+1)+(x+1) to max_S+max_S
        # We can iterate b in S where b > x.
        # Then for each b, we need c in S such that c >= b and (b+c) % x == 0.
        # This is equivalent to: exists c in S, c >= b, c ≡ -b (mod x).
        
        # For speed, let's just loop. S is small.
        S_list = S
        len_S = len(S_list)
        
        # Case 1: x is 'a'
        for i in range(len_S):
            b = S_list[i]
            if b <= x:
                continue
            # b > x
            # Check c in S such that c >= b
            for j in range(len_S):
                c = S_list[j]
                if c < b:
                    continue
                # Check divisibility
                if div_lookup[x][b + c]:
                    return False
        
        # Case 2: x is 'b'
        # We need a, c in S such that a < x <= c and a | (x+c)
        for i in range(len_S):
            a = S_list[i]
            if a >= x:
                continue
            # a < x
            # Check c in S such that c >= x
            for j in range(len_S):
                c = S_list[j]
                if c < x:
                    continue
                if div_lookup[a][x + c]:
                    return False
        
        # Case 3: x is 'c'
        # We need a, b in S such that a < b <= x and a | (b+x)
        for i in range(len_S):
            a = S_list[i]
            if a >= x:
                continue
            # a < x
            for j in range(len_S):
                b = S_list[j]
                if b <= a or b > x:
                    continue
                if div_lookup[a][b + x]:
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
                    if div_lookup[a][b + c]:
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

    def remove_to_fix(S):
        """
        Start with S, remove elements to make it valid.
        Greedy: find a violating triple, remove the element with the most
        conflicts (or the smallest, to keep larger elements).
        """
        S_set = set(S)
        S_list = sorted(S_set)
        changed = True
        while changed:
            changed = False
            # Find a violation
            m = len(S_list)
            violation = None
            for i in range(m):
                a = S_list[i]
                for j in range(i+1, m):
                    b = S_list[j]
                    for k in range(j, m):
                        c = S_list[k]
                        if div_lookup[a][b + c]:
                            violation = (a, b, c)
                            break
                    if violation:
                        break
                if violation:
                    break
            if not violation:
                break
            a, b, c = violation
            # Remove the smallest element (a) as it's the most restrictive
            S_set.discard(a)
            S_list = sorted(S_set)
            changed = True
        return list(S_set)

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

    # Remove-to-fix: start with all elements, remove to fix
    S3 = remove_to_fix(list(range(1, n + 1)))
    S3 = local_search_improve(S3)
    if len(S3) > best_size:
        best_size = len(S3)
        best_set = S3[:]

    # Random restarts with greedy (random order)
    random.seed(42)
    for attempt in range(800):
        if time.time() > deadline - 5:
            break
        
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        S_try = greedy_construct(order)
        S_try = local_search_improve(S_try)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]

    # Random restarts with remove-to-fix (random removal order)
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        S_try = list(range(1, n + 1))
        random.shuffle(S_try)
        S_set = set(S_try)
        S_list = sorted(S_set)
        changed = True
        while changed:
            changed = False
            m = len(S_list)
            violation = None
            for i in range(m):
                a = S_list[i]
                for j in range(i+1, m):
                    b = S_list[j]
                    for k in range(j, m):
                        c = S_list[k]
                        if div_lookup[a][b + c]:
                            violation = (a, b, c)
                            break
                    if violation:
                        break
                if violation:
                    break
            if not violation:
                break
            # Remove a random element from the violating triple
            a, b, c = violation
            candidates = [a, b, c]
            random.shuffle(candidates)
            for cand in candidates:
                if cand in S_set:
                    S_set.discard(cand)
                    S_list = sorted(S_set)
                    changed = True
                    break
        S_try = list(S_set)
        S_try = local_search_improve(S_try)
        
        if len(S_try) > best_size:
            best_size = len(S_try)
            best_set = S_try[:]

    # Final local search on best_set
    if time.time() < deadline - 5:
        best_set = local_search_improve(best_set)

    return sorted(best_set)
