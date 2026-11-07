import random
import time
from itertools import combinations

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    n = P['n']
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        if len(S) < 2:
            return True
        
        for i in range(len(S)):
            a = S[i]
            for j in range(i+1, len(S)):
                b = S[j]
                prod_ab = a * b
                if prod_ab % x == 0:
                    return False
                prod_xb = x * b
                if prod_xb % a == 0:
                    return False
                prod_xa = x * a
                if prod_xa % b == 0:
                    return False
        return True
    
    # Check if a set is valid (all triples satisfy the condition)
    def is_valid_set(S):
        S_list = list(S)
        for a, b, c in combinations(S_list, 3):
            if (a * b) % c == 0 or (a * c) % b == 0 or (b * c) % a == 0:
                return False
        return True
    
    # Local search: try swapping elements to improve the set
    def local_search(S, remaining_candidates):
        S_set = set(S)
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            # Try adding each remaining candidate
            for x in remaining_candidates:
                if x in S_set:
                    continue
                if can_add(x, S):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    if time.time() > deadline:
                        return S
            # Try removing one element and adding two others
            for i in range(len(S)):
                if time.time() > deadline:
                    return S
                S_no_i = [S[j] for j in range(len(S)) if j != i]
                S_no_i_set = set(S_no_i)
                for x in remaining_candidates:
                    if x in S_set or x == S[i]:
                        continue
                    if can_add(x, S_no_i):
                        S_new = S_no_i + [x]
                        S_new_set = S_no_i_set | {x}
                        # Check if we can add another element
                        for y in remaining_candidates:
                            if y in S_new_set or y == S[i] or y == x:
                                continue
                            if can_add(y, S_new):
                                S = S_new + [y]
                                S_set = S_new_set | {y}
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
        return S
    
    best_set = []
    
    # Run multiple greedy passes with different orderings
    for restart in range(30):
        if time.time() > deadline:
            break
            
        candidates = list(range(2, n + 1))
        
        if restart == 0:
            candidates.sort(key=lambda x: -x)
        elif restart == 1:
            candidates.sort(key=lambda x: x)
        elif restart == 2:
            # Sort by number of divisors in range (fewer first)
            def num_divisors(x):
                count = 0
                for d in range(2, x):
                    if x % d == 0:
                        count += 1
                return count
            candidates.sort(key=num_divisors)
        elif restart == 3:
            # Sort by largest prime factor (larger first)
            def largest_prime_factor(x):
                max_p = 1
                d = 2
                while d * d <= x:
                    while x % d == 0:
                        max_p = max(max_p, d)
                        x //= d
                    d += 1
                if x > max_p:
                    max_p = x
                return max_p
            candidates.sort(key=largest_prime_factor, reverse=True)
        else:
            random.shuffle(candidates)
        
        S = []
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, S):
                S.append(x)
        
        # Apply local search
        remaining = [x for x in range(2, n + 1) if x not in set(S)]
        S = local_search(S, remaining)
        
        if len(S) > len(best_set):
            best_set = S[:]
            
    # Ensure return type is list of ints
    return [int(x) for x in best_set]
