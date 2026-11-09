"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s kill
    
    N = 256
    
    # 1. Precompute prime factors for 1..N
    prime_factors = [set() for _ in range(N + 1)]
    for i in range(2, N + 1):
        if not prime_factors[i]:
            for j in range(i, N + 1, i):
                prime_factors[j].add(i)
    # 1 has no prime factors, prime_factors[1] remains empty
    
    def is_compatible(a, b):
        return not (prime_factors[a] & prime_factors[b])
    
    def get_sum(s):
        return sum(s)
    
    # 2. Initial Solution: Greedy by value
    # We start with 1. Then we try to add numbers from N down to 2.
    # A number can be added if it is compatible with all currently selected numbers.
    # However, simple greedy might miss better combinations.
    # Let's use a smarter initialization: Start with all primes? No, primes conflict with composites.
    # Actually, the set of all primes <= N is a valid solution (pairwise coprime).
    # Sum of primes <= 256 is quite large.
    # Let's start with the set of all primes <= N plus 1.
    
    primes = [i for i in range(2, N + 1) if prime_factors[i] == {i}]
    initial_set = [1] + primes
    
    # Ensure distinct and sorted for consistency
    current_set = list(set(initial_set))
    current_sum = get_sum(current_set)
    best_set = list(current_set)
    best_sum = current_sum
    
    # 3. Local Search: 1-for-1 Swaps and Insertions
    # We try to improve the sum by:
    # a) Adding a number if it fits (increases sum)
    # b) Replacing a number with a larger one if it fits and increases sum
    
    # To speed up, we can iterate through candidates multiple times.
    # Candidates: 1..N
    # Since 1 is always included, we focus on 2..N.
    
    # Optimization: Precompute compatibility matrix?
    # N=256, 256x256 matrix is small.
    # compat[i][j] = True if i and j are coprime.
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            compat[i][j] = is_compatible(i, j)
            compat[j][i] = compat[i][j]
        compat[i][i] = False # A number is not compatible with itself in terms of adding it again
    
    # Helper to check if a candidate can be added to a set
    def can_add(candidate, selected_set):
        for s in selected_set:
            if s == candidate:
                return False
            if not compat[s][candidate]:
                return False
        return True
    
    # Helper to check if a candidate can replace an element
    def can_replace(candidate, remove_elem, selected_set):
        if candidate == remove_elem:
            return False
        # Check if candidate is compatible with all other elements in selected_set except remove_elem
        for s in selected_set:
            if s == remove_elem:
                continue
            if not compat[s][candidate]:
                return False
        return True

    # Local Search Loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try Insertions: Add a number not in the set
        for num in range(N, 1, -1):
            if num in current_set:
                continue
            if can_add(num, current_set):
                current_set.append(num)
                current_sum += num
                improved = True
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_set = list(current_set)
        
        # 1-for-1 swaps: replace one element with a larger compatible one
        current_set.sort()
        swapped = False
        for i in range(len(current_set)):
            if time.time() > deadline:
                break
            elem = current_set[i]
            for cand in range(N, elem, -1):
                if cand in current_set:
                    continue
                if can_replace(cand, elem, current_set):
                    current_sum -= elem
                    current_set[i] = cand
                    current_sum += cand
                    improved = True
                    swapped = True
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_set = list(current_set)
                    current_set.sort()
                    break
            if swapped:
                break
        
        # 2-for-1 swap: remove two elements, add one larger element
        if not swapped and time.time() < deadline:
            current_set.sort()
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                for j in range(i + 1, len(current_set)):
                    if time.time() > deadline:
                        break
                    a, b = current_set[i], current_set[j]
                    # Try adding a single number c that is coprime with all others (except a,b)
                    # and c > a + b
                    for cand in range(N, a + b, -1):
                        if cand in current_set:
                            continue
                        # Check cand is coprime with all elements except a and b
                        ok = True
                        for s in current_set:
                            if s == a or s == b:
                                continue
                            if not compat[s][cand]:
                                ok = False
                                break
                        if ok:
                            current_sum -= a
                            current_sum -= b
                            current_sum += cand
                            current_set.remove(a)
                            current_set.remove(b)
                            current_set.append(cand)
                            current_set.sort()
                            improved = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_set = list(current_set)
                            break
                    if improved:
                        break
                if improved:
                    break
        
        # 1-for-2 swap: remove one element, add two smaller elements whose sum is greater
        if not improved and time.time() < deadline:
            current_set.sort()
            for i in range(len(current_set)):
                if time.time() > deadline:
                    break
                elem = current_set[i]
                # Try pairs (c1, c2) not in current_set, c1 < c2, c1+c2 > elem
                # c1, c2 must be coprime with each other and with all other elements
                for c1 in range(1, elem):
                    if time.time() > deadline:
                        break
                    if c1 in current_set:
                        continue
                    for c2 in range(c1 + 1, N + 1):
                        if c1 + c2 <= elem:
                            continue
                        if c2 in current_set:
                            continue
                        # Check c1 and c2 are coprime with each other
                        if not compat[c1][c2]:
                            continue
                        # Check both are coprime with all other elements
                        ok = True
                        for s in current_set:
                            if s == elem:
                                continue
                            if not compat[s][c1] or not compat[s][c2]:
                                ok = False
                                break
                        if ok:
                            current_sum -= elem
                            current_sum += c1
                            current_sum += c2
                            current_set.remove(elem)
                            current_set.append(c1)
                            current_set.append(c2)
                            current_set.sort()
                            improved = True
                            if current_sum > best_sum:
                                best_sum = current_sum
                                best_set = list(current_set)
                            break
                    if improved:
                        break
                if improved:
                    break
        
    # Final check: Ensure distinct integers
    # The logic above ensures distinctness by checking `in current_set`.
    # But let's be safe.
    final_set = list(set(best_set))
    
    # Ensure types are int
    final_set = [int(x) for x in final_set]
    
    return final_set

# EVOLVE-BLOCK-END
