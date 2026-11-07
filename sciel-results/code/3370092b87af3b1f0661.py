import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s limit
    
    N = 256
    # Candidate set: Integers 2..256
    
    # Precompute: for each a, list of (b,c) pairs in 2..256 where a | b*c
    # This is expensive, so we'll compute on the fly with caching
    
    # Function to check if adding x to set S creates any violation
    def can_add(x, S_set, S_list):
        # For each a in S, check if a | x * b for some b in S, b != x, b != a
        # For each b in S, check if b | x * a for some a in S, a != x, a != b
        # For x itself: check if x | a * b for some a,b in S, a != b, a != x, b != x
        n = len(S_list)
        for a in S_list:
            if a == x:
                continue
            # Check if a | x * b for some b in S, b != a, b != x
            for b in S_list:
                if b == a or b == x:
                    continue
                if (x * b) % a == 0:
                    return False
            # Check if x | a * b for some b in S, b != a, b != x
            for b in S_list:
                if b == a or b == x:
                    continue
                if (a * b) % x == 0:
                    return False
        return True

    # Greedy Deletion with better conflict counting
    def greedy_delete(initial_set, max_iter=200):
        S = list(initial_set)
        S_set = set(S)
        
        for _ in range(max_iter):
            if time.time() > deadline:
                return S
            
            # Count conflicts for each element
            # A conflict is a triple (a,b,c) where one divides the product of the other two
            conflicts = {a: 0 for a in S}
            
            S_list = S
            n = len(S_list)
            
            # For efficiency, we'll check triples
            for i in range(n):
                a = S_list[i]
                for j in range(i + 1, n):
                    b = S_list[j]
                    for k in range(j + 1, n):
                        c = S_list[k]
                        # Check all three possibilities
                        if (b * c) % a == 0:
                            conflicts[a] += 1
                            conflicts[b] += 1
                            conflicts[c] += 1
                        if (a * c) % b == 0:
                            conflicts[a] += 1
                            conflicts[c] += 1
                            conflicts[b] += 1
                        if (a * b) % c == 0:
                            conflicts[a] += 1
                            conflicts[b] += 1
                            conflicts[c] += 1
            
            # If no conflicts, done
            max_conf = max(conflicts.values())
            if max_conf == 0:
                return S
            
            # Remove element with max conflicts, with random tie-breaking
            candidates = [a for a in S if conflicts[a] == max_conf]
            elem_to_remove = random.choice(candidates)
            S.remove(elem_to_remove)
            S_set.remove(elem_to_remove)
        
        return S

    # Try to add elements back after greedy deletion
    def try_expand(S, S_set):
        S_list = list(S)
        # Try to add each element from 2..256 not in S
        for x in range(2, N + 1):
            if x in S_set:
                continue
            if can_add(x, S_set, S_list):
                S.append(x)
                S_set.add(x)
                S_list.append(x)
        return S, S_set

    # Local search: try swaps
    def local_search(S, S_set, max_swaps=50):
        S_list = list(S)
        n = len(S_list)
        
        for _ in range(max_swaps):
            if time.time() > deadline:
                break
            
            improved = False
            # Try to swap: remove y from S, add x not in S
            # This is expensive, so we'll do it selectively
            
            # For each x not in S, try to find a y in S such that removing y and adding x is valid
            # This is O(N^2) * validation, which is too slow for large N
            # Instead, we'll try a simpler approach: just try to add elements
            
            # Try adding elements that weren't added before
            for x in range(2, N + 1):
                if x in S_set:
                    continue
                # Check if adding x is valid
                if can_add(x, S_set, S_list):
                    S.append(x)
                    S_set.add(x)
                    S_list.append(x)
                    improved = True
                    break
            
            if not improved:
                break
        
        return S, S_set

    # Strategy:
    # 1. Try multiple initial sets
    # 2. Greedy delete each
    # 3. Try to expand
    # 4. Keep the best
    
    best_set = []
    best_set_set = set()
    
    # Initial Set 1: Upper Half
    initial1 = list(range(129, N + 1))
    S1 = greedy_delete(initial1)
    S1_set = set(S1)
    S1, S1_set = try_expand(S1, S1_set)
    if len(S1) > len(best_set):
        best_set = S1
        best_set_set = S1_set
    
    # Initial Set 2: All Primes
    def is_prime(n):
        if n < 2: return False
        if n == 2: return True
        if n % 2 == 0: return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    primes = [x for x in range(2, N + 1) if is_prime(x)]
    S2 = greedy_delete(primes)
    S2_set = set(S2)
    S2, S2_set = try_expand(S2, S2_set)
    if len(S2) > len(best_set):
        best_set = S2
        best_set_set = S2_set
    
    # Initial Set 3: Numbers with high prime factors (heuristic)
    # Numbers that are products of large primes are less likely to divide products
    initial3 = []
    for x in range(2, N + 1):
        # Count number of divisors
        div_count = 0
        for d in range(2, int(x**0.5) + 1):
            if x % d == 0:
                div_count += 2
                if d * d == x:
                    div_count -= 1
        if x == 2:
            div_count = 1
        # Prefer numbers with fewer divisors
        if div_count <= 3:
            initial3.append(x)
    
    if initial3:
        S3 = greedy_delete(initial3)
        S3_set = set(S3)
        S3, S3_set = try_expand(S3, S3_set)
        if len(S3) > len(best_set):
            best_set = S3
            best_set_set = S3_set
    
    # Try random restarts with different initial sets
    for restart in range(5):
        if time.time() > deadline:
            break
        
        # Random initial set: mix of upper half and some lower numbers
        initial = list(range(129, N + 1))
        # Add some random numbers from lower half
        lower = list(range(2, 129))
        random.shuffle(lower)
        initial.extend(lower[:30])
        
        S = greedy_delete(initial)
        S_set = set(S)
        S, S_set = try_expand(S, S_set)
        
        if len(S) > len(best_set):
            best_set = S
            best_set_set = S_set
    
    # Final expansion pass
    best_set, best_set_set = try_expand(best_set, best_set_set)
    
    # One more check: ensure types are correct
    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
