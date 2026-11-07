import random
import time
from itertools import combinations

P = {'n': 256}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin
    
    n = P['n']
    
    # Precompute conflict relationships
    # conflict[x] = set of pairs (a,b) from [2..n] such that x divides a*b, or a divides x*b, or b divides x*a
    # Actually, we need: for a set S, x can be added iff for all distinct a,b in S:
    #   x does NOT divide a*b, AND a does NOT divide x*b, AND b does NOT divide x*a
    
    # Precompute for each x, which pairs (a,b) with a<b in [2..n] are "conflicting" with x
    # i.e., x | a*b OR a | x*b OR b | x*a
    
    # This is expensive to precompute fully, so let's use a faster can_add
    
    # Helper to check if x can be added to S
    def can_add(x, S):
        if len(S) < 2:
            return True
        
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                # Check all three conditions
                if (a * b) % x == 0:
                    return False
                if (x * b) % a == 0:
                    return False
                if (x * a) % b == 0:
                    return False
        return True
    
    # Dynamic greedy: at each step, pick the candidate that has the fewest conflicts with current S
    # "Conflict" means: if we added x, how many pairs in S would violate the condition?
    # But since we need ZERO violations, we want candidates that are "safe"
    # Among safe candidates, prefer those that are "less restrictive" for future additions
    
    # Score a candidate x given current set S:
    # For each pair (a,b) in S, check if x would create a conflict.
    # If no conflict, x is safe. Among safe x, we want to pick one that leaves the most options.
    # Approximate: pick x that minimizes the number of "near-conflicts" with S
    # A near-conflict: there exists a in S such that for many b, x*b % a == 0 or similar
    
    def count_near_conflicts(x, S):
        """Count how many elements in S are 'close' to conflicting with x.
        For each a in S, check if x*a is divisible by many potential b's, or a*x divisible by many b's.
        Simpler: for each a in S, check if gcd(x, a) > 1 (since if gcd>1, there might be issues)
        Even simpler: just check direct conflicts with pairs."""
        if len(S) < 2:
            return 0
        count = 0
        S_list = list(S)
        for i in range(len(S_list)):
            a = S_list[i]
            for j in range(i+1, len(S_list)):
                b = S_list[j]
                # These are the same checks as can_add, but we count instead of early exit
                if (a * b) % x == 0:
                    return float('inf')  # Hard conflict
                if (x * b) % a == 0:
                    count += 1
                if (x * a) % b == 0:
                    count += 1
        return count
    
    # Local search with 2-for-2 swaps
    def local_search(S, remaining_candidates):
        S_set = set(S)
        improved = True
        max_iterations = 50
        iteration = 0
        while improved and iteration < max_iterations:
            iteration += 1
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
            
            # Try 1-for-2 swaps: remove one, add two
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
            
            # Try 2-for-2 swaps: remove two, add two
            if not improved:
                for i in range(len(S)):
                    if time.time() > deadline:
                        return S
                    for j in range(i+1, len(S)):
                        if time.time() > deadline:
                            return S
                        S_no_ij = [S[k] for k in range(len(S)) if k != i and k != j]
                        S_no_ij_set = set(S_no_ij)
                        for x in remaining_candidates:
                            if x in S_set or x == S[i] or x == S[j]:
                                continue
                            if can_add(x, S_no_ij):
                                S_new1 = S_no_ij + [x]
                                S_new1_set = S_no_ij_set | {x}
                                for y in remaining_candidates:
                                    if y in S_new1_set or y == S[i] or y == S[j] or y == x:
                                        continue
                                    if can_add(y, S_new1):
                                        S = S_new1 + [y]
                                        S_set = S_new1_set | {y}
                                        improved = True
                                        break
                                if improved:
                                    break
                        if improved:
                            break
                    if improved:
                        break
        return S
    
    best_set = []
    
    # Run multiple greedy passes with different strategies
    for restart in range(40):
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
        elif restart == 4:
            # Dynamic greedy: at each step, pick the best candidate
            S = []
            available = set(range(2, n + 1))
            while available:
                if time.time() > deadline:
                    break
                best_x = None
                best_score = float('inf')
                for x in available:
                    score = count_near_conflicts(x, S)
                    if score < best_score:
                        best_score = score
                        best_x = x
                    if time.time() > deadline:
                        break
                if best_x is None or best_score == float('inf'):
                    break
                S.append(best_x)
                available.discard(best_x)
                if time.time() > deadline:
                    break
        else:
            random.shuffle(candidates)
            S = []
            for x in candidates:
                if time.time() > deadline:
                    break
                if can_add(x, S):
                    S.append(x)
        
        if restart == 4:
            # S was already built dynamically
            pass
        else:
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
