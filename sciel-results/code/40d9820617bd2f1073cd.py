"""Rewritten program for large integer sets with no two-other-product divisibility."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    
    def is_valid_set(s):
        """Check if set s satisfies: for every three distinct a,b,c in s, a does not divide b*c."""
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(n):
                if j == i:
                    continue
                for k in range(j + 1, n):
                    if k == i:
                        continue
                    b = s_list[j]
                    c = s_list[k]
                    if b * c % a == 0:
                        return False
        return True
    
    def can_add(candidate, current_set):
        """Check if adding candidate to current_set maintains validity."""
        current_list = list(current_set)
        for i in range(len(current_list)):
            a = candidate
            b = current_list[i]
            for j in range(len(current_list)):
                if j == i:
                    continue
                c = current_list[j]
                if b * c % a == 0:
                    return False
            # Also check: for each existing pair (b,c), does candidate divide b*c?
            # And for each existing element x, does x divide candidate * y for some y?
        # More carefully: after adding candidate, for every three distinct a,b,c in new set:
        # - If a is candidate: check candidate ∤ b*c for all pairs b,c in current_set
        # - If a is in current_set: check a ∤ (b*c) where {b,c} includes candidate
        #   i.e., for each a in current_set, and each other element x in current_set:
        #   check a ∤ (candidate * x)
        
        # Check 1: candidate ∤ b*c for all distinct b,c in current_set
        for b, c in combinations(current_list, 2):
            if (b * c) % candidate == 0:
                return False
        
        # Check 2: for each a in current_set, for each x in current_set where x != a:
        #   a ∤ (candidate * x)
        for a in current_list:
            for x in current_list:
                if x == a:
                    continue
                if (candidate * x) % a == 0:
                    return False
        
        return True
    
    # Greedy approach: try different orderings
    best_set = set()
    
    # Start with primes (known valid baseline)
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True
    
    primes = [x for x in all_candidates if is_prime(x)]
    best_set = set(primes)
    best_size = len(best_set)
    
    # Try greedy with different orderings
    # Ordering 1: descending (largest first)
    candidates_desc = sorted(all_candidates, reverse=True)
    current = set()
    for c in candidates_desc:
        if can_add(c, current):
            current.add(c)
    if len(current) > best_size:
        best_set = current
        best_size = len(current)
    
    # Ordering 2: ascending
    candidates_asc = sorted(all_candidates)
    current = set()
    for c in candidates_asc:
        if can_add(c, current):
            current.add(c)
    if len(current) > best_size:
        best_set = current
        best_size = len(current)
    
    # Try local search / hill climbing from best set
    # Try adding elements not in best_set
    not_in_best = [x for x in all_candidates if x not in best_set]
    
    # Try to add one element at a time (greedy improvement)
    improved = True
    while improved:
        improved = False
        for c in not_in_best:
            if time.time() > deadline:
                break
            if can_add(c, best_set):
                best_set.add(c)
                not_in_best.remove(c)
                improved = True
                best_size = len(best_set)
        if time.time() > deadline:
            break
    
    # Try removing one element and adding two others
    if time.time() < deadline - 5:
        for i in range(len(list(best_set))):
            if time.time() > deadline:
                break
            elem_to_remove = list(best_set)[i]
            test_set = set(best_set)
            test_set.discard(elem_to_remove)
            # Try to add two elements
            candidates_to_add = [x for x in all_candidates if x not in test_set]
            added = []
            for c in candidates_to_add:
                if time.time() > deadline:
                    break
                if len(added) >= 2:
                    break
                if can_add(c, test_set | set(added)):
                    added.append(c)
            if len(added) >= 2:
                new_set = test_set | set(added)
                if len(new_set) > len(best_set):
                    best_set = new_set
                    best_size = len(best_set)
    
    # Final verification
    result = sorted(best_set)
    if not is_valid_set(result):
        # Fallback to primes
        result = sorted(primes)
    
    return result

# EVOLVE-BLOCK-END
