import time
import sys
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    n = 256
    
    def is_valid_set(S):
        """Check: for all a < b <= c in S, a does not divide b+c"""
        S_list = sorted(S)
        for i, a in enumerate(S_list):
            for j in range(i + 1, len(S_list)):
                b = S_list[j]
                for k in range(j, len(S_list)):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def can_add_element(S, x):
        """Check if x can be added to S without violating constraints.
        Assumes x is smaller than all elements in S."""
        S_list = sorted(S)
        for i, b in enumerate(S_list):
            for k in range(i, len(S_list)):
                c = S_list[k]
                if (b + c) % x == 0:
                    return False
        return True

    def greedy_extend(S, candidates, deadline):
        """Try to add elements from candidates to S, largest first."""
        current_set = set(S)
        for x in sorted(candidates, reverse=True):
            if time.time() > deadline:
                break
            if can_add_element(current_set, x):
                current_set.add(x)
        return current_set

    # Build conflict graph: vertices are 1..n, edge (x,y) exists if adding both
    # would create a violation. Two elements x < y conflict if there exists
    # z in the set such that x divides y+z or y divides x+z... but actually
    # the constraint is: for a < b <= c, a must NOT divide b+c.
    # So if we have elements a, b, c with a < b <= c, and a | (b+c), that's bad.
    # A pair (a, b) with a < b conflicts if there exists some c >= b in the set
    # such that a | (b+c). But c could be b itself (b=c) or larger.
    # For a fixed pair (a, b) with a < b, they conflict if there exists c in [b, n]
    # such that (b+c) % a == 0. This depends on what other elements we include.
    # 
    # Actually, for the independent set formulation: two vertices x, y are adjacent
    # (conflict) if including BOTH x and y in the set would force a violation with
    # some third element. But the violation involves three elements. So the "conflict"
    # isn't just pairwise - it's a 3-way constraint.
    #
    # Alternative: model as a 3-uniform hypergraph independence problem. But that's
    # harder. Instead, let's use a different approach:
    # 
    # For a given set S, the constraint is: for all a < b <= c in S, a ∤ (b+c).
    # We can think of this as: for each a in S, and for each pair (b,c) with a < b <= c in S,
    # we need a ∤ (b+c).
    #
    # Let's use a greedy + local search approach with a better heuristic:
    # Start with the upper interval, then try to add smaller elements one by one,
    # and use local search to swap elements.

    def local_search_polish(S, deadline):
        """Try to improve S by removing elements that block others, then re-adding."""
        current_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element and see if we can add more
            for a in sorted(current_set):
                if time.time() > deadline:
                    break
                test_set = set(current_set)
                test_set.remove(a)
                # Try to add back a and also try other elements
                candidates = [x for x in range(1, n+1) if x not in test_set]
                extended = greedy_extend(test_set, candidates, deadline)
                if len(extended) > len(current_set):
                    current_set = extended
                    improved = True
                    break
        return current_set

    def swap_local_search(S, deadline):
        """Try to improve S by swapping one element for another."""
        current_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in sorted(current_set):
                if time.time() > deadline:
                    break
                test_set = set(current_set)
                test_set.remove(a)
                # Try adding each candidate not in test_set
                for x in range(1, n+1):
                    if time.time() > deadline:
                        break
                    if x in test_set:
                        continue
                    if x == a:
                        continue
                    test_set.add(x)
                    if is_valid_set(test_set):
                        # Check if this is an improvement (same size, but maybe enables more)
                        # Actually we need to check if we can extend further
                        extended = greedy_extend(test_set, [y for y in range(1, n+1) if y not in test_set], deadline)
                        if len(extended) > len(current_set):
                            current_set = extended
                            improved = True
                            break
                if improved:
                    break
        return current_set

    # 1. Find the best interval [L, 256]
    best_set = set()
    best_size = 0
    
    for L in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Quick check: does L divide any b+c where b,c in [L+1, n]?
        k_min = (2 * (L + 1) + L - 1) // L
        k_max = (2 * n) // L
        if k_min <= k_max:
            continue
        
        # Full check for promising L
        if L >= 140:
            S_test = set(range(L, n + 1))
            if is_valid_set(S_test):
                size = len(S_test)
                if size > best_size:
                    best_size = size
                    best_set = S_test

    if not best_set:
        return list(range(192, 257))

    # 2. Multi-start greedy with random restarts
    best_result = set(best_set)
    best_result_size = len(best_result)
    
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        if restart == 0:
            # Deterministic: start from best interval
            current_set = set(best_set)
        else:
            # Random: pick a random upper-tail interval
            L = random.randint(150, 256)
            S_test = set(range(L, n + 1))
            if not is_valid_set(S_test):
                continue
            current_set = S_test
        
        # Greedy extend
        candidates = [x for x in range(1, min(list(current_set))) if x not in current_set]
        current_set = greedy_extend(current_set, candidates, deadline)
        
        # Local search polish
        if time.time() < deadline - 2:
            current_set = local_search_polish(current_set, deadline)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    # 3. Branch and bound on conflict structure
    # For the final best set, try to add more elements using a smarter approach
    if time.time() < deadline - 3:
        # Try adding elements in order of "least conflict" 
        # An element x has "conflict count" = number of pairs (b,c) in current_set
        # with x < b <= c and x | (b+c)
        def conflict_count(S, x):
            count = 0
            S_list = sorted(S)
            for i, b in enumerate(S_list):
                if b <= x:
                    continue
                for k in range(i, len(S_list)):
                    c = S_list[k]
                    if (b + c) % x == 0:
                        count += 1
            return count

        current_set = set(best_result)
        # Try to add elements with lowest conflict count first
        candidates = [x for x in range(1, n+1) if x not in current_set]
        candidates.sort(key=lambda x: conflict_count(current_set, x))
        
        for x in candidates:
            if time.time() > deadline - 1:
                break
            if can_add_element(current_set, x):
                current_set.add(x)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    return list(best_result)

# EVOLVE-BLOCK-END
