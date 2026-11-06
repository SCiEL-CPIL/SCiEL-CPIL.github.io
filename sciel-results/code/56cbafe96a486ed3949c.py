import time
import sys
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 87.0  # 3s safety margin
    
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

    def swap_local_search(S, deadline):
        """Try to improve S by swapping: remove one element, try to add two new ones."""
        current_set = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for a in sorted(current_set):
                if time.time() > deadline:
                    break
                test_set = set(current_set)
                test_set.remove(a)
                # Try to add back a and also try other elements
                candidates = [x for x in range(1, n+1) if x not in test_set and x != a]
                extended = greedy_extend(test_set, candidates, deadline)
                if len(extended) > len(current_set):
                    current_set = extended
                    improved = True
                    break
        return current_set

    def final_polish(S, deadline):
        """Final greedy pass: try to add any remaining elements."""
        current_set = set(S)
        all_elements = set(range(1, n+1))
        candidates = list(all_elements - current_set)
        current_set = greedy_extend(current_set, candidates, deadline)
        return current_set

    def build_conflict_graph():
        """Build a conflict graph where edge (i,j) exists if having both i and j
        in the set creates a violation with some third element.
        
        Actually, the constraint is: for all a < b <= c in S, a ∤ (b+c).
        This is a 3-way constraint, not pairwise. So we need a different approach.
        
        Instead, let's define: for a fixed set S, the constraint involves triples.
        We'll use a different strategy: precompute which pairs (b,c) are "dangerous"
        for each a, then use branch-and-bound on the sorted elements.
        """
        # For each a, compute the set of pairs (b,c) with a < b <= c such that a | (b+c)
        # This helps us quickly check conflicts
        dangerous_pairs = [set() for _ in range(n + 1)]
        for a in range(1, n + 1):
            for b in range(a + 1, n + 1):
                for c in range(b, n + 1):
                    if (b + c) % a == 0:
                        dangerous_pairs[a].add((b, c))
        return dangerous_pairs

    def branch_and_bound(deadline):
        """Branch-and-bound to find maximum valid subset.
        
        We process elements in decreasing order (larger elements first).
        At each step, we decide to include or exclude the current element.
        """
        dangerous_pairs = build_conflict_graph()
        
        # We'll use a recursive approach with memoization-like pruning
        # Process elements from n down to 1
        elements = list(range(n, 0, -1))
        
        best_set = set()
        best_size = 0
        
        # Precompute for quick conflict checking
        # For a given set S (as a frozenset or sorted tuple), check if adding x is valid
        # But this is expensive. Instead, we'll maintain the set incrementally.
        
        # Use a different approach: start with all large elements and try to add smaller ones
        # Or: use local search with the conflict structure
        
        # Actually, let's use a simpler but effective approach:
        # 1. Start with a good initial solution (upper tail)
        # 2. Use local search with moves: add one, remove one, swap two
        
        # First, find the best upper-tail interval
        for L in range(1, n + 1):
            if time.time() > deadline:
                break
            S_test = set(range(L, n + 1))
            if is_valid_set(S_test):
                if len(S_test) > best_size:
                    best_size = len(S_test)
                    best_set = S_test
        
        if not best_set:
            return set(range(192, 257))
        
        # Now use local search to improve
        current_set = set(best_set)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding elements
            for x in range(1, min(current_set) if current_set else n + 1):
                if time.time() > deadline - 2:
                    break
                if x not in current_set:
                    test_set = current_set | {x}
                    if is_valid_set(test_set):
                        current_set = test_set
                        improved = True
                        break
            # If no single add worked, try swaps
            if not improved:
                for x in list(current_set):
                    if time.time() > deadline - 2:
                        break
                    test_set = current_set - {x}
                    for y in range(1, min(test_set) if test_set else n + 1):
                        if time.time() > deadline - 2:
                            break
                        if y not in test_set and y != x:
                            test_set2 = test_set | {y}
                            if is_valid_set(test_set2) and len(test_set2) > len(current_set):
                                current_set = test_set2
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
    
    num_restarts = 15
    for restart in range(num_restarts):
        if time.time() > deadline - 3:
            break
        
        if restart == 0:
            # Deterministic: start from best interval
            current_set = set(best_set)
        else:
            # Random: pick a random upper-tail interval
            L = random.randint(140, 256)
            S_test = set(range(L, n + 1))
            if not is_valid_set(S_test):
                continue
            current_set = S_test
        
        # Greedy extend
        candidates = [x for x in range(1, min(list(current_set))) if x not in current_set]
        current_set = greedy_extend(current_set, candidates, deadline)
        
        # Swap-based local search
        if time.time() < deadline - 2:
            current_set = swap_local_search(current_set, deadline)
        
        # Final polish pass
        if time.time() < deadline - 1:
            current_set = final_polish(current_set, deadline)
        
        if len(current_set) > best_result_size:
            best_result = current_set
            best_result_size = len(current_set)

    # 3. Try branch-and-bound approach
    if time.time() < deadline - 5:
        bb_result = branch_and_bound(deadline - 3)
        if len(bb_result) > best_result_size:
            best_result = bb_result
            best_result_size = len(bb_result)

    # Final polish on best result
    if time.time() < deadline - 0.5:
        best_result = final_polish(best_result, deadline)

    return list(best_result)

# EVOLVE-BLOCK-END
