import random
import sys
from itertools import combinations, combinations_with_replacement
import heapq

def construct():
    n = 256
    max_size = 64
    random.seed(42)

    # We'll model this as finding a large independent set in a conflict graph.
    # Each vertex represents (element, set) where set is 'A' or 'B'.
    # Two vertices conflict if adding both would violate Sidon or diff-disjointness.
    
    # For efficiency, we'll use a greedy approach that builds the sets incrementally
    # while checking conflicts, then improve via local search.
    
    def is_sidon(S):
        if len(S) <= 1:
            return True
        sums = set()
        for i, a in enumerate(S):
            for b in S[i:]:
                if (a + b) in sums:
                    return False
                sums.add(a + b)
        return True

    def nonzero_diffs(S):
        if len(S) <= 1:
            return set()
        diffs = set()
        for a in S:
            for b in S:
                if a != b:
                    diffs.add(b - a)
        return diffs

    def can_add_to_S(S, x, other_diffs):
        # Check if x can be added to S maintaining Sidon property
        # and that new diffs don't conflict with other_diffs
        new_S = S + [x]
        if not is_sidon(new_S):
            return False
        new_diffs = nonzero_diffs(new_S)
        if new_diffs & other_diffs:
            return False
        return True

    # Greedy construction: build A and B simultaneously, choosing elements
    # that have the fewest conflicts with remaining candidates
    best_total = 0
    best_A = []
    best_B = []

    for trial in range(100):
        random.seed(trial)
        
        A = []
        B = []
        A_diffs = set()
        B_diffs = set()
        
        # Track which elements are still available
        available = set(range(1, n + 1))
        
        # Build sets by greedily adding elements that minimize future conflicts
        for _ in range(128):  # Up to 64 in each set
            if not available:
                break
            
            # Try to add to A first, then B
            added = False
            
            # Score each available element by how "conflict-free" it is
            best_x_A = None
            best_score_A = float('inf')
            best_x_B = None
            best_score_B = float('inf')
            
            for x in available:
                # Check if x can be added to A
                if len(A) < max_size:
                    new_A = A + [x]
                    if is_sidon(new_A):
                        new_A_diffs = nonzero_diffs(new_A)
                        if not (new_A_diffs & B_diffs):
                            # Count how many other elements would be blocked
                            blocked = 0
                            for y in available:
                                if y == x:
                                    continue
                                # Check if y would conflict with new_A
                                if len(A) + 1 < max_size:
                                    test_A = new_A + [y]
                                    if not is_sidon(test_A):
                                        blocked += 1
                                    else:
                                        test_diffs = nonzero_diffs(test_A)
                                        if test_diffs & B_diffs:
                                            blocked += 1
                                # Check if y would conflict with B given new_A_diffs
                                if len(B) < max_size:
                                    test_B = B + [y]
                                    if is_sidon(test_B):
                                        test_B_diffs = nonzero_diffs(test_B)
                                        if test_B_diffs & new_A_diffs:
                                            blocked += 1
                            score = blocked
                            if score < best_score_A:
                                best_score_A = score
                                best_x_A = x
                
                # Check if x can be added to B
                if len(B) < max_size:
                    new_B = B + [x]
                    if is_sidon(new_B):
                        new_B_diffs = nonzero_diffs(new_B)
                        if not (new_B_diffs & A_diffs):
                            blocked = 0
                            for y in available:
                                if y == x:
                                    continue
                                # Check if y would conflict with new_B
                                if len(B) + 1 < max_size:
                                    test_B = new_B + [y]
                                    if not is_sidon(test_B):
                                        blocked += 1
                                    else:
                                        test_diffs = nonzero_diffs(test_B)
                                        if test_diffs & A_diffs:
                                            blocked += 1
                                # Check if y would conflict with A given new_B_diffs
                                if len(A) < max_size:
                                    test_A = A + [y]
                                    if is_sidon(test_A):
                                        test_A_diffs = nonzero_diffs(test_A)
                                        if test_A_diffs & new_B_diffs:
                                            blocked += 1
                            score = blocked
                            if score < best_score_B:
                                best_score_B = score
                                best_x_B = x
            
            # Choose the better option
            if best_x_A is not None and (best_x_B is None or best_score_A <= best_score_B):
                A.append(best_x_A)
                available.remove(best_x_A)
                A_diffs = nonzero_diffs(A)
                added = True
            elif best_x_B is not None:
                B.append(best_x_B)
                available.remove(best_x_B)
                B_diffs = nonzero_diffs(B)
                added = True
            
            if not added:
                break
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]

    A = best_A[:]
    B = best_B[:]
    A_diffs = nonzero_diffs(A)
    B_diffs = nonzero_diffs(B)

    # Local search: try to add more elements or swap to improve
    improved = True
    max_iterations = 50
    iteration = 0
    
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Try to add a new element to A or B
        for side, S, S_diffs, other_diffs in [(0, A, A_diffs, B_diffs), (1, B, B_diffs, A_diffs)]:
            if len(S) >= max_size:
                continue
            
            existing = set(S)
            for x in range(1, n + 1):
                if x in existing:
                    continue
                if can_add_to_S(S, x, other_diffs):
                    if side == 0:
                        A.append(x)
                        A_diffs = nonzero_diffs(A)
                    else:
                        B.append(x)
                        B_diffs = nonzero_diffs(B)
                    improved = True
                    break
        
        # Try swaps: replace an element to allow more in the other set
        for side, S, S_diffs, other, other_diffs in [(0, A, A_diffs, B, B_diffs), (1, B, B_diffs, A, A_diffs)]:
            if not S:
                continue
            
            for i, val in enumerate(S[:]):
                # Try removing val and replacing with a different element
                S_temp = S[:]
                S_temp.remove(val)
                S_temp_diffs = nonzero_diffs(S_temp)
                
                # Find a replacement that allows more in the other set
                existing = set(S_temp) | set(other)
                for x in range(1, n + 1):
                    if x in existing:
                        continue
                    if can_add_to_S(S_temp, x, other_diffs):
                        S_new = S_temp + [x]
                        S_new_diffs = nonzero_diffs(S_new)
                        
                        # Check if this allows adding to the other set
                        other_new = other[:]
                        added_other = False
                        for y in range(1, n + 1):
                            if y in set(other_new) or y in set(S_new):
                                continue
                            if len(other_new) >= max_size:
                                break
                            if can_add_to_S(other_new, y, S_new_diffs):
                                other_new.append(y)
                                added_other = True
                                break
                        
                        if added_other:
                            if side == 0:
                                A = S_new
                                A_diffs = S_new_diffs
                                B = other_new
                                B_diffs = nonzero_diffs(B)
                            else:
                                B = S_new
                                B_diffs = S_new_diffs
                                A = other_new
                                A_diffs = nonzero_diffs(A)
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break

    return [A, B]
