import random
import sys
from itertools import combinations, combinations_with_replacement
import heapq

def construct():
    n = 256
    max_size = 64
    random.seed(42)

    def is_sidon(S):
        if len(S) <= 1:
            return True
        sums = set()
        for a in S:
            for b in S:
                if a <= b:
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

    def can_add(S, x, other_diffs):
        # Efficient Sidon check
        sums = set()
        for a in S:
            for b in S:
                if a <= b:
                    sums.add(a + b)
        
        # Check x+x
        if x + x in sums:
            return False
        # Check x+s
        for s in S:
            if x + s in sums:
                return False
        
        # Check Diffs
        for s in S:
            d1 = x - s
            d2 = s - x
            if d1 in other_diffs or d2 in other_diffs:
                return False
                
        return True

    best_total = 0
    best_A = []
    best_B = []

    for trial in range(50):
        random.seed(trial)
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A = []
        B = []
        
        for x in order:
            if len(A) < max_size:
                if can_add(A, x, nonzero_diffs(B)):
                    A.append(x)
                    continue
            if len(B) < max_size:
                if can_add(B, x, nonzero_diffs(A)):
                    B.append(x)
                    continue
        
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]

    A = best_A[:]
    B = best_B[:]

    # Local Search
    def local_search(A, B):
        improved = True
        while improved:
            improved = False
            
            # 1. Add
            assigned = set(A) | set(B)
            candidates = [x for x in range(1, n+1) if x not in assigned]
            random.shuffle(candidates)
            
            for x in candidates:
                if len(A) < max_size and can_add(A, x, nonzero_diffs(B)):
                    A.append(x)
                    assigned.add(x)
                    improved = True
                    break
                if len(B) < max_size and can_add(B, x, nonzero_diffs(A)):
                    B.append(x)
                    assigned.add(x)
                    improved = True
                    break
            
            if improved:
                continue

            # 2. Move A -> B
            moved = False
            for x in A[:]:
                A_temp = [a for a in A if a != x]
                if len(B) < max_size and can_add(B, x, nonzero_diffs(A_temp)):
                    A = A_temp
                    B.append(x)
                    moved = True
                    improved = True
                    break
            if moved:
                continue

            # 3. Move B -> A
            moved = False
            for x in B[:]:
                B_temp = [b for b in B if b != x]
                if len(A) < max_size and can_add(A, x, nonzero_diffs(B_temp)):
                    B = B_temp
                    A.append(x)
                    moved = True
                    improved = True
                    break
            if moved:
                continue

            # 4. Swap (Replace a in A with x, if it allows adding to B)
            swapped = False
            for a in A[:]:
                A_temp = [u for u in A if u != a]
                candidates = [x for x in range(1, n+1) if x not in A_temp and x not in B]
                random.shuffle(candidates)
                
                for x in candidates[:20]:
                    if can_add(A_temp, x, nonzero_diffs(B)):
                        A_new = A_temp + [x]
                        B_new = B[:]
                        added = False
                        for y in range(1, n+1):
                            if y in B_new or y in A_new:
                                continue
                            if len(B_new) < max_size and can_add(B_new, y, nonzero_diffs(A_new)):
                                B_new.append(y)
                                added = True
                                break
                        if added:
                            A = A_new
                            B = B_new
                            swapped = True
                            improved = True
                            break
                if swapped:
                    break
            if swapped:
                continue

            # 5. Swap (Replace b in B with x, if it allows adding to A)
            swapped = False
            for b in B[:]:
                B_temp = [u for u in B if u != b]
                candidates = [x for x in range(1, n+1) if x not in B_temp and x not in A]
                random.shuffle(candidates)
                
                for x in candidates[:20]:
                    if can_add(B_temp, x, nonzero_diffs(A)):
                        B_new = B_temp + [x]
                        A_new = A[:]
                        added = False
                        for y in range(1, n+1):
                            if y in A_new or y in B_new:
                                continue
                            if len(A_new) < max_size and can_add(A_new, y, nonzero_diffs(B_new)):
                                A_new.append(y)
                                added = True
                                break
                        if added:
                            B = B_new
                            A = A_new
                            swapped = True
                            improved = True
                            break
                if swapped:
                    break

    local_search(A, B)

    return [A, B]
