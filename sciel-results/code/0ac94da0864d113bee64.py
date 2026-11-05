"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def _is_sidon(arr):
        """Check if arr is a Sidon set (unique pair sums including a+a)."""
        if len(arr) < 2:
            return True
        sums = set()
        for i in range(len(arr)):
            for j in range(i, len(arr)):
                s = arr[i] + arr[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def _diff_set(arr):
        """Compute nonzero difference set of arr."""
        diffs = set()
        for i in range(len(arr)):
            for j in range(len(arr)):
                if i != j:
                    d = arr[j] - arr[i]
                    if d != 0:
                        diffs.add(d)
        return diffs
    
    def _greedy_build(order, max_size=64):
        """
        Build two Sidon sets greedily following the given order.
        Returns (A, B) as lists.
        """
        A = []
        B = []
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            # Try adding to A
            if len(A) < max_size:
                candidate_A = A + [x]
                if _is_sidon(candidate_A):
                    new_diffs = _diff_set(candidate_A)
                    if not (new_diffs & diffs_B):
                        A = candidate_A
                        diffs_A = new_diffs
                        continue
            
            # Try adding to B
            if len(B) < max_size:
                candidate_B = B + [x]
                if _is_sidon(candidate_B):
                    new_diffs = _diff_set(candidate_B)
                    if not (new_diffs & diffs_A):
                        B = candidate_B
                        diffs_B = new_diffs
        
        return A, B
    
    def _local_search(A, B, deadline):
        """
        Local search: try to improve by swapping elements or moving elements.
        Returns improved (A, B).
        """
        A = list(A)
        B = list(B)
        diffs_A = _diff_set(A)
        diffs_B = _diff_set(B)
        improved = True
        
        while improved and time.time() < deadline - 1.0:
            improved = False
            
            # Try swapping an element from A with one not in A or B
            # Or moving an element between sets
            candidates = [x for x in range(1, n+1) if x not in A and x not in B]
            
            random.shuffle(candidates)
            
            for x in candidates:
                if time.time() > deadline - 1.0:
                    break
                
                # Try adding x to A
                if len(A) < 64:
                    candidate_A = A + [x]
                    if _is_sidon(candidate_A):
                        new_diffs_A = _diff_set(candidate_A)
                        if not (new_diffs_A & diffs_B):
                            A = candidate_A
                            diffs_A = new_diffs_A
                            improved = True
                            break
                
                # Try adding x to B
                if len(B) < 64:
                    candidate_B = B + [x]
                    if _is_sidon(candidate_B):
                        new_diffs_B = _diff_set(candidate_B)
                        if not (new_diffs_B & diffs_A):
                            B = candidate_B
                            diffs_B = new_diffs_B
                            improved = True
                            break
            
            # Try removing an element and adding a different one
            if not improved:
                # Try to replace an element in A
                for i in range(len(A)):
                    if time.time() > deadline - 1.0:
                        break
                    old = A[i]
                    for x in range(1, n+1):
                        if x == old or x in A or x in B:
                            continue
                        candidate_A = A[:i] + A[i+1:] + [x]
                        candidate_A.sort()
                        if _is_sidon(candidate_A):
                            new_diffs_A = _diff_set(candidate_A)
                            if not (new_diffs_A & diffs_B):
                                A = candidate_A
                                diffs_A = new_diffs_A
                                improved = True
                                break
                    if improved:
                        break
                
                # Try to replace an element in B
                if not improved:
                    for i in range(len(B)):
                        if time.time() > deadline - 1.0:
                            break
                        old = B[i]
                        for x in range(1, n+1):
                            if x == old or x in A or x in B:
                                continue
                            candidate_B = B[:i] + B[i+1:] + [x]
                            candidate_B.sort()
                            if _is_sidon(candidate_B):
                                new_diffs_B = _diff_set(candidate_B)
                                if not (new_diffs_B & diffs_A):
                                    B = candidate_B
                                    diffs_B = new_diffs_B
                                    improved = True
                                    break
                        if improved:
                            break
        
        return A, B
    
    # Initial greedy with natural order
    order = list(range(1, n+1))
    A, B = _greedy_build(order)
    
    # Try several random orderings
    best_A, best_B = A, B
    best_size = len(A) + len(B)
    
    rng = random.Random(42)
    
    while time.time() < deadline - 10.0:
        # Generate a random ordering with some structure
        order = list(range(1, n+1))
        rng.shuffle(order)
        
        A, B = _greedy_build(order)
        current_size = len(A) + len(B)
        
        if current_size > best_size:
            best_size = current_size
            best_A, best_B = A, B
        
        # Do local search on the best result
        if current_size >= best_size - 2:
            A2, B2 = _local_search(A, B, deadline)
            size2 = len(A2) + len(B2)
            if size2 > best_size:
                best_size = size2
                best_A, best_B = A2, B2
    
    # Final local search on the best
    if time.time() < deadline - 2.0:
        best_A, best_B = _local_search(best_A, best_B, deadline)
    
    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
