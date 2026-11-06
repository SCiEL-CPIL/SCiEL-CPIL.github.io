# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    """
    Construct two nonempty Sidon subsets A, B of [1, 256] with at most 64 elements each.
    Pair sums inside each set (including a+a) must be unique.
    Nonzero difference sets of A and B must be disjoint.
    Maximize |A|+|B|.
    """
    start_time = time.time()
    DEADLINE = 50.0  # safety margin before 60s hard kill
    
    N = 256
    MAX_SIZE = 64
    
    def _pair_sums_unique(elements):
        """Check if all pair sums (with replacement) are unique."""
        sums = set()
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = elements[i] + elements[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def _diff_set(elements):
        """Get the set of nonzero differences (absolute value)."""
        diffs = set()
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                d = abs(elements[i] - elements[j])
                diffs.add(d)
        return diffs
    
    def _try_add(elements, candidate, used_sums, used_diffs, other_diffs):
        """
        Try to add candidate to elements.
        Returns (success, new_elements, new_used_sums, new_used_diffs) or None.
        """
        # Check pair sums with existing elements
        for e in elements:
            s = e + candidate
            if s in used_sums:
                return None
            d = abs(e - candidate)
            if d in used_diffs:
                return None
            if d in other_diffs:
                return None
        # Check self-sum
        s_self = 2 * candidate
        if s_self in used_sums:
            return None
        # Check self-diff (always 0, which is not in nonzero diffs)
        
        # Build new structures
        new_elements = elements + [candidate]
        new_used_sums = set(used_sums)
        for e in elements:
            new_used_sums.add(e + candidate)
        new_used_sums.add(s_self)
        
        new_used_diffs = set(used_diffs)
        for e in elements:
            new_used_diffs.add(abs(e - candidate))
        
        return (True, new_elements, new_used_sums, new_used_diffs)
    
    def _greedy_construct(order_a, order_b, max_iterations=200000):
        """
        Greedy construction with given candidate orders.
        Returns (A, B, total_size).
        """
        A = []
        B = []
        sums_a = set()
        sums_b = set()
        diffs_a = set()
        diffs_b = set()
        
        for x in order_a:
            if len(A) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            # Try to add x to A
            success = False
            for e in A:
                s = e + x
                if s in sums_a:
                    break
                d = abs(e - x)
                if d in diffs_a or d in diffs_b:
                    break
            else:
                if (2 * x) not in sums_a:
                    success = True
            
            if success:
                new_sums = set(sums_a)
                new_diffs = set(diffs_a)
                for e in A:
                    new_sums.add(e + x)
                    new_diffs.add(abs(e - x))
                new_sums.add(2 * x)
                A.append(x)
                sums_a = new_sums
                diffs_a = new_diffs
        
        for x in order_b:
            if len(B) >= MAX_SIZE:
                break
            if time.time() - start_time > DEADLINE:
                break
            success = False
            for e in B:
                s = e + x
                if s in sums_b:
                    break
                d = abs(e - x)
                if d in diffs_b or d in diffs_a:
                    break
            else:
                if (2 * x) not in sums_b:
                    success = True
            
            if success:
                new_sums = set(sums_b)
                new_diffs = set(diffs_b)
                for e in B:
                    new_sums.add(e + x)
                    new_diffs.add(abs(e - x))
                new_sums.add(2 * x)
                B.append(x)
                sums_b = new_sums
                diffs_b = new_diffs
        
        return A, B, len(A) + len(B)
    
    def _local_search(A, B, max_rounds=50):
        """
        Try to improve by swapping elements.
        """
        A = list(A)
        B = list(B)
        best_size = len(A) + len(B)
        
        def _rebuild_sums_diffs(elements):
            sums = set()
            diffs = set()
            for i in range(len(elements)):
                for j in range(i, len(elements)):
                    sums.add(elements[i] + elements[j])
                for j in range(i + 1, len(elements)):
                    diffs.add(abs(elements[i] - elements[j]))
            return sums, diffs
        
        for _ in range(max_rounds):
            if time.time() - start_time > DEADLINE:
                break
            
            improved = False
            sums_a, diffs_a = _rebuild_sums_diffs(A)
            sums_b, diffs_b = _rebuild_sums_diffs(B)
            
            # Try removing an element from A and adding a better one
            for i in range(len(A)):
                orig_a = A[i]
                A_temp = A[:i] + A[i+1:]
                sums_a_temp, diffs_a_temp = _rebuild_sums_diffs(A_temp)
                
                # Find a candidate to replace
                for x in range(1, N + 1):
                    if x in A_temp or x in B:
                        continue
                    if x == orig_a:
                        continue
                    
                    # Check if x can be added to A_temp
                    valid = True
                    for e in A_temp:
                        s = e + x
                        if s in sums_a_temp:
                            valid = False
                            break
                        d = abs(e - x)
                        if d in diffs_a_temp or d in diffs_b:
                            valid = False
                            break
                    if valid and (2 * x) not in sums_a_temp:
                        A_new = A_temp + [x]
                        if len(A_new) > len(A):  # only if we gained
                            A = A_new
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Try adding to A
                for x in range(1, N + 1):
                    if x in A or x in B:
                        continue
                    valid = True
                    for e in A:
                        s = e + x
                        if s in sums_a:
                            valid = False
                            break
                        d = abs(e - x)
                        if d in diffs_a or d in diffs_b:
                            valid = False
                            break
                    if valid and (2 * x) not in sums_a and len(A) < MAX_SIZE:
                        A.append(x)
                        improved = True
                        break
            
            if not improved:
                # Try adding to B
                for x in range(1, N + 1):
                    if x in A or x in B:
                        continue
                    valid = True
                    for e in B:
                        s = e + x
                        if s in sums_b:
                            valid = False
                            break
                        d = abs(e - x)
                        if d in diffs_b or d in diffs_a:
                            valid = False
                            break
                    if valid and (2 * x) not in sums_b and len(B) < MAX_SIZE:
                        B.append(x)
                        improved = True
                        break
            
            if not improved:
                break
        
        return A, B, len(A) + len(B)
    
    # Main search loop with random restarts
    best_A = [1]
    best_B = [2]
    best_total = 2
    
    # First, do a deterministic greedy pass
    order_a = list(range(1, N + 1))
    order_b = list(range(1, N + 1))
    A, B, total = _greedy_construct(order_a, order_b)
    if total > best_total:
        best_A, best_B, best_total = A, B, total
    
    # Random restarts
    random.seed(42)
    for restart in range(500):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate random orders
        order_a = list(range(1, N + 1))
        order_b = list(range(1, N + 1))
        random.shuffle(order_a)
        random.shuffle(order_b)
        
        A, B, total = _greedy_construct(order_a, order_b)
        
        if total > best_total:
            # Try local search to improve
            A_ls, B_ls, total_ls = _local_search(A, B)
            if total_ls > best_total:
                best_A, best_B, best_total = A_ls, B_ls, total_ls
            else:
                best_A, best_B, best_total = A, B, total
    
    # Final local search on best solution
    A_final, B_final, total_final = _local_search(best_A, best_B)
    if total_final > best_total:
        best_A, best_B, best_total = A_final, B_final, total_final
    
    # Ensure nonempty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
