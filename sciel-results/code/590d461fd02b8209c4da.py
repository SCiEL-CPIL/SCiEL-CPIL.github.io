# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    n = 256
    max_size_each = 64
    
    def check_sidon(aset):
        """Check if aset is a Sidon set (unique pair sums including a+a)."""
        sums = set()
        for i, a in enumerate(aset):
            for j in range(i, len(aset)):
                s = a + aset[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def get_diffs(aset):
        """Get all nonzero differences (positive) of aset."""
        diffs = set()
        for i in range(len(aset)):
            for j in range(i + 1, len(aset)):
                d = aset[j] - aset[i]
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def can_add_to_sidon(x, aset, existing_sums):
        """Check if x can be added to aset while maintaining Sidon property."""
        for a in aset:
            s = x + a
            if s in existing_sums:
                return False
        s2 = x + x
        if s2 in existing_sums:
            return False
        return True
    
    def get_sums(aset):
        """Get all pair sums (including a+a) of aset."""
        sums = set()
        for i, a in enumerate(aset):
            for j in range(i, len(aset)):
                sums.add(a + aset[j])
        return sums
    
    # Greedy construction with balanced approach
    def greedy_construct():
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in range(1, n + 1):
            # Try to add to A first
            can_A = True
            for a in A:
                if (x + a) in sums_A or (x + x) in sums_A:
                    can_A = False
                    break
            if can_A:
                # Check difference constraint
                new_diffs_A = set()
                for a in A:
                    d = x - a
                    if d > 0:
                        new_diffs_A.add(d)
                if not (new_diffs_A & diffs_B):
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    diffs_A |= new_diffs_A
            
            # Try to add to B
            can_B = True
            for b in B:
                if (x + b) in sums_B or (x + x) in sums_B:
                    can_B = False
                    break
            if can_B:
                new_diffs_B = set()
                for b in B:
                    d = x - b
                    if d > 0:
                        new_diffs_B.add(d)
                if not (new_diffs_B & diffs_A):
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    diffs_B |= new_diffs_B
        
        return A, B
    
    # Local search / polish
    def polish(A, B):
        """Try to improve by swapping or removing/adding elements."""
        A = list(A)
        B = list(B)
        
        def validate(A, B):
            if not check_sidon(A) or not check_sidon(B):
                return False
            dA = get_diffs(A)
            dB = get_diffs(B)
            if dA & dB:
                return False
            if len(A) > max_size_each or len(B) > max_size_each:
                return False
            return True
        
        improved = True
        while improved and (time.time() - start_time) < time_limit - 2:
            improved = False
            # Try removing an element from A and adding a different one
            for i in range(len(A)):
                if (time.time() - start_time) > time_limit - 2:
                    break
                old = A[i]
                A_temp = A[:i] + A[i+1:]
                sums_A_temp = get_sums(A_temp)
                diffs_A_temp = get_diffs(A_temp)
                diffs_B = get_diffs(B)
                
                for x in range(1, n + 1):
                    if x in A_temp or x == old:
                        continue
                    can_add = True
                    for a in A_temp:
                        if (x + a) in sums_A_temp or (x + x) in sums_A_temp:
                            can_add = False
                            break
                    if can_add:
                        new_diffs = set()
                        for a in A_temp:
                            d = x - a
                            if d > 0:
                                new_diffs.add(d)
                        if not (new_diffs & diffs_B):
                            A[i] = x
                            improved = True
                            break
                if improved:
                    break
            
            # Try removing an element from B and adding a different one
            if not improved:
                for i in range(len(B)):
                    if (time.time() - start_time) > time_limit - 2:
                        break
                    old = B[i]
                    B_temp = B[:i] + B[i+1:]
                    sums_B_temp = get_sums(B_temp)
                    diffs_B_temp = get_diffs(B_temp)
                    diffs_A = get_diffs(A)
                    
                    for x in range(1, n + 1):
                        if x in B_temp or x == old:
                            continue
                        can_add = True
                        for b in B_temp:
                            if (x + b) in sums_B_temp or (x + x) in sums_B_temp:
                                can_add = False
                                break
                        if can_add:
                            new_diffs = set()
                            for b in B_temp:
                                d = x - b
                                if d > 0:
                                    new_diffs.add(d)
                            if not (new_diffs & diffs_A):
                                B[i] = x
                                improved = True
                                break
                    if improved:
                        break
        
        return A, B
    
    # Main algorithm
    best_A, best_B = greedy_construct()
    best_size = len(best_A) + len(best_B)
    
    # Polish
    if (time.time() - start_time) < time_limit - 5:
        A_pol, B_pol = polish(best_A, best_B)
        if len(A_pol) + len(B_pol) > best_size:
            best_A, best_B = A_pol, B_pol
            best_size = len(best_A) + len(best_B)
    
    # Try multiple greedy runs with different orderings
    attempts = 0
    while (time.time() - start_time) < time_limit - 3 and attempts < 50:
        attempts += 1
        random.seed(attempts)
        order = list(range(1, n + 1))
        random.shuffle(order)
        
        A = []
        B = []
        sums_A = set()
        sums_B = set()
        diffs_A = set()
        diffs_B = set()
        
        for x in order:
            # Try to add to A
            can_A = True
            for a in A:
                if (x + a) in sums_A or (x + x) in sums_A:
                    can_A = False
                    break
            if can_A:
                new_diffs_A = set()
                for a in A:
                    d = x - a
                    if d > 0:
                        new_diffs_A.add(d)
                if not (new_diffs_A & diffs_B):
                    A.append(x)
                    sums_A.add(x + x)
                    for a in A[:-1]:
                        sums_A.add(x + a)
                    diffs_A |= new_diffs_A
            
            # Try to add to B
            can_B = True
            for b in B:
                if (x + b) in sums_B or (x + x) in sums_B:
                    can_B = False
                    break
            if can_B:
                new_diffs_B = set()
                for b in B:
                    d = x - b
                    if d > 0:
                        new_diffs_B.add(d)
                if not (new_diffs_B & diffs_A):
                    B.append(x)
                    sums_B.add(x + x)
                    for b in B[:-1]:
                        sums_B.add(x + b)
                    diffs_B |= new_diffs_B
        
        curr_size = len(A) + len(B)
        if curr_size > best_size:
            best_A, best_B = A, B
            best_size = curr_size
            
            # Polish the new best
            if (time.time() - start_time) < time_limit - 5:
                A_pol, B_pol = polish(best_A, best_B)
                if len(A_pol) + len(B_pol) > best_size:
                    best_A, best_B = A_pol, B_pol
                    best_size = len(best_A) + len(best_B)
    
    # Final validation
    if not check_sidon(best_A) or not check_sidon(best_B):
        # Fallback to simple valid sets
        best_A = [1, 2, 4, 8]
        best_B = [3, 5, 9, 17]
    
    dA = get_diffs(best_A)
    dB = get_diffs(best_B)
    if dA & dB:
        best_A = [1, 2, 4, 8]
        best_B = [3, 5, 9, 17]
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
