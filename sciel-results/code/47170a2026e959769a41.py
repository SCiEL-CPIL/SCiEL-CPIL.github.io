"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def _greedy_flexible(order, deadline, prefer_balance=True):
        """
        Forward pass where each element can go to either set.
        If prefer_balance, prefer the smaller set when both are valid.
        """
        setA = []
        setB = []
        sumsA = set()
        sumsB = set()
        diffsA = set()
        diffsB = set()
        
        for x in order:
            if time.time() > deadline:
                break
            
            # Check if x can be added to A
            can_add_A = True
            for a in setA:
                s = a + x
                if s in sumsA:
                    can_add_A = False
                    break
                d = abs(a - x)
                if d in diffsB:
                    can_add_A = False
                    break
            if can_add_A:
                if 2*x in sumsA:
                    can_add_A = False
            
            # Check if x can be added to B
            can_add_B = True
            for b in setB:
                s = b + x
                if s in sumsB:
                    can_add_B = False
                    break
                d = abs(b - x)
                if d in diffsA:
                    can_add_B = False
                    break
            if can_add_B:
                if 2*x in sumsB:
                    can_add_B = False
            
            # Decide where to put x
            if can_add_A and can_add_B:
                if prefer_balance:
                    if len(setA) <= len(setB):
                        target = 'A'
                    else:
                        target = 'B'
                else:
                    target = 'A' if len(setA) < len(setB) else 'B'
            elif can_add_A:
                target = 'A'
            elif can_add_B:
                target = 'B'
            else:
                continue
            
            if target == 'A':
                for a in setA:
                    sumsA.add(a + x)
                    diffsA.add(abs(a - x))
                sumsA.add(2*x)
                setA.append(x)
            else:
                for b in setB:
                    sumsB.add(b + x)
                    diffsB.add(abs(b - x))
                sumsB.add(2*x)
                setB.append(x)
        
        return sorted(setA), sorted(setB)
    
    def _local_search(setA, setB, deadline, max_iters=500):
        """
        Local search: try replacing one element in a set with another.
        """
        setA = list(setA)
        setB = list(setB)
        best_size = len(setA) + len(setB)
        
        def can_add_to_set(x, s, sums, diffs_other, diffs_self):
            """Check if x can be added to set s."""
            for a in s:
                if a + x in sums:
                    return False
                if abs(a - x) in diffs_other:
                    return False
            if 2*x in sums:
                return False
            return True
        
        def get_sums_diffs(s):
            sums = set()
            diffs = set()
            for i in range(len(s)):
                for j in range(i, len(s)):
                    sums.add(s[i] + s[j])
                for j in range(len(s)):
                    if i != j:
                        diffs.add(abs(s[i] - s[j]))
            return sums, diffs
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Compute current sums and diffs
                sumsA, diffsA = get_sums_diffs(setA)
                sumsB, diffsB = get_sums_diffs(setB)
                
                # Try to add a new element to A (without removing anything)
                for x in range(1, n+1):
                    if x in setA or x in setB:
                        continue
                    if can_add_to_set(x, setA, sumsA, diffsB, diffsA):
                        setA.append(x)
                        setA.sort()
                        best_size += 1
                        improved = True
                        break
                if improved:
                    continue
                
                # Try to add a new element to B
                for x in range(1, n+1):
                    if x in setA or x in setB:
                        continue
                    if can_add_to_set(x, setB, sumsB, diffsA, diffsB):
                        setB.append(x)
                        setB.sort()
                        best_size += 1
                        improved = True
                        break
                if improved:
                    continue
                
                # Try removing one from A and adding two to A
                if len(setA) >= 2:
                    for rem_idx in range(len(setA)):
                        rem = setA[rem_idx]
                        newA = setA[:rem_idx] + setA[rem_idx+1:]
                        sumsA_new, diffsA_new = get_sums_diffs(newA)
                        for x in range(1, n+1):
                            if x in newA or x in setB:
                                continue
                            if can_add_to_set(x, newA, sumsA_new, diffsB, diffsA_new):
                                # Check if adding x also allows another element
                                testA = newA + [x]
                                testA.sort()
                                sumsA_test, diffsA_test = get_sums_diffs(testA)
                                added = 0
                                for y in range(1, n+1):
                                    if y in testA or y in setB:
                                        continue
                                    if can_add_to_set(y, testA, sumsA_test, diffsB, diffsA_test):
                                        testA.append(y)
                                        testA.sort()
                                        added += 1
                                        if added >= 2:
                                            break
                                if added >= 1:
                                    setA = testA
                                    best_size = len(setA) + len(setB)
                                    improved = True
                                    break
                        if improved:
                            break
                if improved:
                    continue
                
                # Try removing one from B and adding to B
                if len(setB) >= 2:
                    for rem_idx in range(len(setB)):
                        rem = setB[rem_idx]
                        newB = setB[:rem_idx] + setB[rem_idx+1:]
                        sumsB_new, diffsB_new = get_sums_diffs(newB)
                        for x in range(1, n+1):
                            if x in newB or x in setA:
                                continue
                            if can_add_to_set(x, newB, sumsB_new, diffsA, diffsB_new):
                                testB = newB + [x]
                                testB.sort()
                                sumsB_test, diffsB_test = get_sums_diffs(testB)
                                added = 0
                                for y in range(1, n+1):
                                    if y in testB or y in setA:
                                        continue
                                    if can_add_to_set(y, testB, sumsB_test, diffsA, diffsB_test):
                                        testB.append(y)
                                        testB.sort()
                                        added += 1
                                        if added >= 2:
                                            break
                                if added >= 1:
                                    setB = testB
                                    best_size = len(setA) + len(setB)
                                    improved = True
                                    break
                        if improved:
                            break
        
        return sorted(setA), sorted(setB)
    
    # Multiple random restarts with different strategies
    best_sets = ([1], [2])
    best_size = 2
    
    rng = random.Random(42)
    
    for restart in range(300):
        if time.time() > deadline:
            break
        
        order = list(range(1, n+1))
        rng.shuffle(order)
        
        # Alternate between balanced and unbalanced greedy
        prefer_balance = (restart % 2 == 0)
        
        setA, setB = _greedy_flexible(order, deadline, prefer_balance)
        current_size = len(setA) + len(setB)
        
        if current_size > best_size:
            best_sets = (setA, setB)
            best_size = current_size
        
        # Local search on promising solutions
        if current_size >= best_size - 3:
            setA_ls, setB_ls = _local_search(setA, setB, deadline)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size
    
    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
