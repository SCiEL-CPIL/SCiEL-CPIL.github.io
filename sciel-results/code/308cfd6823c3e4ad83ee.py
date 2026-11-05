"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 90s limit
    n = 256
    
    def _greedy_one_ordering(order, deadline):
        """
        Single forward pass over the given ordering.
        Returns (setA, setB) as sorted lists.
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
            
            # Try adding x to setA
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
                # Check x+x
                if 2*x in sumsA:
                    can_add_A = False
            if can_add_A:
                # Add to A
                for a in setA:
                    sumsA.add(a + x)
                    d = abs(a - x)
                    diffsA.add(d)
                sumsA.add(2*x)
                setA.append(x)
            else:
                # Try adding x to setB
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
                if can_add_B:
                    for b in setB:
                        sumsB.add(b + x)
                        d = abs(b - x)
                        diffsB.add(d)
                    sumsB.add(2*x)
                    setB.append(x)
        
        return sorted(setA), sorted(setB)
    
    def _local_search(setA, setB, deadline, max_swaps=200):
        """
        Try to improve by removing elements from one set and adding from the other,
        or by swapping elements between sets.
        """
        setA = list(setA)
        setB = list(setB)
        
        def rebuild(sA, sB):
            sumsA = set()
            sumsB = set()
            diffsA = set()
            diffsB = set()
            for i, a in enumerate(sA):
                for j in range(i+1):
                    sumsA.add(sA[i] + sA[j])
                for b in sB:
                    diffsA.add(abs(a - b))
            for i, b in enumerate(sB):
                for j in range(i+1):
                    sumsB.add(sB[i] + sB[j])
                for a in sA:
                    diffsB.add(abs(a - b))
            return sumsA, sumsB, diffsA, diffsB
        
        best_size = len(setA) + len(setB)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for _ in range(max_swaps):
                if time.time() > deadline:
                    break
                # Try swapping: remove random from A, add random from B, or vice versa
                if random.random() < 0.5:
                    if len(setA) > 0:
                        idx = random.randint(0, len(setA)-1)
                        removed = setA[idx]
                        newA = setA[:idx] + setA[idx+1:]
                        # Try adding a random element from B or unused to A
                        candidates = [x for x in setB if x != removed] + [x for x in range(1, n+1) if x not in setA and x not in setB]
                        if not candidates:
                            continue
                        cand = random.choice(candidates)
                        testA = sorted(newA + [cand])
                        # Quick check
                        sums = set()
                        valid = True
                        for i in range(len(testA)):
                            for j in range(i, len(testA)):
                                s = testA[i] + testA[j]
                                if s in sums:
                                    valid = False
                                    break
                                sums.add(s)
                            if not valid:
                                break
                        if valid:
                            # Check diffs against B
                            diffs = set()
                            for a in testA:
                                for b in setB:
                                    d = abs(a - b)
                                    if d in diffs:
                                        valid = False
                                        break
                                    diffs.add(d)
                            if valid and len(testA) + len(setB) > best_size:
                                setA = testA
                                best_size = len(setA) + len(setB)
                                improved = True
                else:
                    if len(setB) > 0:
                        idx = random.randint(0, len(setB)-1)
                        removed = setB[idx]
                        newB = setB[:idx] + setB[idx+1:]
                        candidates = [x for x in setA if x != removed] + [x for x in range(1, n+1) if x not in setA and x not in setB]
                        if not candidates:
                            continue
                        cand = random.choice(candidates)
                        testB = sorted(newB + [cand])
                        sums = set()
                        valid = True
                        for i in range(len(testB)):
                            for j in range(i, len(testB)):
                                s = testB[i] + testB[j]
                                if s in sums:
                                    valid = False
                                    break
                                sums.add(s)
                            if not valid:
                                break
                        if valid:
                            diffs = set()
                            for b in testB:
                                for a in setA:
                                    d = abs(a - b)
                                    if d in diffs:
                                        valid = False
                                        break
                                    diffs.add(d)
                            if valid and len(setA) + len(testB) > best_size:
                                setB = testB
                                best_size = len(setA) + len(setB)
                                improved = True
        return sorted(setA), sorted(setB)
    
    # Multiple random restarts
    best_sets = ([1], [2])
    best_size = 2
    
    rng = random.Random(42)
    
    for restart in range(500):
        if time.time() > deadline:
            break
        
        order = list(range(1, n+1))
        rng.shuffle(order)
        
        setA, setB = _greedy_one_ordering(order, deadline)
        current_size = len(setA) + len(setB)
        
        if current_size > best_size:
            best_sets = (setA, setB)
            best_size = current_size
        
        # Local search on the best found so far
        if current_size >= best_size - 2:
            setA_ls, setB_ls = _local_search(setA, setB, deadline)
            ls_size = len(setA_ls) + len(setB_ls)
            if ls_size > best_size:
                best_sets = (setA_ls, setB_ls)
                best_size = ls_size
    
    return [best_sets[0], best_sets[1]]
# EVOLVE-BLOCK-END
