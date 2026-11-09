import time
import random
import numpy as np

N = 512
DEADLINE_MARGIN = 5

def construct():
    t0 = time.time()
    deadline = t0 + DEADLINE_MARGIN

    def elapsed_ok():
        return (time.time() - t0) < deadline

    # Pre-compute all 4-APs in [1, N]
    all_aps = []
    ap_index = {} # map number -> list of ap indices
    degrees = np.zeros(N + 1, dtype=int)
    
    for d in range(1, (N - 1) // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap = (a, a + d, a + 2 * d, a + 3 * d)
            idx = len(all_aps)
            all_aps.append(ap)
            for x in ap:
                if x not in ap_index:
                    ap_index[x] = []
                ap_index[x].append(idx)
                degrees[x] += 1
            
    num_aps = len(all_aps)
    
    # Helper to check if adding x is valid given current in_set
    def can_add(x, in_set):
        if x not in ap_index:
            return True
        for idx in ap_index[x]:
            ap = all_aps[idx]
            # Check if the other 3 elements are in set
            count = 0
            for y in ap:
                if y != x:
                    if in_set[y]:
                        count += 1
                    else:
                        break
            if count == 3:
                return False
        return True

    best_set = set()
    best_size = 0
    
    # Heuristic 1: Start from Base-4 set, add in Min-Degree order
    # Heuristic 2: Start from Base-4 set, add in Max-Degree order
    # Heuristic 3: Start from Empty set, add in Min-Degree order
    # Heuristic 4: Start from Empty set, add in Random order (multiple restarts)
    
    # Pre-compute orders
    numbers = list(range(1, N + 1))
    min_deg_order = sorted(numbers, key=lambda x: degrees[x])
    max_deg_order = sorted(numbers, key=lambda x: -degrees[x])
    
    def run_greedy(start_set, order):
        in_set = np.zeros(N + 1, dtype=bool)
        for x in start_set:
            in_set[x] = True
            
        # We need to add elements from 'order' that are not in start_set
        # Or just iterate 'order' and add if not in set and valid
        # Note: If start_set is not independent, this fails. 
        # Base-4 set is independent. Empty set is independent.
        
        for x in order:
            if not elapsed_ok():
                break
            if not in_set[x] and can_add(x, in_set):
                in_set[x] = True
                
        return {x for x in range(1, N + 1) if in_set[x]}

    # Base-4 set
    base4_set = set()
    for i in range(1, N + 1):
        x = i
        valid = True
        while x > 0:
            digit = x % 4
            if digit > 1:
                valid = False
                break
            x //= 4
        if valid:
            base4_set.add(i)

    # Run heuristics
    # 1. Base4 + MinDeg
    s = run_greedy(base4_set, min_deg_order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # 2. Base4 + MaxDeg
    s = run_greedy(base4_set, max_deg_order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # 3. Empty + MinDeg
    s = run_greedy(set(), min_deg_order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # 4. Empty + MaxDeg
    s = run_greedy(set(), max_deg_order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # 5. Random restarts from Base4
    rng = random.Random(42)
    for trial in range(200):
        if not elapsed_ok():
            break
        candidates = [x for x in range(1, N + 1) if x not in base4_set]
        rng.shuffle(candidates)
        s = run_greedy(base4_set, candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # 6. Random restarts from Empty
    for trial in range(200):
        if not elapsed_ok():
            break
        candidates = list(range(1, N + 1))
        rng.shuffle(candidates)
        s = run_greedy(set(), candidates)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
            
    return sorted(best_set)
