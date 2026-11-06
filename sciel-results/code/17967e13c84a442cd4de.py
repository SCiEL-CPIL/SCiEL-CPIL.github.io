import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    N = 128

    # Precompute bad triples
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            for c in range(b + 1, N + 1):
                g_ab = gcd(a, b)
                g_bc = gcd(b, c)
                g_ac = gcd(a, c)
                if g_ab == g_bc == g_ac:
                    bad_triples.append((a, b, c))

    conflicts = {x: [] for x in range(1, N + 1)}
    for (a, b, c) in bad_triples:
        conflicts[a].append((b, c))
        conflicts[b].append((a, c))
        conflicts[c].append((a, b))

    def is_valid_add(S, x):
        S_set = set(S)
        for (u, v) in conflicts[x]:
            if u in S_set and v in S_set:
                return False
        return True

    def is_valid_set(S):
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    a, b, c = S_list[i], S_list[j], S_list[k]
                    if gcd(a, b) == gcd(b, c) == gcd(a, c):
                        return False
        return True

    best_set = set()
    best_size = 0

    # Greedy Construction
    orders = [
        list(range(1, N + 1)),
        list(range(N, 0, -1)),
        sorted(range(1, N + 1), key=lambda x: len(conflicts[x]))
    ]
    
    for _ in range(10):
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        orders.append(random_order)

    for order in orders:
        S = set()
        for x in order:
            if is_valid_add(S, x):
                S.add(x)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S.copy()

    S = best_set
    current_size = len(S)

    # Local Search: 1-for-2 exchanges
    # Try to remove one element u from S and add two elements x, y not in S.
    # This increases size by 1.
    
    random.seed(42)
    
    while time.time() < deadline:
        improved = False
        
        # Try to add single elements first (fast)
        for x in range(1, N + 1):
            if x not in S:
                if is_valid_add(S, x):
                    S.add(x)
                    current_size += 1
                    improved = True
                    break
        
        if improved:
            continue
            
        # Try 1-for-2 exchanges
        # Pick a random u in S
        if S:
            u = random.choice(list(S))
            candidates = [x for x in range(1, N + 1) if x not in S]
            
            # Try pairs of candidates
            # To keep it fast, limit the number of pairs tried per iteration
            if len(candidates) >= 2:
                # Shuffle candidates to randomize
                random.shuffle(candidates)
                # Only try a subset of pairs
                max_pairs_to_try = 100
                pairs_tried = 0
                
                for i in range(len(candidates)):
                    if pairs_tried >= max_pairs_to_try:
                        break
                    for j in range(i + 1, len(candidates)):
                        if pairs_tried >= max_pairs_to_try:
                            break
                        pairs_tried += 1
                        
                        x = candidates[i]
                        y = candidates[j]
                        
                        # Check if S - {u} + {x, y} is valid
                        # 1. S - {u} is valid (subset of valid S)
                        # 2. Check if adding x to S - {u} is valid
                        # 3. Check if adding y to S - {u} + {x} is valid
                        
                        temp_S = set(S)
                        temp_S.remove(u)
                        
                        if is_valid_add(temp_S, x):
                            temp_S.add(x)
                            if is_valid_add(temp_S, y):
                                # Valid!
                                S = temp_S
                                S.add(y)
                                current_size += 1
                                improved = True
                                break
                    if improved:
                        break
        
        if not improved:
            # If no 1-for-2 worked, try 1-for-1 swaps to escape local minima?
            # Or just break if we've been stuck for a while?
            # For now, let's try a few random 1-for-1 swaps
            if S:
                u = random.choice(list(S))
                candidates = [x for x in range(1, N + 1) if x not in S]
                if candidates:
                    v = random.choice(candidates)
                    temp_S = set(S)
                    temp_S.remove(u)
                    if is_valid_add(temp_S, v):
                        S = temp_S
                        S.add(v)
                        # Size unchanged, but structure changed
                        # Do not set improved = True to avoid infinite loop on same size
                        # But we might loop. Add a counter?
                        pass

    return list(S)
