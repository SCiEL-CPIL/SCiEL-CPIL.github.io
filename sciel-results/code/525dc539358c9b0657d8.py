import random
import time
from math import gcd

def construct():
    N = 128
    elements = list(range(1, N + 1))
    
    # Precompute GCDs
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            gcds[i][j] = g
            gcds[j][i] = g
            
    def is_valid_addition(S, x):
        # S is a set of ints
        # Check if adding x creates an invalid triple
        # Triple (a, b, x) is invalid if gcd(a,b) == gcd(a,x) == gcd(b,x)
        # We iterate over all pairs (a, b) in S
        # To speed up, we can list S once
        S_list = list(S)
        len_S = len(S_list)
        for i in range(len_S):
            a = S_list[i]
            g_ax = gcds[a][x]
            for j in range(i + 1, len_S):
                b = S_list[j]
                if gcds[a][b] == g_ax and gcds[b][x] == g_ax:
                    return False
        return True

    def local_search(S, time_limit):
        S = set(S)
        start_time = time.time()
        improved = True
        # Pre-shuffle elements for random selection
        shuffled_elements = elements[:]
        random.shuffle(shuffled_elements)
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            
            # 1. Try to add any element
            # We can try elements in shuffled order
            for x in shuffled_elements:
                if x in S:
                    continue
                if is_valid_addition(S, x):
                    S.add(x)
                    improved = True
                    break
                    
            if improved:
                continue

            # 2. Try 1-for-1 swaps
            # Remove a, add x
            # This helps rearrange.
            if S:
                # Try a few random swaps
                for _ in range(20):
                    if time.time() - start_time > time_limit:
                        break
                    a = random.choice(list(S))
                    S.remove(a)
                    # Try to add any x not in S
                    added = False
                    for x in shuffled_elements:
                        if x in S or x == a:
                            continue
                        if is_valid_addition(S, x):
                            S.add(x)
                            added = True
                            improved = True
                            break
                    if not added:
                        S.add(a)
                    if added:
                        break

            if improved:
                continue

            # 3. Try 1-for-2 swaps
            # Remove a, add x, y
            # This increases size.
            if S:
                for _ in range(50):
                    if time.time() - start_time > time_limit:
                        break
                    a = random.choice(list(S))
                    S.remove(a)
                    # Pick two random candidates
                    # To avoid bias, pick from shuffled_elements
                    x = None
                    y = None
                    # We need two distinct elements not in S and not a
                    # Since S is small, we can just filter
                    candidates = [z for z in shuffled_elements if z not in S and z != a]
                    if len(candidates) < 2:
                        S.add(a)
                        continue
                    x, y = random.sample(candidates, 2)
                    
                    # Check if x can be added
                    if is_valid_addition(S, x):
                        S.add(x)
                        # Check if y can be added
                        if is_valid_addition(S, y):
                            S.add(y)
                            improved = True
                            break
                        else:
                            S.remove(x)
                    S.add(a)
                    
        return S

    best_set = set()
    global_start = time.time()
    deadline = global_start + 85.0
    
    # Initial greedy
    S = set()
    # Try a few greedy orders
    for _ in range(5):
        random.shuffle(elements)
        S_greedy = set()
        for x in elements:
            if is_valid_addition(S_greedy, x):
                S_greedy.add(x)
        if len(S_greedy) > len(best_set):
            best_set = S_greedy
            
    # Random restarts
    while time.time() < deadline:
        # Random initial set
        size = random.randint(10, 25)
        S = set(random.sample(elements, size))
        # Clean up
        valid_S = set()
        for x in S:
            if is_valid_addition(valid_S, x):
                valid_S.add(x)
        S = valid_S
        
        # Run local search
        time_limit = 0.5
        new_S = local_search(S, time_limit)
        
        if len(new_S) > len(best_set):
            best_set = new_S
            
    return list(best_set)
