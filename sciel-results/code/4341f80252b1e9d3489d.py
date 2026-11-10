import math
import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0  # Leave 5s margin for safety
    
    N = 128
    nums = list(range(1, N + 1))
    
    # Precompute LCMs
    lcm = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm[i][j] = i * j // math.gcd(i, j)
            lcm[j][i] = lcm[i][j]
            
    def is_conflict(a, b, c):
        return lcm[a][b] == lcm[a][c] == lcm[b][c]
        
    def is_valid_addition(S, x):
        s_list = list(S)
        for i in range(len(s_list)):
            for j in range(i + 1, len(s_list)):
                if is_conflict(s_list[i], s_list[j], x):
                    return False
        return True
        
    def greedy_construct(order):
        S = set()
        for x in order:
            if is_valid_addition(S, x):
                S.add(x)
        return S
        
    def check_validity_with_new(S, new_elements, removed_elements):
        """
        Check if (S - removed_elements) U new_elements is valid.
        S is assumed to be valid.
        Only need to check triples involving at least one element from new_elements.
        """
        base_set = [x for x in S if x not in removed_elements]
        
        # Triples with 3 new elements
        if len(new_elements) >= 3:
            for a, b, c in combinations(new_elements, 3):
                if is_conflict(a, b, c):
                    return False
        
        # Triples with 2 new, 1 base
        if len(new_elements) >= 2:
            for a, b in combinations(new_elements, 2):
                for z in base_set:
                    if is_conflict(a, b, z):
                        return False
        
        # Triples with 1 new, 2 base
        for y in new_elements:
            for i in range(len(base_set)):
                for j in range(i + 1, len(base_set)):
                    if is_conflict(y, base_set[i], base_set[j]):
                        return False
                        
        return True

    def local_search(S, time_limit):
        S = set(S)
        best_S = set(S)
        
        while time.time() < time_limit:
            improved = False
            
            # 1. Try to add any element
            for x in range(1, N + 1):
                if x not in S:
                    if is_valid_addition(S, x):
                        S.add(x)
                        improved = True
                        if len(S) > len(best_S):
                            best_S = set(S)
                        break # Restart add loop
            
            if improved:
                continue
            
            # 2. Try 1-swap (kick)
            s_list = list(S)
            if s_list:
                # Try a few random 1-swaps
                for _ in range(50):
                    if time.time() > time_limit:
                        return best_S
                    x = random.choice(s_list)
                    candidates = [i for i in range(1, N + 1) if i not in S]
                    if not candidates:
                        break
                    y = random.choice(candidates)
                    
                    if check_validity_with_new(S, [y], [x]):
                        S.remove(x)
                        S.add(y)
                        # Update s_list
                        s_list = list(S)
                        improved = True # Consider this an improvement for search flow
                        break
            
            if improved:
                continue
                
            # 3. Try 2-swap
            s_list = list(S)
            if len(s_list) >= 2:
                # Try a few random 2-swaps
                for _ in range(20):
                    if time.time() > time_limit:
                        return best_S
                    x1, x2 = random.sample(s_list, 2)
                    candidates = [i for i in range(1, N + 1) if i not in S]
                    if len(candidates) < 2:
                        break
                    y1, y2 = random.sample(candidates, 2)
                    
                    if check_validity_with_new(S, [y1, y2], [x1, x2]):
                        S.remove(x1)
                        S.remove(x2)
                        S.add(y1)
                        S.add(y2)
                        s_list = list(S)
                        improved = True
                        break
            
            if improved:
                continue
                
            # 4. Stuck, break to allow restart
            break
            
        return best_S

    best_set = set()
    
    # Define several initial orders
    orders = []
    orders.append(list(range(1, N + 1)))
    orders.append(list(range(N, 0, -1)))
    
    # Prime + 1 heuristic
    primes = [i for i in range(2, N + 1) if all(i % j != 0 for j in range(2, int(i**0.5) + 1))]
    prime_plus_1 = [p + 1 for p in primes if p + 1 <= N]
    # Add these to the front
    other_nums = [x for x in range(1, N + 1) if x not in prime_plus_1]
    orders.append(prime_plus_1 + other_nums)
    
    # Random orders
    for _ in range(5):
        random_order = random.sample(range(1, N + 1), N)
        orders.append(random_order)
        
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        if len(S) > len(best_set):
            best_set = set(S)
            
    # Local search on the best greedy result
    if len(best_set) > 0:
        S = local_search(best_set, deadline - 2)
        if len(S) > len(best_set):
            best_set = S
            
    # Additional random restarts with local search if time permits
    while time.time() < deadline - 5:
        random_order = random.sample(range(1, N + 1), N)
        S = greedy_construct(random_order)
        # Short local search
        S = local_search(S, min(time.time() + 1.5, deadline))
        if len(S) > len(best_set):
            best_set = set(S)
            
    # Ensure non-empty
    if not best_set:
        best_set = {1}
        
    return list(best_set)
