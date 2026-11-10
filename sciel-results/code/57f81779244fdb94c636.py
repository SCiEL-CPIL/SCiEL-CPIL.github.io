import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 128
    numbers = list(range(1, N + 1))
    
    # Precompute gcds
    gcds = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcds[i][j] = gcd(i, j)
            gcds[j][i] = gcds[i][j]

    def is_bad_triple(a, b, c):
        return gcds[a][b] == gcds[a][c] == gcds[b][c]

    def is_valid_set(S):
        S_list = list(S)
        for a, b, c in combinations(S_list, 3):
            if is_bad_triple(a, b, c):
                return False
        return True

    def can_add(x, S):
        S_list = list(S)
        for a, b in combinations(S_list, 2):
            if is_bad_triple(x, a, b):
                return False
        return True

    def greedy_construct(order):
        S = set()
        for x in order:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S):
        S = set(S)
        improved = True
        while improved and (time.time() < deadline):
            improved = False
            # 1. Try to add
            for x in range(1, N + 1):
                if x not in S and can_add(x, S):
                    S.add(x)
                    improved = True
                    break
            if improved:
                continue
            
            # 2. Try to swap (keep size, change composition)
            # This might help escape local minima
            swapped = False
            for x_out in list(S):
                for x_in in range(1, N + 1):
                    if x_in in S:
                        continue
                    new_S = S - {x_out}
                    if can_add(x_in, new_S):
                        # Check if this swap is "good". 
                        # Since we just want max size, any valid swap is neutral.
                        # But it might allow future additions.
                        # We accept it if it leads to a state where we can add more?
                        # Simpler: just do it if it's valid.
                        # To avoid oscillation, we can track visited or just limit swaps.
                        S = new_S | {x_in}
                        improved = True
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                continue
                
            # 3. Try 1-remove 2-add (increase size by 1)
            for x_out in list(S):
                S_minus = S - {x_out}
                # Find two elements to add
                candidates = [x for x in range(1, N + 1) if x not in S_minus and can_add(x, S_minus)]
                # We need two from candidates that are compatible with each other and S_minus
                # Actually, can_add(x, S_minus) checks against all pairs in S_minus.
                # We also need x1, x2 to not form a bad triple with each other and any in S_minus?
                # can_add(x1, S_minus) ensures x1 is ok with S_minus.
                # can_add(x2, S_minus) ensures x2 is ok with S_minus.
                # We also need (x1, x2, y) for any y in S_minus to be valid?
                # Wait, can_add(x, S) checks pairs in S.
                # If we add x1, then S becomes S_minus U {x1}.
                # Then we add x2 to S_minus U {x1}.
                # So we just need to check if x2 can be added to S_minus U {x1}.
                
                for x1 in candidates:
                    S_plus1 = S_minus | {x1}
                    for x2 in candidates:
                        if x2 == x1:
                            continue
                        if can_add(x2, S_plus1):
                            S = S_plus1 | {x2}
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break

        return S

    best_S = set()
    best_size = 0
    
    # Multi-start
    num_starts = 50
    for i in range(num_starts):
        if time.time() > deadline:
            break
            
        # Different strategies for ordering
        if i % 5 == 0:
            # Random
            order = numbers[:]
            random.shuffle(order)
        elif i % 5 == 1:
            # Descending
            order = numbers[:]
            order.sort(reverse=True)
        elif i % 5 == 2:
            # Ascending
            order = numbers[:]
        elif i % 5 == 3:
            # Prime rich? (Heuristic: numbers with many small factors might be "denser" in gcd space?)
            # Let's try sorting by number of divisors
            def num_divs(n):
                count = 0
                for d in range(1, int(n**0.5)+1):
                    if n % d == 0:
                        count += 1
                        if d != n//d:
                            count += 1
                return count
            order = numbers[:]
            order.sort(key=num_divs, reverse=True)
        else:
            # Random with bias
            order = numbers[:]
            random.shuffle(order)
            
        S = greedy_construct(order)
        S = local_search(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S

    return list(best_S)
