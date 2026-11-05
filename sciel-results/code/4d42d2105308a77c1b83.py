import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    N = 192
    candidates = list(range(1, N + 1))
    
    best_set = []
    best_size = 0
    
    def can_add(x, S):
        """
        Check if x can be added to set S.
        S is a list of distinct integers.
        Returns True if valid, False otherwise.
        """
        n = len(S)
        
        # 1. Check if x divides the sum of any 3 distinct elements in S.
        if n >= 3:
            # Optimization: If x is greater than the max possible sum of 3 elements in S, 
            # it cannot divide them (since sums are positive).
            sorted_S = sorted(S)
            max_sum_3 = sorted_S[-1] + sorted_S[-2] + sorted_S[-3]
            
            if x <= max_sum_3:
                # We need to check all combinations of 3
                for u, v, w in combinations(sorted_S, 3):
                    if (u + v + w) % x == 0:
                        return False
        
        # 2. Check if any a in S divides x + u + v for distinct u, v in S \ {a}.
        # For each a in S:
        #   We need to check if there exist u, v in S such that u != v, u != a, v != a
        #   and (x + u + v) % a == 0.
        
        for a in S:
            # Optimization: Min possible sum involving x and two others from S\{a}
            # is x + min(S\{a}) + second_min(S\{a}).
            # If this min sum >= a, we must check.
            # Actually, if x+u+v < a, it can't be divisible (unless 0, but x,u,v > 0).
            # Since x,u,v are positive, x+u+v >= 1+1+1 = 3.
            # If a > x + max(S\{a}) + second_max(S\{a}), then x+u+v < a, so no divisibility.
            # Let's use a simpler bound: if a > x + sum of two largest in S\{a}, skip?
            # No, (x+u+v) % a == 0 means x+u+v is a multiple of a.
            # Smallest multiple is a itself.
            # So if x + u + v < a, it's not divisible.
            # Max x+u+v is x + max(S\{a}) + second_max(S\{a}).
            # If max_possible < a, skip.
            # But checking this requires sorting or finding top 2.
            # Given n is small, let's just iterate but add a quick check.
            
            # Quick check: if a is very large compared to x and S, it might be safe.
            # But let's stick to the reliable iteration for correctness first, 
            # optimizing only if needed.
            
            # Iterate over pairs u, v in S \ {a}
            # To avoid O(n^3) in the worst case, we can precompute sums?
            # For now, simple nested loops.
            
            # Optimization: If a is large, (x+u+v) % a == 0 implies x+u+v is a multiple of a.
            # Since x,u,v <= 192, x+u+v <= 576.
            # So a must be <= 576. Which is always true.
            # But if a > x + max(S\{a}) + second_max(S\{a}), then x+u+v < a, so impossible.
            # Let's compute max_sum_for_a = x + sum of two largest in S\{a}.
            # This is O(n) per a, so O(n^2) total for this part.
            
            others = [v for v in S if v != a]
            if len(others) < 2:
                continue
            
            # Find two largest in others
            # Sort others? O(n log n) per a -> O(n^2 log n) total.
            # Or just find max and second max in O(n).
            max1 = -1
            max2 = -1
            for v in others:
                if v > max1:
                    max2 = max1
                    max1 = v
                elif v > max2:
                    max2 = v
            
            max_possible_sum = x + max1 + max2
            if max_possible_sum < a:
                continue
            
            # Now check pairs
            # We can iterate over u in others, and v in others where v != u
            for i in range(len(others)):
                u = others[i]
                for j in range(len(others)):
                    v = others[j]
                    if v == u:
                        continue
                    if (x + u + v) % a == 0:
                        return False
        
        return True

    def run_greedy(order):
        S = []
        for x in order:
            if can_add(x, S):
                S.append(x)
        return S

    # 1. Try descending order (often good for avoiding small divisors)
    current_best = run_greedy(sorted(candidates, reverse=True))
    best_set = current_best
    best_size = len(current_best)
    
    # 2. Try ascending order
    current_best = run_greedy(sorted(candidates))
    if len(current_best) > best_size:
        best_set = current_best
        best_size = len(current_best)

    # 3. Try random orders with various biases
    remaining = deadline - time.time()
    if remaining > 1.0:
        # We will run random restarts until time is up.
        # To improve efficiency, we can try to bias towards larger numbers first,
        # as they are less likely to divide sums of smaller numbers.
        
        # Strategy:
        # - Pure random
        # - Descending with noise
        # - Ascending with noise
        # - Biased: Large numbers first
        
        for _ in range(1000):  # Max iterations, but break on time
            if time.time() > deadline - 5.0:
                break
            
            r = random.random()
            if r < 0.3:
                # Descending
                order = sorted(candidates, reverse=True)
            elif r < 0.6:
                # Ascending
                order = sorted(candidates)
            elif r < 0.8:
                # Random
                order = candidates[:]
                random.shuffle(order)
            else:
                # Biased: Large numbers first, but shuffled within halves
                large = [x for x in candidates if x > N // 2]
                small = [x for x in candidates if x <= N // 2]
                random.shuffle(large)
                random.shuffle(small)
                order = large + small
                
            current_best = run_greedy(order)
            if len(current_best) > best_size:
                best_size = len(current_best)
                best_set = current_best
                
                # If we found a very good set, we might try to improve it with local search?
                # Or just continue.
                
    return best_set
