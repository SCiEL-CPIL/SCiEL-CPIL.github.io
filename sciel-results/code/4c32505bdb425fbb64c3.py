import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin
    
    N = 192
    candidates = list(range(1, N + 1))
    
    best_set = []
    
    def can_add(x, S):
        """
        Check if x can be added to set S.
        S is a list of distinct integers.
        Returns True if valid, False otherwise.
        """
        n = len(S)
        
        # If S has fewer than 3 elements, no 3 distinct others exist for x to divide.
        # But we still need to check if elements in S divide sums involving x.
        
        # 1. Check if x divides the sum of any 3 distinct elements in S.
        if n >= 3:
            # Optimization: If x is greater than the max possible sum of 3 elements in S, 
            # it cannot divide them.
            # Max sum is sum of 3 largest in S.
            # Sorting S every time is O(n log n). We can maintain sorted S?
            # Or just compute max sum.
            # For n=48, sorting is fast.
            sorted_S = sorted(S)
            max_sum_3 = sorted_S[-1] + sorted_S[-2] + sorted_S[-3]
            
            if x <= max_sum_3:
                # We need to check all combinations of 3
                # Optimization: If x is large, only sums close to multiples of x matter.
                # But simple check is O(n^3).
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
            
            # To be efficient, we can just iterate.
            # n is small.
            
            # We can pre-filter u,v?
            # Let's just use nested loops.
            for i in range(n):
                u = S[i]
                if u == a:
                    continue
                for j in range(n):
                    v = S[j]
                    if v == a or v == u:
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

    # 1. Try descending order
    current_best = run_greedy(sorted(candidates, reverse=True))
    best_set = current_best
    
    # 2. Try ascending order
    current_best = run_greedy(sorted(candidates))
    if len(current_best) > len(best_set):
        best_set = current_best

    # 3. Try random orders
    # Budget remaining
    remaining = deadline - time.time()
    if remaining > 1.0:
        # Estimate time per run.
        # A single run takes ~0.1s - 0.5s depending on set size.
        # Let's try to run as many as possible.
        
        # Use a faster can_add?
        # The current can_add is O(n^3).
        
        # Let's run random restarts.
        # We'll do 100 restarts if time permits, or stop if deadline approached.
        
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            random.shuffle(candidates)
            # It might be better to bias towards large numbers?
            # Let's try pure random first.
            current_best = run_greedy(candidates)
            if len(current_best) > len(best_set):
                best_set = current_best
                # If we found a new best, we might want to try to improve it further?
                # Or just continue random searches.
                
    # 4. Local Search / Improvement?
    # If we have a set S, can we swap an element to allow adding more?
    # This is complex. Given the time limit, random restarts are a solid bet.
    
    return best_set
