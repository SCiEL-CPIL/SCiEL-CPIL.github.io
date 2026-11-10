import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    n = 192
    max_k = 48
    
    best_set = []
    best_size = 0

    def can_add(x, S):
        """
        Check if x can be added to set S.
        S is a list of distinct integers.
        Condition:
        1. x does not divide sum of any 3 distinct elements in S.
        2. No element a in S divides x + u + v for any distinct u, v in S\{a}.
        """
        n_s = len(S)
        
        # 1. x divides sum of 3 in S?
        if n_s >= 3:
            # Optimize: check sums
            # We can break early
            for c in combinations(S, 3):
                if sum(c) % x == 0:
                    return False
        
        # 2. a in S divides x + u + v?
        if n_s >= 2:
            for i, a in enumerate(S):
                # We need to check pairs (u, v) from S\{a}
                # u != v
                # Note: u and v are distinct elements from S, and neither is a.
                # So we iterate over indices j, k such that j != k, j != i, k != i.
                # And u != v is guaranteed by j != k since elements are distinct.
                
                # Optimization: 
                # (x + u + v) % a == 0
                # u + v ≡ -x (mod a)
                
                # We can precompute residues? 
                # For a fixed a, we want to know if there exist u, v in S\{a} such that u+v ≡ -x (mod a).
                
                # Since n_s is small, direct check is fine.
                for j in range(n_s):
                    if j == i:
                        continue
                    u = S[j]
                    for k in range(j + 1, n_s):
                        if k == i:
                            continue
                        v = S[k]
                        if (x + u + v) % a == 0:
                            return False
        
        return True

    def run_greedy(order_type):
        """
        Run a greedy construction with a specific ordering.
        order_type: 'desc', 'asc', 'random'
        """
        candidates = list(range(1, n + 1))
        
        if order_type == 'desc':
            candidates.sort(reverse=True)
        elif order_type == 'asc':
            candidates.sort()
        elif order_type == 'random':
            random.shuffle(candidates)
        elif order_type == 'random_desc_bias':
            # Weight towards larger numbers
            weights = [c for c in candidates]
            random.shuffle(candidates)
            # Simple bias: sort by random + value?
            # Or just pick from top half more often?
            # Let's stick to simple random or desc/asc for now.
            pass

        current_set = []
        
        for x in candidates:
            if time.time() > deadline:
                break
            if can_add(x, current_set):
                current_set.append(x)
                if len(current_set) >= max_k:
                    break
        
        return current_set

    # Try different strategies
    # 1. Descending is usually best for this type of problem
    # 2. Random restarts to find better local optima
    
    # First, try deterministic Descending
    s = run_greedy('desc')
    if len(s) > best_size:
        best_size = len(s)
        best_set = s[:]

    # Try Ascending (just in case, though unlikely)
    s = run_greedy('asc')
    if len(s) > best_size:
        best_size = len(s)
        best_set = s[:]

    # Random restarts
    # We can afford many restarts because can_add is fast for small sets
    # But as set grows, can_add gets slower.
    # 48 elements -> 48^3 ~ 110k ops. 192 candidates * 110k ~ 20M ops per restart.
    # 20M ops in Python might take ~1-2 seconds.
    # So we can do ~20-30 restarts in 50s.
    
    for i in range(50):
        if time.time() > deadline:
            break
        
        s = run_greedy('random')
        if len(s) > best_size:
            best_size = len(s)
            best_set = s[:]
            
        # Also try a "biased random" where we shuffle but keep large numbers slightly more likely?
        # Or just shuffle.
        
    # Local Search / Improvement
    # Try to add any remaining number
    if time.time() < deadline:
        current_best = best_set[:]
        added_any = True
        while added_any:
            added_any = False
            for x in range(1, n + 1):
                if time.time() > deadline:
                    break
                if x not in current_best:
                    if can_add(x, current_best):
                        current_best.append(x)
                        added_any = True
                        break # restart loop to check others
            if time.time() > deadline:
                break
        
        if len(current_best) > best_size:
            best_size = len(current_best)
            best_set = current_best[:]

    return best_set
