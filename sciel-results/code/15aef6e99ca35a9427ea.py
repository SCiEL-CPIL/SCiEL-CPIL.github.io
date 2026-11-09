import random
from math import gcd
from itertools import combinations

def construct():
    N = 128
    numbers = list(range(1, N + 1))
    
    def is_valid_triple(a, b, c):
        g_ab = gcd(a, b)
        g_bc = gcd(b, c)
        g_ca = gcd(c, a)
        return not (g_ab == g_bc == g_ca)

    def can_add(x, S):
        # Check if adding x to S creates a forbidden triple
        # We need to check pairs (y, z) in S
        # To be efficient, we can just check all pairs if S is small, 
        # but S can be ~80. 80^2/2 = 3200 checks. Fast.
        for y in S:
            for z in S:
                if y == z:
                    continue
                if not is_valid_triple(x, y, z):
                    return False
        return True

    def greedy_solve(seed_val):
        random.seed(seed_val)
        S = set()
        candidates = numbers.copy()
        random.shuffle(candidates)
        
        for x in candidates:
            if can_add(x, S):
                S.add(x)
        return S

    def local_search(S, max_iters=100):
        S = set(S)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # 1. Try to add any element
            added = False
            for x in numbers:
                if x not in S:
                    if can_add(x, S):
                        S.add(x)
                        added = True
                        improved = True
                        break # Restart loop to re-evaluate options
            
            if added:
                continue
                
            # 2. Try to swap one element to allow adding another?
            # Or just swap to a "better" position? 
            # Since we only care about size, a swap that keeps size same is only useful 
            # if it allows future additions. 
            # Heuristic: Try to remove x, add y, then try to add z.
            # This is expensive. 
            # Let's try simple swaps first to break local minima?
            # Actually, if we can't add any, we are at a local maximum for 1-add.
            # We might need to remove 1 to add 2.
            
            # Try: Remove x, Add y.
            # If valid, check if we can add z to the new set.
            # If yes, accept (size increases by 1).
            
            # To speed up, we can sample.
            # But N=128 is small.
            
            # Let's try all x in S, y not in S.
            # This is O(|S| * (N-|S|)) ~ 60*60 = 3600 pairs.
            # For each pair, check validity of S\{x} U {y}.
            # Then try to add any z.
            
            for x in list(S):
                for y in numbers:
                    if y in S:
                        continue
                    S_new = S - {x}
                    S_new.add(y)
                    if can_add(y, S - {x}): # y is valid in S\{x}
                        # Check if we can add z
                        for z in numbers:
                            if z in S_new:
                                continue
                            if can_add(z, S_new):
                                S = S_new
                                S.add(z)
                                improved = True
                                break
                        if improved:
                            break
                if improved:
                    break
                    
        return S

    best_set = set()
    best_size = 0
    
    # Run multiple greedy + local search
    for i in range(50):
        S = greedy_solve(i)
        S = local_search(S)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
    # Final check and return
    return sorted(list(best_set))
