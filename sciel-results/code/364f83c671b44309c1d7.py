import time
import random
from math import isqrt

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin for 90s limit
    
    N = 256
    candidates = list(range(2, N + 1))
    
    def is_prime(n):
        if n < 2:
            return False
        for i in range(2, isqrt(n) + 1):
            if n % i == 0:
                return False
        return True

    # Initial set: Primes
    S = [p for p in candidates if is_prime(p)]
    
    def is_valid_set(S_list):
        # Check if S_list is valid
        # For all distinct a, b, c in S_list, a does not divide b * c
        n = len(S_list)
        if n < 3:
            return True
        
        # Optimization: precompute or just loop
        # Loop is fine for n < 256
        for i in range(n):
            a = S_list[i]
            for j in range(n):
                if i == j:
                    continue
                b = S_list[j]
                for k in range(n):
                    if i == k or j == k:
                        continue
                    c = S_list[k]
                    if (b * c) % a == 0:
                        return False
        return True

    # Greedy Addition
    # Sort candidates by some heuristic? 
    # Numbers with fewer prime factors might be safer? 
    # Or just iterate.
    remaining = [x for x in candidates if x not in S]
    
    added = True
    while added and time.time() < deadline:
        added = False
        # Shuffle to avoid order bias? Or sort by value?
        # Let's try sorting by number of prime factors (ascending)
        # Def count_factors(x): ...
        # For now, just iterate
        for x in remaining:
            if time.time() > deadline - 5:
                break
            if x in S:
                continue
            new_S = S + [x]
            if is_valid_set(new_S):
                S = new_S
                remaining.remove(x)
                added = True
                break # Restart loop to check others with new S? 
                # Or just continue? Breaking and restarting is safer for greedy.
    
    # Local Search: 1-1 Swaps
    # Try to replace an element in S with one in remaining to potentially allow more additions later?
    # Or just to maintain validity while exploring.
    # Actually, if we just want max size, we want to ADD.
    # If we are stuck, maybe removing a "bad" element allows adding two "good" ones?
    # 1-1 swap doesn't increase size.
    # 1-2 exchange: remove 1, add 2.
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        S_set = set(S)
        # Try 1-2 exchanges
        # For each p in S:
        #   For each x1 in remaining:
        #     For each x2 in remaining:
        #       if x1 != x2:
        #         new_S = (S - {p}) + {x1, x2}
        #         if is_valid_set(new_S):
        #            S = new_S
        #            improved = True
        #            break loops
        
        # This is O(|S| * |R|^2 * |S|^3) which is huge.
        # |S| ~ 60, |R| ~ 200. 60 * 40000 * 60^3 = 5e9. Too slow.
        
        # Heuristic: Only try to swap with elements that "conflict" less?
        # Or just stick to greedy addition which is likely close to optimal for this small N.
        # The prime set is 54. Greedy might add a few composites.
        
        # Let's try a simpler local search:
        # Try to add any element. If none, try to remove one element and see if we can add two.
        
        for i in range(len(S)):
            if time.time() > deadline - 5:
                break
            p = S[i]
            S_minus_p = S[:i] + S[i+1:]
            # Try to find 2 elements from remaining that can be added
            # To speed up, we can just iterate pairs in remaining
            found = False
            R = [x for x in remaining if x not in S_minus_p]
            # Limit R to a subset if too large?
            # Let's just iterate all pairs in R
            for j in range(len(R)):
                if time.time() > deadline - 5:
                    break
                x1 = R[j]
                for k in range(j+1, len(R)):
                    x2 = R[k]
                    new_S = S_minus_p + [x1, x2]
                    if is_valid_set(new_S):
                        S = new_S
                        remaining = [x for x in remaining if x != x1 and x != x2]
                        improved = True
                        found = True
                        break
                if found:
                    break
            if improved:
                break
    
    # Final check and return
    if is_valid_set(S):
        return [int(x) for x in S]
    else:
        # Fallback to primes if something went wrong
        return [int(p) for p in S if is_prime(p)]
