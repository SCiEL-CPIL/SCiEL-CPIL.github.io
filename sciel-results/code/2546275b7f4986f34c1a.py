import time
import random
from itertools import combinations

N = 192
MAX_SIZE = 48
K = 3

def construct():
    start = time.time()
    deadline = start + 80.0

    # Precompute a fast "safety" helper
    # is_safe_add(x, S, S_set) -> bool
    # 1) No 3 distinct elements in S sum to a multiple of x
    # 2) No a in S divides (x + b + c) for any distinct b,c in S (b!=a, c!=a, b!=c)

    def is_safe_add(x, S, S_set):
        n = len(S)
        # Check 1: x divides sum of any 3 distinct in S
        if n >= 3:
            # Optimization: sum of 3 elements.
            # We can precompute pair sums? Or just iterate.
            # n is small (<=48). Combinations is fine.
            for a, b, c in combinations(S, 3):
                if (a + b + c) % x == 0:
                    return False
        
        # Check 2: a in S divides (x + b + c)
        # Iterate over each a in S.
        # For a given a, we need to know if there exist distinct b, c in S \ {a} such that (x + b + c) % a == 0.
        # Let rem = (-x) % a. We need (b + c) % a == rem.
        # We can precompute the residues of elements in S \ {a} modulo a?
        # Or just iterate b, c.
        if n >= 2:
            for a in S:
                # We need b, c in S \ {a}, b != c.
                # Let's collect residues of (val % a) for val in S \ {a}.
                # We need a pair summing to rem mod a.
                # Since n is small, O(n^2) per a is O(n^3) total.
                # 48^3 = 110k. Fine.
                found = False
                # To avoid checking pairs multiple times, we can just check all pairs.
                # Or, for speed:
                # rem = (-x) % a
                # seen = set()
                # for b in S:
                #   if b == a: continue
                #   r_b = b % a
                #   if (rem - r_b) % a in seen:
                #      # Need to ensure c != b?
                #      # If (rem - r_b) % a == r_b, we need two instances of that residue?
                #      # But elements are distinct. Residues can repeat.
                #      # If we find a residue match, we need to verify the elements are distinct.
                #      # Let's just use the explicit check for correctness and simplicity first.
                #      pass
                # Explicit check is safer.
                for b in S:
                    if b == a: continue
                    for c in S:
                        if c == a or c == b: continue
                        if (x + b + c) % a == 0:
                            found = True
                            break
                    if found: break
                if found:
                    return False
        return True

    def is_valid_full(S):
        S_set = set(S)
        for x in S:
            if not is_safe_add(x, S, S_set):
                return False
        return True

    # Heuristic Score:
    # Lower is better.
    # Small numbers are "bad" because they divide many sums.
    # Large numbers are "good".
    # Score ~ 1/x or just x (ascending preference for large? No, we want to pick large first).
    # Let's use score = N - x. So 192 has score 0, 1 has score 191.
    # We want to pick the candidate with the LOWEST score (highest value) that is safe.
    # Actually, let's try picking from the top down.

    def greedy_from_pool(pool, S, S_set, deadline):
        # pool is a list of candidates to consider, already filtered or we filter here.
        # We sort pool by "preference" (largest first).
        # Then for each candidate in preference order, check if safe.
        # If safe, add it.
        # Wait, the order of addition matters.
        # If we pick 192, then 191, etc.
        
        # Let's sort pool descending.
        pool_sorted = sorted(pool, reverse=True)
        
        for x in pool_sorted:
            if len(S) >= MAX_SIZE:
                break
            if time.time() > deadline:
                break
            if x not in S_set:
                if is_safe_add(x, S, S_set):
                    S.append(x)
                    S_set.add(x)
        return S, S_set

    def local_search_swap(S, S_set, deadline, max_iters=100):
        # Try to remove one element and add two? Or just add one if space?
        # If len(S) < MAX_SIZE, we just add.
        # If len(S) == MAX_SIZE, we can't add. We must swap.
        # But we want to INCREASE size.
        # So if len(S) < MAX_SIZE, we just try to fill it.
        # The greedy should have filled it.
        # If greedy failed to fill, it means no remaining element is safe.
        # To fix this, we might need to remove an element that is "blocking" many others.
        
        # Strategy:
        # 1. If len(S) < MAX_SIZE:
        #    Try adding any remaining safe element. (Greedy should have done this, but maybe order mattered?)
        #    If none safe, try removing an element x from S, and then see if we can add TWO elements?
        #    Or just remove x and try to fill with the best available.
        
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            current_size = len(S)
            
            # If we haven't reached max size, try to add
            if current_size < MAX_SIZE:
                # Try adding from remaining
                remaining = [x for x in range(1, N+1) if x not in S_set]
                # Sort by preference (largest first)
                remaining.sort(reverse=True)
                added_any = False
                for x in remaining:
                    if len(S) >= MAX_SIZE:
                        break
                    if is_safe_add(x, S, S_set):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        added_any = True
                        # Reset loop to check if we can add more?
                        # The while loop will restart.
                        break
                
                if not added_any and current_size < MAX_SIZE:
                    # No single element can be added.
                    # Try removing one element and adding two?
                    # Or removing one and adding one (swap) to potentially unlock?
                    # Let's try removing an element and then greedily refilling.
                    
                    # Pick the element to remove.
                    # Heuristic: Remove the element with the highest "impact" on others?
                    # Or just try each one.
                    # Since N=192, trying each is expensive if refill is slow.
                    # But refill is just greedy.
                    
                    # Let's try removing the SMALLEST element in S (most likely to cause conflicts).
                    # Or the one with the most conflicts?
                    # Let's just try removing the first element in S (arbitrary) and see if we can grow.
                    
                    # Better: Try removing each element in S, and see if the resulting set can be grown larger.
                    # We only care if we can grow.
                    
                    for i in range(len(S)):
                        x_remove = S[i]
                        S_new = S[:i] + S[i+1:]
                        S_new_set = set(S_new)
                        
                        # Refill
                        S_refilled, _ = greedy_from_pool(range(1, N+1), S_new, S_new_set, deadline)
                        
                        if len(S_refilled) > current_size:
                            S = S_refilled
                            S_set = set(S)
                            improved = True
                            break
                            
            else:
                # We are at MAX_SIZE. No improvement possible by size.
                break
                
        return S

    # Strategy 1: Greedy from 192 down to 1
    S1 = []
    S1_set = set()
    S1, S1_set = greedy_from_pool(range(N, 0, -1), S1, S1_set, deadline)
    S1 = local_search_swap(S1, S1_set, deadline)

    best_S = S1

    # Strategy 2: Greedy from 192 down, but skip small numbers initially?
    # The "pool" idea.
    # Try starting with a restricted pool, e.g., 150-192.
    # Then expand.
    
    S2 = []
    S2_set = set()
    # Phase 1: 192 down to 150
    S2, S2_set = greedy_from_pool(range(N, 149, -1), S2, S2_set, deadline)
    # Phase 2: 149 down to 1
    if time.time() < deadline:
        S2, S2_set = greedy_from_pool(range(149, 0, -1), S2, S2_set, deadline)
    S2 = local_search_swap(S2, S2_set, deadline)
    
    if len(S2) > len(best_S):
        best_S = S2

    # Strategy 3: Randomized Greedy
    # Shuffle the order, but bias towards larger numbers.
    # Or just shuffle completely.
    
    if time.time() < deadline - 5:
        for _ in range(50):
            if time.time() > deadline - 2:
                break
            order = list(range(1, N+1))
            random.shuffle(order)
            # Bias: put large numbers at the front?
            # Let's just shuffle.
            
            S3 = []
            S3_set = set()
            S3, S3_set = greedy_from_pool(order, S3, S3_set, deadline)
            S3 = local_search_swap(S3, S3_set, deadline)
            
            if len(S3) > len(best_S):
                best_S = S3

    return best_S
