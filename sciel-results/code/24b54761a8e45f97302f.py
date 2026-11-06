import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # Safety margin before 90s kill
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    def is_valid_add(S, x):
        """
        Check if x can be added to S without forming an equal-gcd triple.
        S is a list of elements.
        """
        # Optimization: If S is empty or size 1, always valid
        if len(S) < 2:
            return True
            
        for i in range(len(S)):
            a = S[i]
            g_ax = gcd_table[a][x]
            for j in range(i+1, len(S)):
                b = S[j]
                g_bx = gcd_table[b][x]
                g_ab = gcd_table[a][b]
                if g_ax == g_bx == g_ab:
                    return False
        return True

    def greedy_construct(order):
        S = []
        for x in order:
            if is_valid_add(S, x):
                S.append(x)
        return S

    def local_search_add(S):
        """
        Try to add any remaining elements to S.
        Returns the modified list.
        """
        S = list(S)
        S_set = set(S)
        improved = True
        # We loop until no more additions can be made
        while improved and time.time() < deadline:
            improved = False
            # Try to add any element from 2..128 not in S
            # Iterate in random order to avoid bias?
            # Or just 2..128.
            for x in range(2, N+1):
                if x not in S_set:
                    if is_valid_add(S, x):
                        S.append(x)
                        S_set.add(x)
                        improved = True
                        # Continue checking other x's, as adding one might allow others?
                        # No, adding an element only adds constraints. 
                        # So if x was invalid before, it remains invalid.
                        # If x was valid, we add it.
                        # So we can just do one pass?
                        # Wait. If A and B are both valid individually, adding A might make B invalid.
                        # But we are iterating. If we add A, then when we check B, we use the updated S.
                        # So a single pass in a fixed order might miss B if B becomes invalid after A is added, 
                        # but B would have been valid if we checked B first.
                        # To maximize, we should try all candidates, and if any can be added, add them.
                        # But which one to pick first?
                        # Greedy: Pick the first one found?
                        # Better: Collect all valid candidates, then add them?
                        # No, adding one invalidates others.
                        # So we must add one, then re-evaluate.
                        # The loop above does this: it adds one, sets improved=True, and restarts the inner loop?
                        # No, the inner loop continues.
                        # If we add x, S changes. The subsequent checks for y use the new S.
                        # This is a valid greedy addition strategy.
        return S

    def count_divisors(x):
        return sum(1 for i in range(1, int(x**0.5)+1) if x % i == 0)

    best_size = 0
    best_set = []
    
    # Candidates: 2 to 128
    items_base = list(range(2, N+1))
    
    # Precompute divisor counts for sorting strategies
    div_counts = {x: count_divisors(x) for x in items_base}
    
    def get_order(strategy, rng):
        items = list(items_base)
        if strategy == 0:
            rng.shuffle(items)
        elif strategy == 1:
            # Primes (low divisors) first
            items.sort(key=lambda x: div_counts[x])
            rng.shuffle(items) # Add noise to break ties? No, sort is stable.
            # Actually, if we sort, we lose the random aspect of "which prime".
            # Let's group by divisor count and shuffle within groups?
            # Simpler: Just sort.
        elif strategy == 2:
            # High divisor count first
            items.sort(key=lambda x: -div_counts[x])
        elif strategy == 3:
            # Evens first
            evens = [x for x in items if x % 2 == 0]
            odds = [x for x in items if x % 2 != 0]
            rng.shuffle(evens)
            rng.shuffle(odds)
            items = evens + odds
        elif strategy == 4:
            # Multiples of 3 first?
            mult3 = [x for x in items if x % 3 == 0]
            rest = [x for x in items if x % 3 != 0]
            rng.shuffle(mult3)
            rng.shuffle(rest)
            items = mult3 + rest
        return items

    rng = random.Random(123)
    
    while time.time() < deadline:
        strategy = rng.randint(0, 4)
        order = get_order(strategy, rng)
        S = greedy_construct(order)
        S = local_search_add(S)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = list(S)
            
        # If we have a very good set, try to improve with swaps
        # Only if we have significant time left
        if time.time() < deadline - 15.0 and len(S) < 120:
            S_current = list(S)
            S_set = set(S_current)
            candidates = [x for x in items_base if x not in S_set]
            
            # Try a few random 1-for-1 swaps
            # We limit the number of swaps to avoid getting stuck
            for _ in range(10):
                if time.time() > deadline - 5.0:
                    break
                if not candidates:
                    break
                u_idx = rng.randint(0, len(S_current)-1)
                u = S_current[u_idx]
                v = rng.choice(candidates)
                
                # Check validity of S_current - {u} + {v}
                S_temp = S_current[:u_idx] + S_current[u_idx+1:]
                if is_valid_add(S_temp, v):
                    # Valid swap. Apply it.
                    S_new = S_temp + [v]
                    # Try to add more
                    S_new = local_search_add(S_new)
                    
                    if len(S_new) > len(S_current):
                        S_current = S_new
                        S_set = set(S_current)
                        candidates = [x for x in items_base if x not in S_set]
                        if len(S_current) > best_size:
                            best_size = len(S_current)
                            best_set = list(S_current)
                    else:
                        # Revert swap if it didn't lead to improvement
                        pass

    # Final check for 1
    valid_with_1 = True
    if len(best_set) > 0:
        for i in range(len(best_set)):
            a = best_set[i]
            for j in range(i+1, len(best_set)):
                b = best_set[j]
                if gcd_table[a][b] == 1:
                    valid_with_1 = False
                    break
            if not valid_with_1:
                break
    
    if valid_with_1:
        best_set.append(1)

    return best_set
