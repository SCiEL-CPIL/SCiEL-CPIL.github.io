import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_n = 512
    max_size = 48
    
    def is_valid(S):
        """
        Checks if S is a valid set.
        Returns True if no element is the mean of any subset of >= 2 others.
        """
        N = len(S)
        if N == 0:
            return True
            
        # Build DP for S
        # dp[k] is bitset of sums achievable by k elements
        max_sum = sum(S)
        dp = [0] * (N + 1)
        dp[0] = 1
        
        for x in S:
            for k in range(N - 1, -1, -1):
                if dp[k]:
                    dp[k + 1] |= (dp[k] << x)
                    
        # For each x in S, check if x is mean of subset of S \ {x}
        for i, x in enumerate(S):
            # Build DP for S \ {x}
            # We can do this by rebuilding, or by using the fact that 
            # we just need to check specific sums.
            # Rebuilding is O(N^2).
            S_rest = S[:i] + S[i+1:]
            N_rest = N - 1
            if N_rest < 2:
                continue
                
            dp_rest = [0] * (N_rest + 1)
            dp_rest[0] = 1
            
            for y in S_rest:
                for k in range(N_rest - 1, -1, -1):
                    if dp_rest[k]:
                        dp_rest[k + 1] |= (dp_rest[k] << y)
                        
            # Check if x is mean of subset of size k >= 2
            # Sum = k * x
            for k in range(2, N_rest + 1):
                target = k * x
                # Check if target is achievable
                if (dp_rest[k] >> target) & 1:
                    return False
                    
        return True

    def fast_greedy(order):
        """
        Builds a set using fast greedy.
        """
        S = []
        for x in order:
            if len(S) >= max_size:
                break
            if time.time() > deadline - 5.0:
                break
                
            # Fast check: x is not mean of subset of S
            # We use a simplified check here for speed, then exact check later?
            # Or just use exact check if it's fast enough.
            # is_valid(S + [x]) is O(N^3).
            # With N=48, this is ~110k ops.
            # 48 iterations * 110k = 5M ops.
            # This is fast enough.
            
            if is_valid(S + [x]):
                S.append(x)
                
        return S

    # 1. Multi-start Fast Greedy
    best_set = []
    best_size = 0
    
    # Deterministic starts
    random.seed(42)
    
    # Try a few different orders
    orders = [
        list(range(max_n, 0, -1)),
        list(range(1, max_n + 1)),
        list(range(1, max_n + 1, 2)), # Odds
        list(range(2, max_n + 1, 2)), # Evens
    ]
    
    for order in orders:
        S = fast_greedy(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size >= max_size:
                break
                
    # Random starts
    while time.time() < deadline - 10.0 and best_size < max_size:
        order = list(range(1, max_n + 1))
        random.shuffle(order)
        S = fast_greedy(order)
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size >= max_size:
                break

    # 2. Local Search / Improvement
    # If we haven't reached max_size, try to improve.
    # We use a swap-based local search.
    
    S = list(best_set)
    S_set = set(S)
    
    # If we have time, try to add more elements or swap to enable additions
    while time.time() < deadline - 2.0 and len(S) < max_size:
        
        # Try to add a random element
        added = False
        remaining = [x for x in range(1, max_n + 1) if x not in S_set]
        
        if remaining:
            # Try a few random additions
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                x = random.choice(remaining)
                new_S = S + [x]
                if is_valid(new_S):
                    S = new_S
                    S_set.add(x)
                    added = True
                    break
        
        if added:
            continue
            
        # If no addition worked, try a swap
        # Remove one, add another
        # We want to keep size same, but change composition
        if len(S) > 0:
            swap_done = False
            for _ in range(50):
                if time.time() > deadline - 2.0:
                    break
                
                # Pick random element to remove
                idx = random.randint(0, len(S) - 1)
                x_remove = S[idx]
                
                # Pick random element to add
                if not remaining:
                    continue
                x_add = random.choice(remaining)
                
                if x_add == x_remove:
                    continue
                    
                # Construct new set
                new_S = S[:idx] + S[idx+1:] + [x_add]
                
                # Check validity
                if is_valid(new_S):
                    S = new_S
                    S_set.discard(x_remove)
                    S_set.add(x_add)
                    remaining.remove(x_remove) # x_remove is now available
                    remaining.append(x_remove) # Wait, x_remove is removed from S, so it's in remaining
                    # Actually, remaining is [x for x in 1..512 if x not in S_set]
                    # So we should just recalculate or update carefully.
                    # For simplicity, just recalculate remaining if swap happens?
                    # Or just update the set.
                    remaining = [x for x in range(1, max_n + 1) if x not in S_set]
                    swap_done = True
                    break
                    
            if swap_done:
                continue

    return S

