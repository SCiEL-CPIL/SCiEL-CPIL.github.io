import time
import random

def _is_valid_set(S):
    """
    Check if S is a valid set.
    Returns True if valid.
    """
    N = len(S)
    if N < 3:
        return True
        
    S = list(S)
    
    for i in range(N):
        a = S[i]
        others = S[:i] + S[i+1:]
        n_others = len(others)
        
        if n_others < 2:
            continue
            
        max_sum = sum(others)
        
        # DP for subset sums by size
        # dp[k] is an integer where bit s is set if sum s is achievable with k elements
        dp = [0] * (n_others + 1)
        dp[0] = 1  # bit 0 set
        
        for val in others:
            # Update dp in reverse to avoid reusing same element
            for k in range(n_others - 1, -1, -1):
                if dp[k]:
                    dp[k+1] |= dp[k] << val
                    
        # Check if any dp[k] for k >= 2 has a bit set at position k*a
        for k in range(2, n_others + 1):
            if dp[k] == 0:
                continue
            target_sum = k * a
            if target_sum > max_sum:
                break
            if (dp[k] >> target_sum) & 1:
                return False
                
    return True

def construct():
    start_time = time.time()
    deadline = 85.0
    rng = random.Random(42)
    
    best_set = []
    
    # Phase 1: Greedy Construction
    candidates = list(range(1, 513))
    
    def greedy_build(order):
        current = []
        S_set = set()
        for x in order:
            if time.time() - start_time > deadline - 10:
                break
            if len(current) >= 48:
                break
            
            # Fast pair check to prune obvious violations
            violation = False
            # 1. x is mean of a, b in S
            for a in current:
                b = 2 * x - a
                if b in S_set and b != a:
                    violation = True
                    break
            # 2. Some a in S is mean of x, b in S
            if not violation:
                for a in current:
                    b = 2 * a - x
                    if b in S_set and b != a and b != x:
                        violation = True
                        break
            
            if violation:
                continue
                
            # Full check
            if _is_valid_set(current + [x]):
                current.append(x)
                S_set.add(x)
                
        return current

    # Try descending order
    best_set = greedy_build(candidates[::-1])
    
    # Try ascending order
    s_asc = greedy_build(candidates)
    if len(s_asc) > len(best_set):
        best_set = s_asc
        
    # Try random orders
    for _ in range(30):
        if time.time() - start_time > deadline - 15:
            break
        order = candidates[:]
        rng.shuffle(order)
        s_rand = greedy_build(order)
        if len(s_rand) > len(best_set):
            best_set = s_rand

    # Phase 2: Local Search / Random Walk
    current_set = best_set[:]
    current_set_set = set(current_set)
    all_elements = set(range(1, 513))
    
    # We want to maximize size.
    # Strategy:
    # 1. Try to add a random unused element.
    # 2. If that fails, try to swap a random element in current_set with a random unused element.
    #    This keeps size constant but explores the space of valid sets of that size.
    #    Eventually, we might find a configuration that allows an addition.
    
    stall_counter = 0
    max_stall = 500
    
    while time.time() - start_time < deadline - 2:
        
        # Try to add
        added = False
        unused = list(all_elements - current_set_set)
        
        if unused and len(current_set) < 48:
            # Pick a random candidate to add
            # We can bias towards elements that are "far" from the mean of the set?
            # For now, random.
            x = rng.choice(unused)
            new_set = current_set + [x]
            
            if _is_valid_set(new_set):
                current_set = new_set
                current_set_set.add(x)
                added = True
                stall_counter = 0
                
                # Update best
                if len(current_set) > len(best_set):
                    best_set = current_set[:]
                    
                # If we reached 48, we can stop
                if len(best_set) >= 48:
                    break
            
        if not added:
            stall_counter += 1
            
            # If stalled too long, try a more aggressive move or restart?
            # For now, just do swaps.
            
            if current_set and unused:
                # Swap: remove y, add x
                y_idx = rng.randrange(len(current_set))
                y = current_set[y_idx]
                x = rng.choice(unused)
                
                new_set = current_set[:]
                new_set.remove(y)
                new_set.append(x)
                
                if _is_valid_set(new_set):
                    current_set = new_set
                    current_set_set.remove(y)
                    current_set_set.add(x)
                    
                    # Update best? No, size is same.
                    # But this might lead to a larger set later.
                    
        # Optional: If stall_counter is very high, we might be stuck in a local optimum
        # that cannot be escaped by single swaps.
        # We could try 2-swaps, but that's complex.
        # Given the time limit, single swaps should be enough to explore.

    return best_set
