import time
import random
import numpy as np

def _dp_subset_sums(elements, max_sum):
    """
    Returns a list of bitsets where dp[k] has bit s set if sum s is achievable with exactly k elements.
    """
    n = len(elements)
    dp = [0] * (n + 1)
    dp[0] = 1
    mask = (1 << (max_sum + 1)) - 1
    for x in elements:
        for k in range(n, 0, -1):
            dp[k] |= (dp[k - 1] << x)
        for k in range(1, n + 1):
            dp[k] &= mask
    return dp

def _is_valid_addition(current_set, x):
    """
    Check if adding x to current_set maintains validity.
    Returns True if valid.
    """
    n = len(current_set)
    if n < 2:
        return True

    # 1. Check if x is the mean of any subset of current_set of size >= 2
    # Sum of subset T must be k * x for k >= 2.
    max_target_x = n * x
    if max_target_x > 25000: # Cap for performance, though n=48, x=512 -> 24576
        max_target_x = 25000
    dp_cur = _dp_subset_sums(current_set, max_target_x)
    for k in range(2, n + 1):
        target = k * x
        if target <= max_target_x and (dp_cur[k] >> target) & 1:
            return False

    # 2. Check if any a in current_set becomes the mean of a subset including x
    # Subset T' from current_set \ {a} plus x.
    # Sum(T') + x = k * a  => Sum(T') = k * a - x
    # k >= 2.
    # Note: T' can be empty? No, subset size of OTHER elements must be >= 2.
    # The subset is T' U {x}. Size is |T'| + 1.
    # Condition: |T'| + 1 >= 2 => |T'| >= 1.
    # Wait, the problem says "subset of at least two OTHER elements".
    # If the subset is {x, y1, ..., ym}, the "other elements" relative to 'a' are x, y1...ym.
    # So we need |{x} U T'| >= 2 => |T'| >= 1.
    
    for a in current_set:
        others = [y for y in current_set if y != a]
        m = len(others)
        if m < 1:
            continue
        
        # We need sum(T') = k * a - x for some k >= 2 and T' subset of others with |T'| >= 1.
        # Let t = |T'|. t ranges from 1 to m.
        # Target sum S_t = k * a - x.
        # Max possible sum of T' is sum(others).
        # Min possible sum of T' is min(others).
        
        # Calculate DP for 'others'
        max_sum_others = sum(others)
        if max_sum_others > 25000:
             max_sum_others = 25000
             
        dp_others = _dp_subset_sums(others, max_sum_others)
        
        for t in range(1, m + 1):
            # k can be 2, 3, ...
            # S_t = k * a - x
            # k * a = S_t + x
            # So S_t + x must be divisible by a? No, k is integer.
            # We just iterate k?
            # k * a = S_t + x.
            # Since S_t <= sum(others) <= 25000, and a <= 512, k <= 25000/512 ~ 50.
            # So k is small.
            
            # Alternatively, iterate t and check if any k works.
            # For a fixed t, S_t is a specific value? No, S_t is the sum of a specific subset.
            # We have the bitset dp_others[t].
            # We need to check if there exists a sum s in dp_others[t] such that (s + x) % a == 0 and (s + x) // a >= 2.
            
            # This is a bit slow to iterate all bits.
            # Instead, iterate possible k values.
            # k * a - x = s.
            # s must be achievable with t elements.
            # k ranges from 2 to (sum(others) + x) // a.
            
            max_k = (max_sum_others + x) // a
            if max_k < 2:
                continue
                
            for k in range(2, max_k + 1):
                s_target = k * a - x
                if s_target < 0 or s_target > max_sum_others:
                    continue
                if (dp_others[t] >> s_target) & 1:
                    return False
    return True

def _construct_greedy(order, deadline):
    selected = []
    for x in order:
        if time.time() > deadline:
            break
        if len(selected) >= 48:
            break
        if _is_valid_addition(selected, x):
            selected.append(x)
    return selected

def _local_search_swap(initial_set, deadline):
    """
    Try to increase the size of the set by performing swaps.
    """
    current = initial_set[:]
    best = current[:]
    best_size = len(best)
    
    # Pre-compute a list of candidates to try adding
    all_candidates = list(range(1, 513))
    
    # If we already have 48, we are done
    if best_size >= 48:
        return best
    
    while time.time() < deadline and best_size < 48:
        improved = False
        # Try to add any element
        for x in all_candidates:
            if x in current:
                continue
            if _is_valid_addition(current, x):
                current.append(x)
                best_size += 1
                best = current[:]
                improved = True
                if best_size >= 48:
                    return best
                break
        
        if improved:
            continue
            
        # If no direct addition, try a swap
        # Remove a random element u, try to add a random element v
        # To be efficient, we can try all pairs (u, v) but that's 48*464 ~ 22k checks.
        # Each check is O(N^2 * DP). Might be too slow.
        # Let's sample.
        
        u_idx = random.randint(0, len(current) - 1)
        u = current[u_idx]
        
        # Try to find a v that allows the set to remain valid (size same)
        # And potentially allows future growth.
        # For now, just find a valid swap to shuffle the set.
        # We want to escape local optima where no x can be added.
        
        # Randomly pick v
        v = random.choice([x for x in all_candidates if x not in current])
        
        new_set = current[:u_idx] + current[u_idx+1:]
        if _is_valid_addition(new_set, v):
            new_set.append(v)
            current = new_set
            # After swap, try to add again in next iteration
            # To avoid getting stuck in 2-cycles, we can track visited states (hash of sorted set)
            # But for 90s, simple random walk is often enough if we have many restarts.
            
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Multiple Greedy starts
    for seed in range(100):
        if time.time() > deadline:
            break
        rng = random.Random(seed)
        order = list(range(1, 513))
        rng.shuffle(order)
        selected = _construct_greedy(order, deadline)
        if len(selected) > best_size:
            best_size = len(selected)
            best_set = selected[:]
            if best_size >= 48:
                return best_set
        
        # Strategy 2: Local Search from this greedy result
        if len(selected) >= 40: # Only search if promising
            if time.time() < deadline - 10:
                improved = _local_search_swap(selected, deadline - 5)
                if len(improved) > best_size:
                    best_size = len(improved)
                    best_set = improved[:]
                    if best_size >= 48:
                        return best_set
                        
    # Strategy 3: If still stuck, try specific structured sets
    # e.g. powers of 2, or numbers with large gaps
    # But random search should cover this.
    
    return best_set[:48]
