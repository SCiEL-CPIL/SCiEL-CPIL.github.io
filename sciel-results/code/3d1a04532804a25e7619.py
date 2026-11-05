import time
import random
import sys

# Set recursion limit just in case, though not strictly needed for iterative DP
sys.setrecursionlimit(10000)

def creates_3ap(S_set, x):
    """Check if adding x to S_set creates a 3-term arithmetic progression."""
    # S_set is a set of integers for O(1) lookup
    for y in S_set:
        # Case 1: x is the middle term. y, x, z.
        # y + z = 2x => z = 2x - y
        z = 2 * x - y
        if z in S_set and z != y and z != x:
            return True
        
        # Case 2: x is an end term.
        # If x is the larger end: y, z, x. 2z = y + x => z = (y + x) / 2
        if (y + x) % 2 == 0:
            z = (y + x) // 2
            if z in S_set and z != y and z != x:
                return True
        
        # If x is the smaller end: x, y, z. 2y = x + z => z = 2y - x
        z = 2 * y - x
        if z in S_set and z != y and z != x:
            return True
            
    return False

def construct_3ap_free(order):
    """Construct a maximal 3-AP-free set using the given order of candidates."""
    S = []
    S_set = set()
    for x in order:
        if not creates_3ap(S_set, x):
            S.append(x)
            S_set.add(x)
    return S

def is_valid(S):
    """
    Check if S is a valid set: no element is the mean of any subset of at least 2 other elements.
    Uses bitset DP for subset sums by size.
    """
    M = len(S)
    if M < 3:
        return True
        
    for a in S:
        others = [x for x in S if x != a]
        # dp[k] is a bitset where bit j is 1 if there is a subset of others of size k with sum j
        # Max sum is sum(others)
        # We only need to check up to k=M-1
        dp = [0] * (M)
        dp[0] = 1
        current_size = 0
        
        for val in others:
            # Update dp in reverse order to avoid using 'val' more than once
            # We iterate k from current_size down to 0
            # dp[k+1] |= dp[k] << val
            # Note: We must ensure we don't exceed M-1
            for k in range(current_size, -1, -1):
                dp[k+1] |= (dp[k] << val)
            current_size += 1
            
        # Check if any k*a is a valid sum for subset size k (k >= 2)
        valid_a = True
        for k in range(2, M):
            target = k * a
            # Check if bit 'target' is set in dp[k]
            if (dp[k] >> target) & 1:
                valid_a = False
                break
        if not valid_a:
            return False
            
    return True

def fill_set(S, pool):
    """
    Try to add elements from pool to S while maintaining validity.
    S is a list, pool is a list of candidates.
    Returns the new list S.
    """
    S_set = set(S)
    # We iterate through pool. To make it deterministic per call but varied across calls,
    # we assume pool is already shuffled.
    for x in pool:
        if x in S_set:
            continue
        # Check if adding x is valid
        # We can do a fast check first? 
        # is_valid is fast enough for M~40.
        candidate_S = S + [x]
        if is_valid(candidate_S):
            S = candidate_S
            S_set.add(x)
    return S

def local_search_polish(S, pool, time_limit):
    """
    Try to improve S by swapping elements or adding more.
    """
    S_set = set(S)
    pool_set = set(pool)
    improved = True
    start_time = time.time()
    
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try adding any element from pool
        # Shuffle pool for randomization
        random.shuffle(pool)
        for x in pool:
            if x in S_set:
                continue
            candidate_S = S + [x]
            if is_valid(candidate_S):
                S = candidate_S
                S_set.add(x)
                pool.remove(x)
                pool_set.remove(x)
                improved = True
                break # Restart loop with new S
        
        if improved:
            continue
            
        # If no additions, try swaps: remove a, add b
        # Only if we expect size to stay same or increase? 
        # Since we want max size, and we couldn't add, size is stuck.
        # Swaps might allow future additions? 
        # Given time constraints, simple addition greedy is often strong.
        # Let's stick to addition only for simplicity and speed, 
        # unless we have significant time left.
        
        if time.time() - start_time > time_limit * 0.8:
            break
            
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    best_set = []
    
    # Initial deterministic attempt
    order = list(range(1, 513))
    S0 = construct_3ap_free(order)
    pool = [x for x in order if x not in S0]
    S1 = fill_set(S0, pool)
    best_set = S1
    
    # Randomized restarts
    while time.time() < deadline:
        # Generate random order
        order = list(range(1, 513))
        random.shuffle(order)
        
        S0 = construct_3ap_free(order)
        pool = [x for x in order if x not in S0]
        S1 = fill_set(S0, pool)
        
        if len(S1) > len(best_set):
            best_set = S1
            # If we found a very large set, try to polish it?
            # But fill_set already does greedy addition.
            # Local search might help if greedy got stuck.
            # Let's check if we have time for a quick polish on the best set.
            if time.time() < deadline - 10.0:
                remaining_pool = [x for x in range(1, 513) if x not in best_set]
                # Polish with a short time limit
                polished = local_search_polish(best_set, remaining_pool, 2.0)
                if len(polished) > len(best_set):
                    best_set = polished

    return best_set

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    print(f"Size: {len(res)}")
    # Verify
    if is_valid(res):
        print("Valid")
    else:
        print("INVALID")
