import time
import random

def construct():
    """
    Construct a 4-term-progression-free subset of [1, 512].
    Uses greedy initialization followed by local search with swaps to maximize cardinality.
    """
    start_time = time.time()
    deadline = 88.0  # Safety margin before 90s kill
    n = 512
    
    def can_add(v, S):
        """
        Check if adding v to S creates a nonconstant 4-AP.
        S is a set of integers.
        Returns True if S union {v} is 4-AP-free.
        """
        # v is in [1, n]
        # Check all d >= 1
        # Case 1: v is the first term (a).
        # v, v+d, v+2d, v+3d. Need v+d, v+2d, v+3d in S.
        # Max d: v + 3d <= n => d <= (n - v) // 3
        max_d_1 = (n - v) // 3
        for d in range(1, max_d_1 + 1):
            if (v + d) in S and (v + 2 * d) in S and (v + 3 * d) in S:
                return False

        # Case 2: v is the second term (a+d).
        # a, v, v+d, v+2d. Need a, v+d, v+2d in S.
        # a = v - d >= 1 => d <= v - 1
        # v + 2d <= n => d <= (n - v) // 2
        max_d_2 = min(v - 1, (n - v) // 2)
        for d in range(1, max_d_2 + 1):
            if (v - d) in S and (v + d) in S and (v + 2 * d) in S:
                return False

        # Case 3: v is the third term (a+2d).
        # a, a+d, v, v+d. Need a, a+d, v+d in S.
        # a = v - 2d >= 1 => d <= (v - 1) // 2
        # v + d <= n => d <= n - v
        max_d_3 = min((v - 1) // 2, n - v)
        for d in range(1, max_d_3 + 1):
            if (v - 2 * d) in S and (v - d) in S and (v + d) in S:
                return False

        # Case 4: v is the fourth term (a+3d).
        # a, a+d, a+2d, v. Need a, a+d, a+2d in S.
        # a = v - 3d >= 1 => d <= (v - 1) // 3
        # a + 2d = v - d <= n (always true for v <= n, d >= 1)
        max_d_4 = (v - 1) // 3
        for d in range(1, max_d_4 + 1):
            if (v - 3 * d) in S and (v - 2 * d) in S and (v - d) in S:
                return False

        return True

    def greedy_init(n, seed):
        """
        Perform a greedy construction using a specific random seed for ordering.
        """
        random.seed(seed)
        S = set()
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for v in candidates:
            if can_add(v, S):
                S.add(v)
        return S

    # 1. Initialize with multiple random greedy runs to find a good starting point
    best_S = set()
    best_size = 0
    
    # Run 5 different greedy initializations
    for i in range(5):
        S = greedy_init(n, i)
        if len(S) > best_size:
            best_size = len(S)
            best_S = S.copy()
            
    # 2. Local Search / Hill Climbing
    # We will alternate between trying to add elements and performing swaps.
    # A swap removes u from S and adds v (if valid), keeping size constant.
    # This helps escape local minima where no single addition is possible.
    
    current_S = best_S
    current_size = best_size
    
    # Pre-generate a list of all numbers for random access
    all_nums = list(range(1, n + 1))
    
    # To speed up "find a v to add", we maintain a list of "available" candidates
    # But since S changes, it's easier to just iterate or pick random.
    # Given n=512, iterating all 512 numbers is cheap.
    
    max_iterations = 20000 # High limit, time is the real constraint
    
    for _ in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
            
        # Try to grow the set
        grew = False
        # Iterate through a random permutation of candidates to avoid bias
        # But generating a permutation every step is O(N log N).
        # Instead, just iterate 1..N and check can_add.
        # If we find one, add it and restart the loop (or continue to find more).
        
        # Efficient growth:
        for v in all_nums:
            if v not in current_S:
                if can_add(v, current_S):
                    current_S.add(v)
                    current_size += 1
                    grew = True
                    # Break to re-evaluate, or continue?
                    # Continuing might allow adding multiple elements in one pass.
                    # Let's continue.
        
        if grew:
            if current_size > best_size:
                best_size = current_size
                best_S = current_S.copy()
            # If we grew, continue to next iteration (might have room for more)
            continue
            
        # If no growth, perform a swap
        # Pick a random element u to remove
        # Pick a random element v to add (v not in S, v != u)
        # We want v to be "close" to being valid, i.e., blocked by u.
        # Simple strategy: Random swap.
        
        if len(current_S) == 0:
            break
            
        u = random.choice(list(current_S))
        current_S.remove(u)
        
        # Try to add a random v from the remaining candidates
        # To improve chances, we can try to add v that conflicts with u?
        # For now, just random v.
        # We need to ensure v is not in S.
        # Since S is large, most v are in S? No, S is ~100-200, N=512.
        # So ~60% of v are in S.
        
        swapped = False
        # Try up to 10 random v's to find one that fits
        for _ in range(10):
            v = random.randint(1, n)
            if v in current_S:
                continue
            if v == u:
                continue
            if can_add(v, current_S):
                current_S.add(v)
                swapped = True
                break
        
        if not swapped:
            # Revert: add u back
            current_S.add(u)
            
        # Occasionally, if we are stuck, we might try a "deletion" to reduce size?
        # No, we want to maximize size.
        
        # Check time and break if necessary
        if time.time() > deadline:
            break

    return best_S

if __name__ == "__main__":
    # For local testing
    S = construct()
    print(f"Size: {len(S)}")
    # Verify
    is_valid = True
    S_set = set(S)
    n = 512
    for a in S:
        for d in range(1, (n - a) // 3 + 1):
            if (a + d) in S_set and (a + 2 * d) in S_set and (a + 3 * d) in S_set:
                is_valid = False
                break
        if not is_valid:
            break
    print(f"Valid: {is_valid}")
