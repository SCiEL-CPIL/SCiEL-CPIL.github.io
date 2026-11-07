import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    max_val = 256
    max_size = 60
    
    def is_valid(S):
        m = len(S)
        if m == 0:
            return True
        S = sorted(S)
        # Calculate prefix sums
        prefix = [0] * (m + 1)
        for i in range(m):
            prefix[i+1] = prefix[i] + S[i]
        
        # Check condition: Min sum of size k+1 > Max sum of size k
        # Min sum of size k+1 is prefix[k+1]
        # Max sum of size k is prefix[m] - prefix[m-k]
        for k in range(1, m):
            min_sum_k_plus_1 = prefix[k+1]
            max_sum_k = prefix[m] - prefix[m-k]
            if min_sum_k_plus_1 <= max_sum_k:
                return False
        return True

    # Try to find the largest valid set
    # We iterate m from max_size down to 1
    # For each m, we try to construct the smallest possible set of size m
    # Greedy construction:
    # x_1 = 1
    # x_i = smallest integer > x_{i-1} such that the set can be extended to size m
    # This is hard to check online.
    # Instead, we can try to find ANY valid set of size m.
    # Heuristic: Start with the smallest possible numbers and increase them if needed.
    # Or, simply try to find a set of size m by random search or local search?
    # Given the tight constraints, a structured approach is better.
    
    # We know m=31 is possible with consecutive ints starting at 226.
    # Let's try to find if m=32 or larger is possible.
    # We can use a simple greedy that picks the smallest available x_i 
    # and checks if the remaining slots can be filled with numbers <= 256 
    # while maintaining the potential to satisfy the constraints.
    # A simpler check: If we pick x_1..x_i, what is the minimal possible sum of the remaining m-i elements?
    # And what is the maximal possible sum?
    
    # Let's just try to construct the lexicographically smallest set of size m.
    # If we fail, decrease m.
    
    def find_set(m):
        # Try to find a valid set of size m
        # We use a backtracking search with pruning
        # But m can be up to 60, which is too deep for naive backtracking.
        # However, the constraints are very tight.
        # Let's try a greedy approach that works for the "consecutive" case and see if it can be extended.
        
        # Initialize with smallest possible
        S = [1]
        # Actually, let's start with a base and shift.
        # We found that for m=31, a=225 works.
        # Let's try to find the minimal a for each m.
        # a is such that x_i = a+i is valid.
        # We need a+1 > (m-1)^2/4.
        # And a+m <= 256.
        
        # Check if consecutive set works
        min_a = int((m-1)**2 / 4)
        # We need a > min_a - 1, so a >= min_a
        # Actually a+1 > (m-1)^2/4 => a >= floor((m-1)^2/4) if not integer, else floor+1?
        # Let's just iterate a from 0 to 256-m
        for a in range(0, 256 - m + 1):
            S = [a + i + 1 for i in range(m)]
            if is_valid(S):
                return S
        return None

    # Try m from max_size down to 1
    for m in range(max_size, 0, -1):
        if time.time() > deadline:
            break
        S = find_set(m)
        if S is not None:
            return S
            
    # Fallback
    return [1]
