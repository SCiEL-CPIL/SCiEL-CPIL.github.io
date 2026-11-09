import time

def construct():
    """
    Constructs a set of 12 distinct positive integers with unique subset sums,
    minimizing the largest integer.
    
    The problem is equivalent to finding a superincreasing sequence of length 12
    that minimizes the last term. The greedy strategy of choosing the smallest
    possible next integer that is greater than the sum of all previous integers
    yields the optimal sequence: 1, 2, 4, ..., 2^(n-1).
    
    For n=12, this results in [1, 2, 4, ..., 2048].
    The largest integer is 2048.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    # Greedy construction of the optimal superincreasing sequence
    # a_i = 1 + sum(a_0 ... a_{i-1})
    # This ensures all subset sums are distinct and minimizes the growth.
    result = []
    current_sum = 0
    
    for i in range(n):
        # Check time just in case (should never trigger for n=12)
        if time.time() > deadline:
            # Fallback to a valid but potentially non-optimal sequence if we ran out of time
            # (This block is unreachable for n=12)
            if not result:
                result = [1]
            while len(result) < n:
                result.append(current_sum + 1)
                current_sum += result[-1]
            return result
        
        next_val = current_sum + 1
        result.append(next_val)
        current_sum += next_val
        
    # The result is [1, 2, 4, ..., 2048]
    # Verify distinct subset sums (optional but good for robustness)
    # For n=12, 2^12 = 4096 sums. This is fast.
    if len(result) == n:
        subset_sums = set()
        for mask in range(1 << n):
            s = 0
            for i in range(n):
                if mask & (1 << i):
                    s += result[i]
            subset_sums.add(s)
        
        if len(subset_sums) != (1 << n):
            # This should never happen for the greedy superincreasing sequence
            # If it does, fall back to a known valid sequence
            return [1 << i for i in range(n)]
            
    return result
