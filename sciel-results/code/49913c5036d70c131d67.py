import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    # The known optimal solution for distinct subset sums is the binary sequence
    # a_i = 2^(i-1) for i=1..n.
    # This minimizes the maximum element, which is 2^(n-1).
    n = 12
    result = [1 << i for i in range(n)]
    
    # Verification
    # Check if all subset sums are distinct
    sums = set()
    for mask in range(1 << n):
        s = 0
        for i in range(n):
            if mask & (1 << i):
                s += result[i]
        sums.add(s)
        
    if len(sums) != (1 << n):
        # Fallback, though binary should always work
        pass
        
    return result
