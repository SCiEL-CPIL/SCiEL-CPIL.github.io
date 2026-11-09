import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s limit to be safe
    
    # Choose L based on time budget
    # Estimate: K ~ sqrt(L). Ops ~ K^2/2.
    # If L=2e8, K~14000, Ops~1e8. Time~20s.
    L = 200_000_000
    
    # Generate powerful numbers up to L
    powerful = set()
    max_y = int(round(L ** (1/3))) + 1
    for y in range(1, max_y + 1):
        y3 = y ** 3
        if y3 > L:
            break
        max_x = int(round((L // y3) ** 0.5)) + 1
        for x in range(1, max_x + 1):
            val = x * x * y3
            if val <= L:
                powerful.add(val)
    
    P = sorted(powerful)
    K = len(P)
    
    # Check time
    if time.time() > deadline - 5:
        # If generation took too long, reduce L and restart?
        # For simplicity, just proceed if we have time for counting.
        pass
    
    # Count sums using NumPy array for memory efficiency
    max_sum = 2 * L
    # Use int32, max count won't exceed 2^31
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    
    # Iterate pairs
    # We need to be careful with the loop speed.
    # Python loop over K^2/2.
    # To speed up, we can use local vars.
    P_list = P
    counts_arr = counts
    K_local = K
    
    # Check if we have enough time for the loop
    # Estimated time: K^2/2 * 50ns? No, Python is slower.
    # Let's just run it and check time periodically.
    
    for i in range(K_local):
        a = P_list[i]
        # b >= a
        for j in range(i, K_local):
            b = P_list[j]
            s = a + b
            counts_arr[s] += 1
            
        # Check time every 1000 iterations of i
        if i % 1000 == 0 and time.time() > deadline:
            break
    
    # Find max
    # np.argmax returns the index of the max value
    # But we need the index corresponding to the sum N.
    # The array index IS the sum N.
    best_N = int(np.argmax(counts_arr))
    best_count = int(counts_arr[best_N])
    
    # If best_count is 0, something went wrong
    if best_count == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # Reconstruct pairs
    P_set = set(P_list)
    pairs = []
    for a in P_list:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': best_N, 'pairs': pairs}
