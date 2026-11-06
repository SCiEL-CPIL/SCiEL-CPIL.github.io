import time
import random
import sys

N = 512

def _can_add(S, x):
    """
    Check if x can be added to the set represented by bytearray S (1-indexed).
    S[i] == 1 if i is in the set.
    """
    # Case 1: x is the 4th term (a, a+d, a+2d, x)
    # x-3d, x-2d, x-d must be in S
    max_d = (x - 1) // 3
    for d in range(1, max_d + 1):
        if S[x - d] and S[x - 2*d] and S[x - 3*d]:
            return False
            
    # Case 2: x is the 3rd term (a, a+d, x, a+3d)
    # x-2d, x-d, x+d must be in S
    max_d = (x - 1) // 2
    for d in range(1, max_d + 1):
        y = x + d
        if y <= N and S[x - 2*d] and S[x - d] and S[y]:
            return False
            
    # Case 3: x is the 2nd term (a, x, a+2d, a+3d)
    # x-d, x+d, x+2d must be in S
    max_d = x - 1
    for d in range(1, max_d + 1):
        y1 = x + d
        y2 = x + 2*d
        if y2 <= N and S[x - d] and S[y1] and S[y2]:
            return False
            
    # Case 4: x is the 1st term (x, x+d, x+2d, x+3d)
    # x+d, x+2d, x+3d must be in S
    max_d = (N - x) // 3
    for d in range(1, max_d + 1):
        if S[x + d] and S[x + 2*d] and S[x + 3*d]:
            return False
            
    return True

def _greedy_greedy(order):
    """
    Construct a 4-AP free set using greedy algorithm with given order.
    """
    S = bytearray(N + 1)
    for x in order:
        if _can_add(S, x):
            S[x] = 1
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Strict safety margin for 60s limit
    
    best_S = bytearray(N + 1)
    best_size = 0
    
    # 1. Deterministic Greedy: Try adding 1, 2, ..., N
    # This is fast and provides a baseline.
    S_det = _greedy_greedy(range(1, N + 1))
    size_det = sum(S_det)
    if size_det > best_size:
        best_size = size_det
        best_S = S_det
        
    # 2. Randomized Greedy: Try a few random orders
    # Each pass is O(N^2). We limit the number of passes to stay within time.
    # Estimate: 1 pass ~ 0.05s. 50 passes ~ 2.5s.
    # We can afford more, but let's be conservative.
    max_passes = 200
    passes = 0
    while time.time() < deadline - 5.0 and passes < max_passes:
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        S_rand = _greedy_greedy(candidates)
        size_rand = sum(S_rand)
        if size_rand > best_size:
            best_size = size_rand
            best_S = S_rand
        passes += 1
        
    # Return sorted list of elements
    return [i for i in range(1, N + 1) if best_S[i]]

if __name__ == "__main__":
    # For testing
    res = construct()
    print(len(res))
