import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    N = 1000000
    
    # Sieve for Euler's totient function
    # phi[i] will store phi(i)
    phi = list(range(N + 1))
    
    for i in range(2, N + 1):
        if phi[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi[j] -= phi[j] // i
                
    # Find the smallest n such that phi(n) is divisible by 120
    for n in range(1, N + 1):
        if time.time() > deadline:
            break
        if phi[n] % 120 == 0:
            return n
            
    # Fallback (should not happen for this problem size)
    return 120 * 120
