import time
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 1000000
    N_MIN = 10000
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # Function to generate P-smooth numbers up to N
    def generate_smooth(prime_list, limit):
        # Use a set to store unique numbers
        smooths = set()
        # Min-heap for generation
        heap = [1]
        smooths.add(1)
        
        while heap:
            x = heapq.heappop(heap)
            if x > limit:
                break
            for p in prime_list:
                next_x = x * p
                if next_x > limit:
                    break
                if next_x not in smooths:
                    smooths.add(next_x)
                    heapq.heappush(heap, next_x)
                    
        return smooths

    # Try increasing sets of primes
    for k in range(1, len(primes) + 1):
        if time.time() > deadline:
            break
            
        current_primes = primes[:k]
        smooths = generate_smooth(current_primes, N)
        
        # Check for consecutive pairs in range
        # We can iterate through sorted smooths or just check n in range
        # Since smooths is a set, we can check n and n+1
        # But iterating 10000 to 1000000 is fast enough if we break early
        # Better: iterate through smooths, check if n-1 or n+1 is in smooths
        
        found = False
        best_n = -1
        
        # To ensure we find the one with smallest LPF if multiple exist (though logic says they all have LPF = current_primes[-1])
        # We just need any one.
        for n in smooths:
            if n < N_MIN:
                continue
            if n > N:
                break # Not sorted, so can't break easily. But set iteration is arbitrary.
            
            # Check n-1
            if n - 1 >= N_MIN and (n - 1) in smooths:
                # Pair (n-1, n)
                # n-1 is in range [10000, 1000000]
                # n is in range [10001, 1000001]
                # We need n in [10000, 1000000] and n+1 in [10001, 1000001]
                # The task asks for n in [10000, 1000000].
                # So if we find pair (a, a+1), we return a.
                # Here a = n-1.
                if a := n - 1:
                    if N_MIN <= a <= N:
                        return a
                        
            # Check n+1
            if n + 1 <= N and (n + 1) in smooths:
                # Pair (n, n+1)
                if N_MIN <= n <= N:
                    return n
                    
    # Fallback to sieve if no pair found in reasonable primes
    # This is unlikely but safe
    lpf = [0] * (N + 1)
    for i in range(2, N + 1):
        if lpf[i] == 0:
            lpf[i] = i
            for j in range(i, N + 1, i):
                lpf[j] = i
                if time.time() > deadline:
                    return 10000

    best_n = N_MIN
    best_score = float('inf')
    for n in range(N_MIN, N):
        score = max(lpf[n], lpf[n+1])
        if score < best_score:
            best_score = score
            best_n = n
            if best_score == 3:
                break
                
    return best_n
