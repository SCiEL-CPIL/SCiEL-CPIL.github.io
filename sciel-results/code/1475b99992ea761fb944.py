import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin out of 60s
    
    N = 123457
    
    # Sieve of Eratosthenes to get primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(N)) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Boolean array for smooth numbers: smooth[n] = True if n is y-smooth for current y
    smooth = [False] * (N + 1)
    smooth[1] = True
    
    # Track the best solution found so far
    best_pair = None
    best_score = float('inf')
    
    for p in primes:
        if time.time() > deadline:
            # Return best found or fallback
            if best_pair is not None:
                return best_pair
            return [1, N - 1]
        
        # Generate new smooth numbers involving p
        # We want to add numbers of form x * p^k where x in smooth and x * p^k <= N
        new_smooth = [False] * (N + 1)
        for x in range(1, N + 1):
            if smooth[x]:
                val = x
                while val <= N:
                    new_smooth[val] = True
                    if val > N // p:
                        break
                    val *= p
        
        # Merge into smooth
        for i in range(1, N + 1):
            if new_smooth[i]:
                smooth[i] = True
        
        # Check if any a in smooth has N-a in smooth
        # Optimization: only check a <= N/2 to avoid duplicate pairs
        for a in range(1, N // 2 + 1):
            if smooth[a]:
                b = N - a
                if b > 0 and smooth[b]:
                    # Found a valid pair
                    # Score is max(P(a), P(b)), but since both are p-smooth, P(a),P(b) <= p
                    # So the score is at most p
                    current_score = p
                    if current_score < best_score:
                        best_score = current_score
                        best_pair = [a, b]
                    return best_pair
        
        # If no pair found with this prime, continue
    
    # If loop finishes without finding a pair (should not happen for N=123457)
    if best_pair is not None:
        return best_pair
    return [1, N - 1]
