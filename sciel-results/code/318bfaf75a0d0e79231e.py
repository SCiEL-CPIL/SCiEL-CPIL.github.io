import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def generate_powerful(L):
        pows = set()
        # t must be squarefree
        # n = s^2 * t^3 <= L
        # t <= L^(1/3)
        max_t = int(L ** (1.0/3.0)) + 2
        
        # Primes up to sqrt(max_t) for squarefree check
        limit_p = int(max_t ** 0.5) + 1
        is_prime = [True] * limit_p
        primes = []
        for i in range(2, limit_p):
            if is_prime[i]:
                primes.append(i)
                for j in range(i*i, limit_p, i):
                    is_prime[j] = False
        
        def is_squarefree(n):
            for p in primes:
                if p * p > n:
                    break
                if n % (p * p) == 0:
                    return False
            return True

        for t in range(1, max_t + 1):
            if not is_squarefree(t):
                continue
            t3 = t ** 3
            if t3 > L:
                continue
            max_s = int((L // t3) ** 0.5)
            for s in range(1, max_s + 1):
                pows.add(s * s * t3)
        return sorted(pows)

    # We want to maximize the number of pairs.
    # The number of powerful numbers K grows as L^(1/3).
    # The number of pairs is ~ K^2.
    # The memory required for the sum array is ~ K^2 * 8 bytes.
    # For L=10^11, K ~ 50,000. Pairs ~ 1.25 * 10^9. Memory ~ 10 GB.
    # This might be too much for some environments.
    # Let's start with L=10^10 (K ~ 20,000, Pairs ~ 2*10^8, Memory ~ 1.6 GB).
    # If time permits, we can try larger L.
    
    # Try L = 10^10 first
    L_gen = 10**10
    
    try:
        P = generate_powerful(L_gen)
        P_arr = np.array(P, dtype=np.int64)
        K = len(P_arr)
        
        if time.time() > deadline:
            raise TimeoutError()

        # Calculate upper triangle sums
        # If K is large, this might be slow/memory heavy
        # K=20,000 -> 200M elements.
        iu = np.triu_indices(K, k=0)
        sums = P_arr[iu[0]] + P_arr[iu[1]]
        
        if time.time() > deadline:
            raise TimeoutError()

        # Find most frequent sum
        unique_sums, counts = np.unique(sums, return_counts=True)
        max_idx = np.argmax(counts)
        best_N = int(unique_sums[max_idx])
        max_count = int(counts[max_idx])
        
        if time.time() > deadline:
            raise TimeoutError()

        # Extract pairs
        P_set = set(P)
        pairs = []
        half_N = best_N // 2
        for a in P:
            if a > half_N:
                break
            b = best_N - a
            if b in P_set:
                pairs.append([a, b])
                
        return {'N': best_N, 'pairs': pairs}

    except Exception as e:
        # Fallback to smaller L if memory or time fails
        try:
            L_gen = 10**8
            P = generate_powerful(L_gen)
            P_arr = np.array(P, dtype=np.int64)
            K = len(P_arr)
            iu = np.triu_indices(K, k=0)
            sums = P_arr[iu[0]] + P_arr[iu[1]]
            unique_sums, counts = np.unique(sums, return_counts=True)
            max_idx = np.argmax(counts)
            best_N = int(unique_sums[max_idx])
            
            P_set = set(P)
            pairs = []
            half_N = best_N // 2
            for a in P:
                if a > half_N:
                    break
                b = best_N - a
                if b in P_set:
                    pairs.append([a, b])
            
            return {'N': best_N, 'pairs': pairs}
        except:
            return {'N': 2, 'pairs': [[1, 1]]}
