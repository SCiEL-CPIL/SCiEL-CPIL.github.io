import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # Target generation limit
    # We generate powerful numbers up to L_gen
    # Sums will be up to 2 * L_gen
    # We want to cover N up to ~ 10^9 potentially
    # Let's aim for L_gen = 10^8, so N up to 2*10^8
    # If time permits, we can try higher
    
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

    # Try L_gen = 10^8 first
    L_gen = 10**8
    
    try:
        P = generate_powerful(L_gen)
        P_arr = np.array(P, dtype=np.int64)
        K = len(P_arr)
        
        if time.time() > deadline:
            raise TimeoutError()

        # If K is small enough, we can try a larger L
        # But 10^8 should give K ~ 5000-6000
        # Pairs ~ 1.5 * 10^7
        # This is fast.
        
        # Calculate upper triangle sums
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
        # a <= b => 2a <= N => a <= N/2
        # Also a must be in P
        # b = N - a must be in P
        
        # Optimize: iterate a in P
        # Since P is sorted, we can break early
        half_N = best_N // 2
        for a in P:
            if a > half_N:
                break
            b = best_N - a
            if b in P_set:
                pairs.append([a, b])
                
        # Verify count
        # The count from unique includes (a,b) where a!=b once, and (a,a) once.
        # My extraction also counts them once.
        # So len(pairs) should match max_count.
        
        return {'N': best_N, 'pairs': pairs}

    except Exception as e:
        # Fallback for safety
        # Use a smaller limit
        try:
            L_gen = 10**6
            P = generate_powerful(L_gen)
            P_set = set(P)
            best_N = 0
            best_pairs = []
            max_c = 0
            # O(K^2) is fine for small K
            for i, a in enumerate(P):
                for b in P[i:]:
                    s = a + b
                    # We need to group by s to find max
                    # This fallback is just to return *something*
                    # Let's just return the first few pairs for a small N
                    pass
            
            # Simple fallback: return N=2, pairs=[[1,1]]
            # Or find max in small range
            counts = {}
            for i, a in enumerate(P):
                for b in P[i:]:
                    s = a + b
                    counts[s] = counts.get(s, 0) + 1
                    if counts[s] > max_c:
                        max_c = counts[s]
                        best_N = s
                        best_pairs = [] # Need to re-collect
            if best_N > 0:
                for a in P:
                    if 2*a > best_N: break
                    b = best_N - a
                    if b in P_set:
                        best_pairs.append([a, b])
            if best_pairs:
                return {'N': best_N, 'pairs': best_pairs}
        except:
            pass
        
        return {'N': 2, 'pairs': [[1, 1]]}
