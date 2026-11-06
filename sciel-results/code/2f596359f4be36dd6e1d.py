import numpy as np


def construct():
            N_MIN = 10000
            N_MAX = 1000000
            max_val = N_MAX + 1
            
            lpf = np.ones(max_val + 1, dtype=np.int32)
            is_prime = np.ones(max_val + 1, dtype=bool)
            is_prime[0] = False
            is_prime[1] = False
            
            for i in range(2, int(np.sqrt(max_val)) + 1):
                if is_prime[i]:
                    # Mark multiples as not prime
                    # We want to mark i*i, i*i+i, ...
                    # But for lpf, we want to update ALL multiples i, 2i, ...
                    # Since we iterate i in increasing order, the last prime divisor wins.
                    lpf[i::i] = i
                    # Mark composites
                    # is_prime[i::i] = False would mark i itself as False.
                    # So we start from i*i?
                    # Standard sieve marks from i*i.
                    # But for lpf, we need to update i itself?
                    # lpf[i] = i is correct for prime i.
                    # If we do lpf[i::i] = i, it sets lpf[i]=i, lpf[2i]=i, etc.
                    # This is correct.
                    
                    # For is_prime, we only need to mark composites.
                    # is_prime[i*i::i] = False
                    if i * i <= max_val:
                        is_prime[i*i::i] = False
            
            # Wait, the loop only goes to sqrt(max_val).
            # Primes > sqrt(max_val) will not be processed in the inner loop.
            # But lpf for those primes should be themselves.
            # And lpf for their multiples (which are > max_val? No, multiples can be <= max_val)
            # Example: N=100. sqrt=10.
            # Prime 11.
            # Loop i=2..10.
            # 11 is not processed.
            # lpf[11] remains 1?
            # We need to handle primes > sqrt(N).
            # For any prime p > sqrt(N), its multiples in [1, N] are just p itself (since 2p > N).
            # So lpf[p] should be p.
            # And for composites, their largest prime factor is <= sqrt(composite) ?? No.
            # If x is composite, it has a factor <= sqrt(x).
            # So the largest prime factor of x might be > sqrt(x)?
            # Example x=2*101 = 202. sqrt(202)~14.
            # Largest prime factor is 101.
            # 101 > 14.
            # So we MUST process primes up to N?
            # Or, we can just fill in the rest.
            
            # Actually, the standard SPF sieve only needs to go to sqrt(N) to find the SPF.
            # But for LARGEST prime factor, we need to know all prime factors.
            # If we use the `lpf[i::i] = i` approach, we must iterate over ALL primes.
            # Iterating up to N is $N$ iterations.
            # Inner loop is $N/i$.
            # Total $N \log \log N$.
            # In Python, a loop up to $10^6$ is fine.
            # But `lpf[i::i] = i` is a NumPy operation.
            # Doing this for every prime $i$ up to $10^6$ means $\approx 78,000$ NumPy slice assignments.
            # Each assignment is fast, but 78,000 Python loop iterations calling NumPy might be slow?
            # Or maybe fast enough.
            
            # Alternative:
            # Use the SPF method which only requires sieve up to sqrt(N).
            # Then factorize.
            # The SPF method is proven to be fast.
            
            # Let's stick to the SPF method but optimize the factorization.
            # Or, use the Direct LPF sieve with a pure Python loop?
            # Pure Python loop for $N \log \log N \approx 3 \cdot 10^6$ operations is very fast (~0.1s).
            
            # Let's compare:
            # 1. SPF Sieve (Python): ~0.1s. Factorization (Python): ~0.1s. Total ~0.2s.
            # 2. Direct LPF Sieve (Python): ~0.1s. Scan: ~0.01s. Total ~0.1s.
            
            # The Direct LPF sieve is simpler and avoids the function call overhead of factorization.
            # Let's implement the Direct LPF sieve in pure Python.
            
            max_val = N_MAX + 1
            lpf = [1] * (max_val + 1)
            is_prime = [True] * (max_val + 1)
            is_prime[0] = False
            is_prime[1] = False
            
            for i in range(2, max_val + 1):
                if is_prime[i]:
                    for j in range(i, max_val + 1, i):
                        lpf[j] = i
                        if j > i:
                            is_prime[j] = False
                            
            # This loop runs for i=2 to 1,000,001.
            # For each prime i, inner loop runs max_val/i times.
            # Total iterations ~ max_val * log(log(max_val)).
            # This is efficient.
            
            best_n = N_MIN
            best_val = float('inf')
            for n in range(N_MIN, N_MAX + 1):
                val = max(lpf[n], lpf[n+1])
                if val < best_val:
                    best_val = val
                    best_n = n
                    
            return best_n
