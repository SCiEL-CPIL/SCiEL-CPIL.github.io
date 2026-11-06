import random
import time

def construct():
                # Sieve
                limit = 1000000
                is_prime = [True] * (limit + 1)
                is_prime[0] = is_prime[1] = False
                for i in range(2, int(limit**0.5) + 1):
                    if is_prime[i]:
                        for j in range(i*i, limit+1, i):
                            is_prime[j] = False
                primes = [i for i, flag in enumerate(is_prime) if flag]
                
                def phi(n):
                    if n == 1: return 1
                    result = n
                    temp = n
                    for p in primes:
                        if p * p > temp:
                            break
                        if temp % p == 0:
                            result -= result // p
                            while temp % p == 0:
                                temp //= p
                    if temp > 1:
                        result -= result // temp
                    return result

                best_k = 1
                best_s = 1
                start_time = time.time()
                deadline = start_time + 110 # 10s margin
                
                # Try a few deterministic starts first?
                # e.g., 1, 10^12 - 100, etc.
                candidates = [1, 10**12 - 200, 10**12 - 1000]
                
                for s in candidates:
                    if time.time() > deadline: break
                    seen = set()
                    k = 0
                    current_s = s
                    while current_s + k <= 10**12 and k < 200:
                        val = phi(current_s + k)
                        if val in seen:
                            break
                        seen.add(val)
                        k += 1
                    if k > best_k:
                        best_k = k
                        best_s = current_s
                        
                # Random search
                while time.time() < deadline:
                    s = random.randint(1, 10**12 - 200)
                    seen = set()
                    k = 0
                    current_s = s
                    while current_s + k <= 10**12 and k < 200:
                        val = phi(current_s + k)
                        if val in seen:
                            break
                        seen.add(val)
                        k += 1
                    if k > best_k:
                        best_k = k
                        best_s = current_s
                        # If we found a very long one, we might want to search neighbors?
                        # But random is fine.
                        
                return {'start': best_s, 'length': best_k}
