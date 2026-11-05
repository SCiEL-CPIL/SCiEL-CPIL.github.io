import time
import numpy as np

def construct():
          start_time = time.time()
          LIMIT = 90
          MARGIN = 5
          
          N_MAX = 1000001
          START = 10000
          
          # 1. Sieve for Primes
          is_prime = np.ones(N_MAX + 1, dtype=bool)
          is_prime[:2] = False
          for i in range(2, int(N_MAX**0.5) + 1):
              if is_prime[i]:
                  is_prime[i*i::i] = False
          
          primes = [i for i in range(2, N_MAX + 1) if is_prime[i]]
          
          # 2. Compute Powerful Part P
          P = [1] * (N_MAX + 1)
          
          for p in primes:
              p2 = p * p
              if p2 > N_MAX:
                  break
              
              # Limit for k such that k * p2 <= N_MAX
              limit = N_MAX // p2
              
              # W[k] = p^v_p(k)
              # We only need W for k in 1..limit
              W = [1] * (limit + 1)
              
              # Compute W
              # W[k] = p * W[k//p] if k%p==0 else 1
              for k in range(1, limit + 1):
                  if k % p == 0:
                      W[k] = W[k // p] * p
                  else:
                      W[k] = 1
                      
              # Update P
              # m = k * p2
              # P[m] *= p2 * W[k]
              for k in range(1, limit + 1):
                  m = k * p2
                  P[m] *= p2 * W[k]
                  
          if time.time() - start_time > LIMIT - MARGIN:
              return 10000
              
          # 3. Find max product
          P_arr = np.array(P, dtype=np.int64)
          left = P_arr[START:1000001]
          right = P_arr[START+1:1000002]
          products = left * right
          max_idx = np.argmax(products)
          best_n = int(START + max_idx)
          
          return best_n
