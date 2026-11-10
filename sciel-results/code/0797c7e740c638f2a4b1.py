import time
import itertools
import random

def construct():
       start_time = time.time()
       deadline = start_time + 80.0  # Safety margin for 90s limit
       
       # 1. Precompute primes
       max_sum = 128 + 128
       is_prime = [True] * (max_sum + 1)
       is_prime[0] = is_prime[1] = False
       for i in range(2, int(max_sum**0.5) + 1):
           if is_prime[i]:
               for j in range(i*i, max_sum + 1, i):
                   is_prime[j] = False
                   
       # 2. Iterate over all subsets A of size 3 from [1, 128]
       # To speed up, we can randomize the order or just iterate.
       # Let's iterate all, but break early if we find a valid B.
       
       # Optimization: Precompute valid b's for each pair? No, depends on triplet.
       
       # We can speed up the inner check by using bit operations or lists.
       
       for A in itertools.combinations(range(1, 129), 3):
           if time.time() > deadline:
               break
               
           # Calculate valid b's for this A
           # b is valid if b+a1, b+a2, b+a3 are all prime
           a1, a2, a3 = A
           valid_bs = []
           for b in range(1, 129):
               if is_prime[b+a1] and is_prime[b+a2] and is_prime[b+a3]:
                   valid_bs.append(b)
                   
           if len(valid_bs) < 3:
               continue
               
           # Calculate forbidden differences for A
           # D_A = {a_i - a_k}
           d_a = {
               a1-a2, a1-a3,
               a2-a1, a2-a3,
               a3-a1, a3-a2
           }
           
           # Search for B in valid_bs such that no pair difference is in d_a
           found = False
           for B in itertools.combinations(valid_bs, 3):
               b1, b2, b3 = B
               # Check pairwise differences
               if (b1-b2) in d_a: continue
               if (b1-b3) in d_a: continue
               if (b2-b1) in d_a: continue
               if (b2-b3) in d_a: continue
               if (b3-b1) in d_a: continue
               if (b3-b2) in d_a: continue
               
               # If we pass, all 9 sums are distinct primes
               # Verify just in case
               sums = set()
               for a in A:
                   for b in B:
                       sums.add(a+b)
               if len(sums) == 9:
                   # All are prime by construction of valid_bs
                   # Are they pairwise coprime? Yes, distinct primes are pairwise coprime.
                   return [list(A), list(B)]
                   
       # Fallback if no 9-sum solution found (should not happen, but just in case)
       # Return seed
       return [[1, 7, 13], [4, 10, 16]]
