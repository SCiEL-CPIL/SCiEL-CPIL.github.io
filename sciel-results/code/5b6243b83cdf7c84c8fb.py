   import time
   import random
   import math

   MAX_VAL = 10000
   MAX_SUM = 20000

   # Precompute SPF
   spf = list(range(MAX_SUM + 1))
   for i in range(2, int(MAX_SUM**0.5) + 1):
       if spf[i] == i:
           for j in range(i*i, MAX_SUM + 1, i):
               if spf[j] == j:
                   spf[j] = i

   # Precompute prime factors for all sums
   prime_factors = [[] for _ in range(MAX_SUM + 1)]
   for n in range(2, MAX_SUM + 1):
       x = n
       factors = set()
       while x > 1:
           p = spf[x]
           factors.add(p)
           while x % p == 0:
               x //= p
       prime_factors[n] = list(factors)

   def construct():
       start_time = time.time()
       deadline = start_time + 85.0
       
       # Helper to get distinct prime factors of sum
       # We use the precomputed list
       
       def get_factors_sum(a, b):
           s = a + b
           return prime_factors[s]
           
       # Initialize
       # Try to start with a good configuration?
       # Random start
       a = random.sample(range(1, MAX_VAL + 1), 12)
       
       # State
       # counts[p] = number of pairs with prime p
       # We need a dict or array. Max prime is ~20000.
       # Array is faster.
       prime_counts = [0] * (MAX_SUM + 1)
       active_primes = 0
       
       def update_state(add=True):
           nonlocal active_primes
           for i in range(12):
               for j in range(i+1, 12):
                   s = a[i] + a[j]
                   for p in prime_factors[s]:
                       if add:
                           if prime_counts[p] == 0:
                               active_primes += 1
                           prime_counts[p] += 1
                       else:
                           prime_counts[p] -= 1
                           if prime_counts[p] == 0:
                               active_primes -= 1
                               
       update_state(True)
       
       best_a = a[:]
       best_score = active_primes
       
       # Local Search
       # We can use Simulated Annealing or simple Hill Climbing with random restarts
       
       while time.time() < deadline - 1:
           i = random.randint(0, 11)
           old_val = a[i]
           
           # Generate new value
           # Ensure distinctness
           new_val = random.randint(1, MAX_VAL)
           if new_val in a:
               continue
               
           # Calculate delta
           # Remove old contributions
           delta_active = 0
           # We need to simulate the change
           # For each j != i:
           #   old_sum = old_val + a[j]
           #   new_sum = new_val + a[j]
           
           # We can't just modify prime_counts directly without rolling back if rejected.
           # Better: calculate the change in active_primes without modifying global state,
           # or modify and rollback.
           # Modify and rollback is safer for correctness but slower if we have to restore.
           # However, the number of affected primes is small.
           
           affected_primes = {}
           
           # Remove old
           for j in range(12):
               if j == i: continue
               s_old = old_val + a[j]
               for p in prime_factors[s_old]:
                   affected_primes[p] = affected_primes.get(p, 0) - 1
                   
           # Add new
           for j in range(12):
               if j == i: continue
               s_new = new_val + a[j]
               for p in prime_factors[s_new]:
                   affected_primes[p] = affected_primes.get(p, 0) + 1
                   
           # Calculate change in active_primes
           # A prime p becomes active if count goes 0 -> 1
           # A prime p becomes inactive if count goes 1 -> 0
           
           change = 0
           for p, delta in affected_primes.items():
               old_count = prime_counts[p]
               new_count = old_count + delta
               
               if old_count == 0 and new_count > 0:
                   change += 1
               elif old_count > 0 and new_count == 0:
                   change -= 1
                   
           # Acceptance
           # Simple Hill Climbing: accept if change < 0
           # Or accept if change <= 0 to allow plateau exploration?
           # Let's accept if change < 0, or (change == 0 and random() < 0.5)
           
           if change < 0 or (change == 0 and random.random() < 0.5):
               # Apply change
               a[i] = new_val
               for p, delta in affected_primes.items():
                   prime_counts[p] += delta
               active_primes += change
               
               if active_primes < best_score:
                   best_score = active_primes
                   best_a = a[:]
           else:
               # Reject, no state change
               pass
               
       return best_a
   