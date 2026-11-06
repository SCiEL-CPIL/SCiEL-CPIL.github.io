import math
from functools import lru_cache

def construct():
      target_val = math.factorial(12)
      
      # Generate divisors
      divisors = []
      i = 1
      while i * i <= target_val:
          if target_val % i == 0:
              divisors.append(i)
              if i != target_val // i:
                  divisors.append(target_val // i)
          i += 1
          
      # Filter and sort
      valid_divs = sorted([d for d in divisors if d > 12])
      
      if not valid_divs:
          return [target_val] # Fallback, though 12! > 12
      
      # Precompute vectors
      primes = [2, 3, 5, 7, 11]
      def get_vec(n):
          v = [0]*5
          for idx, p in enumerate(primes):
              while n % p == 0:
                  v[idx] += 1
                  n //= p
          return tuple(v)
          
      target_vec = get_vec(target_val)
      div_vecs = {d: get_vec(d) for d in valid_divs}
      
      def check(pool_indices):
          # pool_indices is a list of indices into valid_divs
          # Actually, just pass the list of divisors
          pool = valid_divs[:pool_indices] if isinstance(pool_indices, int) else pool_indices
          
          reachable = { (0,0,0,0,0) }
          path = {}
          
          for d in pool:
              v = div_vecs[d]
              new_reachable = set()
              for s in reachable:
                  s_new = tuple(s[i] + v[i] for i in range(5))
                  # Check bounds
                  valid = True
                  for i in range(5):
                      if s_new[i] > target_vec[i]:
                          valid = False
                          break
                  if valid:
                      new_reachable.add(s_new)
                      if s_new not in path:
                          path[s_new] = (s, d)
                      
              reachable = reachable.union(new_reachable)
              
              if target_vec in reachable:
                  # Reconstruct
                  factors = []
                  curr = target_vec
                  while curr != (0,0,0,0,0):
                      prev, d = path[curr]
                      factors.append(d)
                      curr = prev
                  return factors
          
          return None

      # Binary search
      lo = 0
      hi = len(valid_divs) - 1
      best_factors = None
      best_M = valid_divs[-1]
      
      # We want the smallest M.
      # check(i) means using valid_divs[0..i]
      # If check(i) is True, then M = valid_divs[i] is a candidate.
      # We want the smallest i such that check(i) is True.
      
      # Note: check(i) is monotonic?
      # If a solution exists with max factor M, then a solution exists with max factor M' > M?
      # Yes, because the pool for M' is a superset of the pool for M.
      # So if we can form 12! with factors <= M, we can also form it with factors <= M' (by using the same set).
      # So the predicate "solution exists" is monotonic increasing with M.
      
      while lo <= hi:
          mid = (lo + hi) // 2
          # Pool is valid_divs[0..mid]
          res = check(mid + 1) # Pass count or slice?
          
          # Let's adjust check to take an integer end_index
          # Re-defining check logic inline or passing slice is easier.
          # For speed, passing slice creates a new list.
          # valid_divs is small (780). Slicing is fine.
          
          if res is not None:
              best_factors = res
              best_M = valid_divs[mid]
              hi = mid - 1
          else:
              lo = mid + 1
              
      if best_factors is None:
          # Fallback to trivial solution
          return [target_val]
      
      return best_factors
