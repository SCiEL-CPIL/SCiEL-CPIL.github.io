import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 55.0 # 5s margin
       
       max_val = 256
       max_size = 60
       
       # dp[k] is a bitset of sums
       # Max sum for 60 elements of 256 is 15360.
       # Python ints are arbitrary precision, so bitsets are fine.
       
       def is_valid_addition(dp, current_sum_bits, x, n_current):
           # Check if adding x to a set of size n_current is valid
           # New sums for size k are (dp[k-1] << x)
           # We need to ensure these new sums don't intersect with existing dp[m] for m != k
           # Note: dp is the state BEFORE adding x.
           # We iterate k from n_current down to 1.
           # Actually, we just need to check the NEW sums.
           
           # The new sums generated for size k are: new_sums_k = (dp[k-1] << x)
           # These are added to dp[k].
           # We must ensure new_sums_k does not intersect with:
           # 1. dp[m] for all m != k (existing sums of other cardinalities)
           # 2. Note: new_sums_k might intersect with dp[k] (existing sums of same cardinality) - this is ALLOWED.
           # 3. new_sums_k might intersect with new_sums_j for j != k?
           #    If we add x, it generates new sums for ALL k.
           #    So new_sums_k and new_sums_j are both new.
           #    Do they need to be disjoint?
           #    The condition is that the FINAL set of sums for size k and size j are disjoint.
           #    If new_sums_k intersects new_sums_j, then the final sets intersect.
           #    So yes, new_sums_k must be disjoint from new_sums_j for k != j.
           #    And new_sums_k must be disjoint from OLD dp[m] for m != k.
           
           # So we need to check:
           # For all k, let N_k = (dp[k-1] << x)
           # For all k != m:
           #   N_k & N_m == 0
           #   N_k & dp[m] == 0  (for m != k, dp[m] is old state)
           #   N_k & dp[k] == 0 is NOT required (collision within same k is ok)
           
           # Efficient check:
           # Let Union_New = OR of all N_k.
           # This doesn't help directly because we need pairwise disjointness of N_k's.
           
           # Let's just compute N_k for all k and check.
           n_new = n_current + 1
           new_sums = [0] * (n_new + 1)
           for k in range(1, n_new + 1):
               new_sums[k] = (dp[k-1] << x)
               
           # Check collisions between new sums of different cardinalities
           for k in range(1, n_new + 1):
               nk = new_sums[k]
               if nk == 0: continue
               for m in range(1, n_new + 1):
                   if m == k: continue
                   # Check against other new sums
                   if nk & new_sums[m]:
                       return False
                   # Check against old sums of cardinality m
                   # Note: dp[m] is the old state.
                   # If m > n_current, dp[m] is 0.
                   if m <= n_current and (nk & dp[m]):
                       return False
                       
           return True

       dp = [0] * (max_size + 1)
       dp[0] = 1
       current_set = []
       used = [False] * (max_val + 1)
       
       # Heuristic: Try numbers in a specific order.
       # Let's try random restarts with different orderings.
       
       best_size = 0
       best_set = []
       
       for attempt in range(50):
           if time.time() > deadline:
               break
           
           # Shuffle candidates or use a specific strategy
           # Strategy 1: Increasing
           # Strategy 2: Decreasing
           # Strategy 3: Random
           
           if attempt % 3 == 0:
               candidates = list(range(1, max_val + 1))
           elif attempt % 3 == 1:
               candidates = list(range(max_val, 0, -1))
           else:
               candidates = list(range(1, max_val + 1))
               random.shuffle(candidates)
               
           # Reset state
           dp = [0] * (max_size + 1)
           dp[0] = 1
           current_set = []
           used = [False] * (max_val + 1)
           
           for c in candidates:
               if len(current_set) >= max_size:
                   break
               if used[c]:
                   continue
               
               if is_valid_addition(dp, None, c, len(current_set)):
                   # Add c
                   current_set.append(c)
                   used[c] = True
                   # Update dp
                   for k in range(len(current_set), 0, -1):
                       dp[k] |= (dp[k-1] << c)
                   
           if len(current_set) > best_size:
               best_size = len(current_set)
               best_set = current_set[:]
               
           if best_size == max_size:
               break
               
       return best_set
