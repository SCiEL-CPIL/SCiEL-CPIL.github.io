import numpy as np
import time

def construct():
       N = 51975
       start_time = time.time()
       deadline = start_time + 115.0
       
       # Get divisors
       divisors = [d for d in range(2, N + 1) if N % d == 0]
       
       best_score = N
       best_classes = []
       
       def run_strategy(tie_breaker):
           coverage = np.zeros(N, dtype=np.int8)
           active = {} # m -> r
           inactive = set(divisors)
           
           # Greedy
           while inactive:
               best_m = None
               best_r = None
               best_gain = 0
               
               # Get candidates
               # To allow tie-breaking, we collect all max gains
               max_gain = 0
               candidates = []
               
               for m in inactive:
                   # Find best r for this m
                   # We can vectorize the gain calculation for all r at once?
                   # No, r is index.
                   # Loop r
                   for r in range(m):
                       gain = np.sum(coverage[r::m] == 0)
                       if gain > max_gain:
                           max_gain = gain
                           candidates = [(m, r)]
                       elif gain == max_gain:
                           candidates.append((m, r))
               
               if max_gain == 0:
                   break
                   
               # Tie break
               m, r = tie_breaker(candidates)
               
               active[m] = r
               inactive.remove(m)
               coverage[r::m] += 1
           
           # Local Search
           # Optimize residues
           improved = True
           while improved and time.time() < deadline:
               improved = False
               # Shuffle active moduli to avoid bias?
               items = list(active.items())
               np.random.shuffle(items)
               
               for m, r_old in items:
                   # Calculate L = count(coverage[i]==1 for i in r_old class)
                   L = np.sum(coverage[r_old::m] == 1)
                   
                   # Calculate G(r) for all r
                   # G(r) = count(coverage[i]==0 for i in r class)
                   # We want max G(r)
                   # We can compute an array G of size m
                   G = np.zeros(m, dtype=np.int32)
                   # Iterate i where coverage[i]==0
                   # This is sparse if coverage is high.
                   # Or just loop r and sum?
                   # Looping r and summing is m * (N/m) = N ops.
                   # Using np.sum in a loop is slow due to overhead.
                   # Better:
                   # Create array of zeros indices?
                   # Or just:
                   for r in range(m):
                       G[r] = np.sum(coverage[r::m] == 0)
                   
                   best_r = np.argmax(G)
                   best_gain = G[best_r]
                   
                   if best_gain - L > 0: # Improvement
                       # Apply change
                       coverage[r_old::m] -= 1
                       coverage[best_r::m] += 1
                       active[m] = best_r
                       improved = True
                       # Break and restart pass? Or continue?
                       # Continue is fine, but L for other m's is now stale?
                       # No, L depends on coverage. Coverage changed.
                       # So we should probably restart the pass or recompute.
                       # Restarting is safer.
                       break
           
           # Calculate score
           score = np.sum(coverage == 0)
           return score, list(active.items())
       
       # Strategies
       # 1. Tie break: smallest m
       s1, c1 = run_strategy(lambda c: min(c, key=lambda x: x[0]))
       if s1 < best_score:
           best_score = s1
           best_classes = c1
           
       # 2. Tie break: largest m
       if time.time() < deadline:
           s2, c2 = run_strategy(lambda c: max(c, key=lambda x: x[0]))
           if s2 < best_score:
               best_score = s2
               best_classes = c2
       
       # 3. Random tie break
       if time.time() < deadline:
           s3, c3 = run_strategy(lambda c: c[np.random.randint(len(c))])
           if s3 < best_score:
               best_score = s3
               best_classes = c3
               
       # Format output
       return [[m, r] for m, r in best_classes]
