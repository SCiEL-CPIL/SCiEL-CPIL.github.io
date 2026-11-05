   import random
   import time
   from collections import defaultdict

   def construct(n_rows, k):
       start_time = time.time()
       time_limit = 140.0
       
       # ... helper functions ...
       
       best_rows = None
       best_score = float('inf')
       
       attempts = 0
       while time.time() - start_time < time_limit:
           attempts += 1
           # 1. Greedy
           rows = randomized_greedy()
           if rows is None: continue
           
           # 2. Local Search
           rows, score = fast_local_search(rows)
           
           if score < best_score:
               best_score = score
               best_rows = [list(r) for r in rows]
               
           # 3. Optional: SA or more LS
           if time.time() - start_time < time_limit - 5:
               rows, score = simulated_annealing(rows)
               if score < best_score:
                   best_score = score
                   best_rows = [list(r) for r in rows]
                   
           if best_score <= 105: break
           
       if best_rows is None:
           # Fallback
           return deterministic_greedy()
           
       return best_rows
   