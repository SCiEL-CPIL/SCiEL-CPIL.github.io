import math
import random

def construct():
       # 1. Greedy Initialization
       # Sort by weight descending (which is 'a' descending)
       candidates = list(range(511, 0, -1))
       chosen = []
       for a in candidates:
           if all(math.gcd(a, b) == 1 for b in chosen):
               chosen.append(a)
       
       # 2. Local Search
       # Precompute weights
       weights = {a: 1.0 / (512 - a) for a in range(1, 512)}
       
       # Helper to check if x is coprime to all in set S
       def is_coprime_to_set(x, S):
           for y in S:
               if math.gcd(x, y) > 1:
                   return False
           return True
           
       # Helper to get current score
       def get_score(S):
           return sum(weights[a] for a in S)
           
       current_score = get_score(chosen)
       improved = True
       # Limit iterations to prevent infinite loops (though score is bounded and discrete, it should converge)
       max_iters = 1000
       iter_count = 0
       
       while improved and iter_count < max_iters:
           improved = False
           iter_count += 1
           
           # Try 1-swap: remove a_out, add a_in
           # We iterate over all a_out in chosen
           for i, a_out in enumerate(chosen):
               # Create a view of the set without a_out
               # To avoid modifying list during iteration, we work on a copy or indices
               rest = [chosen[j] for j in range(len(chosen)) if j != i]
               
               # Find best a_in
               best_in = -1
               best_gain = 0
               
               # We can prune: only consider a_in with weight > weight[a_out]
               # And a_in not in chosen
               for a_in in candidates:
                   if a_in in chosen:
                       continue
                   if weights[a_in] <= weights[a_out]:
                       continue
                   
                   if is_coprime_to_set(a_in, rest):
                       gain = weights[a_in] - weights[a_out]
                       if gain > best_gain:
                           best_gain = gain
                           best_in = a_in
                           
               if best_in != -1:
                   # Perform swap
                   chosen[i] = best_in
                   current_score += best_gain
                   improved = True
                   # Restart search from this new state to find further improvements
                   # Breaking out of loops to restart the while loop is safer for convergence
                   break
           
           # If we broke out of the for loop due to improvement, we restart the while loop
           # If we finished the for loop without improvement, improved remains False
       
       return chosen
