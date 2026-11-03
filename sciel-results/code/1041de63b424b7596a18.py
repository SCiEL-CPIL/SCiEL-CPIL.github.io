import numpy as np
import random
import itertools
import time

def construct():
       D = 11
       supports = list(itertools.combinations(range(D), 4))
       n_sup = len(supports)
       
       # Build adjacency list
       adj = [set() for _ in range(n_sup)]
       for i in range(n_sup):
           s_i = set(supports[i])
           for j in range(i+1, n_sup):
               s_j = set(supports[j])
               if len(s_i.intersection(s_j)) == 3:
                   adj[i].add(j)
                   adj[j].add(i)
                   
       def solve_mis():
           # Greedy init
           remaining = set(range(n_sup))
           current_set = set()
           while remaining:
               # Pick vertex with min degree in remaining
               min_deg = float('inf')
               best_v = None
               for v in remaining:
                   deg = len(adj[v].intersection(remaining))
                   if deg < min_deg:
                       min_deg = deg
                       best_v = v
                       if deg == 0: break
               current_set.add(best_v)
               to_remove = {best_v}.union(adj[best_v].intersection(remaining))
               remaining.difference_update(to_remove)
           
           S = current_set
           best_size = len(S)
           best_set = S.copy()
           
           # SA
           T = 1.0
           cooling = 0.999
           steps = 200000
           
           for _ in range(steps):
               v = random.randint(0, n_sup - 1)
               
               if v in S:
                   # Try removing v? 
                   # If we remove v, we might be able to add multiple neighbors later.
                   # But standard SA on IS usually does swaps.
                   # Let's try a "remove" move with low prob?
                   # Or just skip.
                   continue
               
               N = adj[v].intersection(S)
               
               if len(N) == 0:
                   S.add(v)
                   if len(S) > best_size:
                       best_size = len(S)
                       best_set = S.copy()
               else:
                   delta = 1 - len(N)
                   if delta > 0:
                       S.difference_update(N)
                       S.add(v)
                       if len(S) > best_size:
                           best_size = len(S)
                           best_set = S.copy()
                   else:
                       # Accept with prob exp(delta/T)
                       # delta is negative or zero
                       p = np.exp(delta / T)
                       if random.random() < p:
                           S.difference_update(N)
                           S.add(v)
                           
               T *= cooling
               
           return best_set

       # Run multiple restarts?
       # Time limit 120s.
       # 200k steps * 10 restarts = 2M steps.
       # Each step is O(1) set ops.
       # Should be fast enough.
       
       best_global = set()
       start_time = time.time()
       
       for restart in range(20):
           if time.time() - start_time > 100:
               break
           res = solve_mis()
           if len(res) > len(best_global):
               best_global = res
               
       # Construct vectors
       vectors = []
       for idx in best_global:
           sup = supports[idx]
           for signs in itertools.product([1, -1], repeat=4):
               v = [0] * D
               for k, s in enumerate(sup):
                   v[s] = signs[k]
               vectors.append(v)
               
       return np.array(vectors, dtype=np.int64)
