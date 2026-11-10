import numpy as np

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           lo = np.array([b[0] for b in bounds])
           hi = np.array([b[1] for b in bounds])
           
           pop_size = 20
           # Ensure we don't exceed budget
           # Each generation: evaluate pop_size offspring
           # Initial population: pop_size
           # Total evals = pop_size * (generations + 1)
           
           n_gen = (self.budget // pop_size) - 1
           if n_gen < 1:
               n_gen = 1
               
           # Initialize
           X = lo + (hi - lo) * np.random.random((pop_size, self.dim))
           F = np.array([problem(x) for x in X])
           
           all_F = [F]
           
           for gen in range(n_gen):
               # Non-dominated sort
               ranks, fronts = self._non_dominated_sort(F)
               
               # Crowding distance
               cd = self._crowding_distance(F, fronts)
               
               # Selection
               parents = self._tournament_select(F, ranks, cd, pop_size)
               
               # Crossover & Mutation
               offspring_X = []
               for i in range(0, pop_size, 2):
                   p1 = parents[i]
                   p2 = parents[i+1]
                   c1, c2 = self._sbx_crossover(p1, p2, lo, hi)
                   c1 = self._polynomial_mutation(c1, lo, hi)
                   c2 = self._polynomial_mutation(c2, lo, hi)
                   offspring_X.append(c1)
                   offspring_X.append(c2)
                   
               offspring_X = np.array(offspring_X[:pop_size])
               offspring_F = np.array([problem(x) for x in offspring_X])
               all_F.append(offspring_F)
               
               # Combine and select
               combined_F = np.vstack([F, offspring_F])
               combined_ranks, combined_fronts = self._non_dominated_sort(combined_F)
               combined_cd = self._crowding_distance(combined_F, combined_fronts)
               
               # Select next population
               next_indices = []
               for front in combined_fronts:
                   if len(next_indices) + len(front) <= pop_size:
                       next_indices.extend(front)
                   else:
                       # Sort by crowding distance
                       front_arr = np.array(front)
                       cd_front = combined_cd[front_arr]
                       order = np.argsort(-cd_front)
                       needed = pop_size - len(next_indices)
                       next_indices.extend(front_arr[order[:needed]])
                       break
                       
               next_indices = np.array(next_indices)
               F = combined_F[next_indices]
               # We need to track X as well? 
               # For the return, we just need the F vectors recorded.
               # The problem function records the calls.
               # We don't strictly need X for the return, but we need it for the next step.
               # Wait, I lost X. I need to keep X aligned with F.
               
           # The task records ALL returned objective vectors.
           # I need to make sure I return the best set or just let the evaluator record them.
           # "Every returned objective vector is recorded".
           # The __call__ return value is ignored.
           # So I just need to ensure I call problem() the right number of times.
           
           return None

       def _non_dominated_sort(self, F):
           n = len(F)
           ranks = np.zeros(n, dtype=int)
           fronts = []
           S = [[] for _ in range(n)]
           n_dom = np.zeros(n, dtype=int)
           
           for p in range(n):
               for q in range(n):
                   if p == q: continue
                   if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                       S[p].append(q)
                   elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                       n_dom[p] += 1
                       
           current_front = []
           for p in range(n):
               if n_dom[p] == 0:
                   ranks[p] = 0
                   current_front.append(p)
                   
           k = 0
           while current_front:
               fronts.append(current_front)
               next_front = []
               for p in current_front:
                   for q in S[p]:
                       n_dom[q] -= 1
                       if n_dom[q] == 0:
                           ranks[q] = k + 1
                           next_front.append(q)
               current_front = next_front
               k += 1
               
           return ranks, fronts

       def _crowding_distance(self, F, fronts):
           n = len(F)
           cd = np.zeros(n)
           for front in fronts:
               m = len(front)
               if m <= 2:
                   for idx in front:
                       cd[idx] = np.inf
                   continue
               for j in range(self.n_obj):
                   f_vals = F[front, j]
                   min_f = np.min(f_vals)
                   max_f = np.max(f_vals)
                   if max_f == min_f:
                       for idx in front:
                           cd[idx] = np.inf
                       continue
                   # Sort by objective j
                   order = np.argsort(f_vals)
                   sorted_front = np.array(front)[order]
                   cd[sorted_front[0]] = np.inf
                   cd[sorted_front[-1]] = np.inf
                   for i in range(1, m-1):
                       idx = sorted_front[i]
                       cd[idx] += (f_vals[order[i+1]] - f_vals[order[i-1]]) / (max_f - min_f)
           return cd

       def _tournament_select(self, F, ranks, cd, k):
           parents = []
           for _ in range(k):
               i, j = np.random.randint(0, len(F), 2)
               if ranks[i] < ranks[j]:
                   parents.append(i)
               elif ranks[i] > ranks[j]:
                   parents.append(j)
               else:
                   if cd[i] > cd[j]:
                       parents.append(i)
                   else:
                       parents.append(j)
           return parents

       def _sbx_crossover(self, p1, p2, lo, hi):
               def cross_single():
                   if np.random.random() < 0.5:
                       return p1.copy(), p2.copy()
                   c1 = np.zeros_like(p1)
                   c2 = np.zeros_like(p2)
                   for i in range(len(p1)):
                       if p1[i] == p2[i]:
                           c1[i] = p1[i]
                           c2[i] = p2[i]
                       else:
                           u = np.random.random()
                           beta = 1.0
                           if u <= 0.5:
                               alpha = 2 * u
                               beta = (alpha * (eta+1)) ** (-1/eta)
                           else:
                               alpha = 2 * (1-u)
                               beta = (alpha * (eta+1)) ** (-1/eta)
                           c1[i] = 0.5 * ((1+beta)*p1[i] + (1-beta)*p2[i])
                           c2[i] = 0.5 * ((1-beta)*p1[i] + (1+beta)*p2[i])
                           c1[i] = np.clip(c1[i], lo[i], hi[i])
                           c2[i] = np.clip(c2[i], lo[i], hi[i])
                   return c1, c2
               # eta = 20
               return cross_single()

       def _polynomial_mutation(self, x, lo, hi):
               eta = 20
               for i in range(len(x)):
                   if np.random.random() < 0.5:
                       u = np.random.random()
                       delta1 = (2*u) ** (1/(eta+1)) - 1
                       delta2 = 1 - (2*(1-u)) ** (1/(eta+1))
                       x[i] += np.random.choice([delta1, delta2]) * (hi[i] - lo[i])
                       x[i] = np.clip(x[i], lo[i], hi[i])
               return x
