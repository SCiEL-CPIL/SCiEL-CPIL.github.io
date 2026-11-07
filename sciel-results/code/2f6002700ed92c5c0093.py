import numpy as np
import time

class Optimizer:
       def __init__(self, budget, dim, n_obj):
           self.budget = budget
           self.dim = dim
           self.n_obj = n_obj

       def __call__(self, problem, bounds):
           t0 = time.time()
           deadline = t0 + 290.0
           dim = self.dim
           n_obj = self.n_obj
           lo = np.array([b[0] for b in bounds])
           hi = np.array([b[1] for b in bounds])
           span = hi - lo
           
           # 1. Initial Sampling: Latin Hypercube
           n_init = 40
           X_init = np.empty((n_init, dim))
           for i in range(dim):
               X_init[:, i] = lo[i] + span[i] * (np.random.rand(n_init) + np.arange(n_init)) / n_init
           idx = np.random.permutation(n_init)
           X_init = X_init[idx]
           
           F_init = np.array([problem(x) for x in X_init])
           used = n_init
           
           X_all = X_init.copy()
           F_all = F_init.copy()
           
           # 2. Evolutionary Search (NSGA-II like)
           pop_size = 20
           
           # Initialize population from initial samples
           # Use non-dominated points if available, otherwise random
           nd_idx = self._get_non_dominated(F_all)
           
           if len(nd_idx) >= pop_size:
               X_pop = X_all[nd_idx[:pop_size]].copy()
               F_pop = F_all[nd_idx[:pop_size]].copy()
           else:
               X_pop = np.empty((pop_size, dim))
               F_pop = np.empty((pop_size, n_obj))
               X_pop[:len(nd_idx)] = X_all[nd_idx]
               F_pop[:len(nd_idx)] = F_all[nd_idx]
               remaining = pop_size - len(nd_idx)
               # Fill with random points from X_all not in nd_idx
               all_idx = np.arange(len(X_all))
               nd_set = set(nd_idx.tolist())
               available = [i for i in all_idx if i not in nd_set]
               if len(available) >= remaining:
                   fill_idx = np.random.choice(available, remaining, replace=False)
               else:
                   fill_idx = np.random.choice(len(X_all), remaining, replace=True)
               X_pop[len(nd_idx):] = X_all[fill_idx]
               F_pop[len(nd_idx):] = F_all[fill_idx]
               
           # Keep track of all evaluated points for final result
           X_evaluated = X_all.copy()
           F_evaluated = F_all.copy()
           
           # Evolution loop
           while used < self.budget and time.time() < deadline:
               # Generate offspring
               n_offspring = pop_size
               X_off = np.empty((n_offspring, dim))
               F_off = np.empty((n_offspring, n_obj))
               
               for k in range(n_offspring):
                   # Tournament Selection (size 2)
                   p1_idx, p2_idx = self._tournament_select(F_pop)
                   
                   # Crossover
                   child = self._crossover(X_pop[p1_idx], X_pop[p2_idx], lo, hi)
                   
                   # Mutation
                   child = self._mutate(child, lo, hi)
                   
                   fc = problem(child)
                   used += 1
                   X_off[k] = child
                   F_off[k] = fc
                   
                   # Update evaluated points
                   X_evaluated = np.vstack([X_evaluated, [child]])
                   F_evaluated = np.vstack([F_evaluated, [fc]])
                   
                   if used >= self.budget or time.time() > deadline:
                       break
               
               if k == 0:
                   break
                   
               # Merge Parent and Offspring
               X_m = np.vstack([X_pop, X_off[:k+1]])
               F_m = np.vstack([F_pop, F_off[:k+1]])
               
               # Non-dominated sort and selection
               X_pop, F_pop = self._select_population(X_m, F_m, pop_size)
           
           return X_evaluated
       
       def _tournament_select(self, F):
           n = len(F)
           idx1 = np.random.randint(n)
           idx2 = np.random.randint(n)
           s1 = np.sum(F[idx1])
           s2 = np.sum(F[idx2])
           return (idx1, idx2) if s1 < s2 else (idx2, idx1)
           
       def _crossover(self, p1, p2, lo, hi):
           alpha = np.random.rand(self.dim)
           child = alpha * p1 + (1 - alpha) * p2
           return np.clip(child, lo, hi)
           
       def _mutate(self, x, lo, hi):
           span = hi - lo
           rate = 0.2
           mask = np.random.rand(self.dim) < rate
           if np.any(mask):
               x[mask] = x[mask] + 0.1 * span[mask] * np.random.randn(np.sum(mask))
           return np.clip(x, lo, hi)
           
       def _get_non_dominated(self, F):
           n = len(F)
           dominated = np.zeros(n, dtype=bool)
           for i in range(n):
               if dominated[i]:
                   continue
               for j in range(n):
                   if i == j or dominated[j]:
                       continue
                   if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                       dominated[j] = True
           return np.where(~dominated)[0]
           
       def _select_population(self, X, F, pop_size):
           n = len(F)
           # Get non-dominated front
           nd_idx = self._get_non_dominated(F)
           
           if len(nd_idx) >= pop_size:
               # If more than pop_size non-dominated, use crowding distance or sum
               sums = np.sum(F[nd_idx], axis=1)
               keep = np.argsort(sums)[:pop_size]
               nd_idx = nd_idx[keep]
               return X[nd_idx].copy(), F[nd_idx].copy()
           else:
               # Add dominated points with best sums
               dominated_idx = np.where(np.isin(np.arange(n), nd_idx, invert=True))[0]
               if len(dominated_idx) > 0:
                   sums_d = np.sum(F[dominated_idx], axis=1)
                   keep_d = np.argsort(sums_d)[:pop_size - len(nd_idx)]
                   dominated_idx = dominated_idx[keep_d]
                   nd_idx = np.concatenate([nd_idx, dominated_idx])
               return X[nd_idx].copy(), F[nd_idx].copy()
