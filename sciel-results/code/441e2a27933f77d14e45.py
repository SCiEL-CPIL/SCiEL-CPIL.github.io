def construct():
       import numpy as np
       from scipy.optimize import milp, LinearConstraint, Bounds
       from random import Random

       def instance_edges(n, p, seed):
           rng = Random(seed)
           edges = []
           for u in range(n):
               for v in range(u + 1, n):
                   if rng.random() < p:
                       edges.append((u, v))
           return edges

       def get_maximal_cliques(n, edges):
           adj = [set() for _ in range(n)]
           for u, v in edges:
               adj[u].add(v)
               adj[v].add(u)
           
           cliques = []
           # Bron-Kerbosch with pivot
           def bronk(R, P, X):
               if not P and not X:
                   if len(R) >= 2:
                       cliques.append(frozenset(R))
                   return
               
               if P:
                   # Pivot
                   u = max(P | X, key=lambda u: len(P & adj[u]))
                   candidates = P - adj[u]
               else:
                   candidates = P
                   
               for v in list(candidates):
                   bronk(R | {v}, P & adj[v], X & adj[v])
                   P.remove(v)
                   X.add(v)
                   
           bronk(set(), set(range(n)), set())
           return cliques

       def solve_set_cover(cliques, n):
           if not cliques:
               return []
           
           # Try MILP
           try:
               # Variables x_0 ... x_{n-1}
               # Min sum(x_i)
               c = np.ones(n)
               
               # Constraints: sum_{v in C} x_v >= 1 for each clique C
               # A is matrix of shape (len(cliques), n)
               A = np.zeros((len(cliques), n))
               for i, C in enumerate(cliques):
                   for v in C:
                       A[i, v] = 1
               
               # 1 <= sum(A_row * x) <= inf
               lb = np.ones(len(cliques))
               ub = np.full(len(cliques), np.inf)
               
               constraints = LinearConstraint(A, lb, ub)
               bounds = Bounds(0, 1) # x_i in [0, 1]
               
               # integrality: 1 for all variables
               integrality = np.ones(n)
               
               res = milp(c, constraints=constraints, integrality=integrality, bounds=bounds)
               
               if res.status == 0:
                   sol = [i for i in range(n) if res.x[i] > 0.5]
                   return sorted(sol)
               else:
                   # Fallback to greedy
                   pass
           except Exception:
               pass
               
           # Greedy Fallback
           uncovered = set(range(len(cliques)))
           selected = set()
           
           # Map vertex to list of clique indices containing it
           vert_to_cliques = {v: [] for v in range(n)}
           for i, C in enumerate(cliques):
               for v in C:
                   vert_to_cliques[v].append(i)
                   
           while uncovered:
               best_v = -1
               best_count = 0
               for v in range(n):
                   count = sum(1 for i in vert_to_cliques[v] if i in uncovered)
                   if count > best_count:
                       best_count = count
                       best_v = v
                       
               if best_v == -1:
                   break
                   
               selected.add(best_v)
               for i in vert_to_cliques[best_v]:
                   uncovered.discard(i)
                   
           return sorted(selected)

       out = []
       for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
           edges = instance_edges(n, p, seed)
           cliques = get_maximal_cliques(n, edges)
           sol = solve_set_cover(cliques, n)
           out.append(sol)
           
       return out
