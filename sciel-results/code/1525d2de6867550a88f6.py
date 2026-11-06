def construct():
       import numpy as np
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
           def bronk(R, P, X):
               if not P and not X:
                   if len(R) >= 2:
                       cliques.append(frozenset(R))
                   return
               
               if P:
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

       def solve_hitting_set(cliques, n):
           if not cliques:
               return []
           
           # Convert cliques to bitmasks
           clique_masks = []
           for C in cliques:
               mask = 0
               for v in C:
                   mask |= (1 << v)
               clique_masks.append(mask)
           
           # Sort cliques by size (ascending) for better pruning?
           # Smaller cliques are harder to hit? No, they require fewer choices.
           # Actually, in B&B, we pick a clique to branch on.
           # Picking a small clique reduces the branching factor.
           clique_masks.sort(key=lambda m: bin(m).count('1'))
           
           best_size = n # Upper bound
           best_set = set(range(n))
           
           # Precompute vertex frequencies for ordering?
           # We will branch on vertices in a selected clique.
           
           def lower_bound_packing(cliques_list):
               # Greedy disjoint clique packing
               # Return size of packing
               packed = 0
               used = 0
               # Sort by size ascending to pack more?
               # Just iterate
               for mask in cliques_list:
                   if mask & used == 0:
                       packed += 1
                       used |= mask
                       if packed >= best_size: # Prune early if bound is too high
                           return packed
               return packed
           
           def search(removed_mask, candidates_mask, active_cliques):
               nonlocal best_size, best_set
               
               # Prune: if current removed count >= best, stop
               if bin(removed_mask).count('1') >= best_size:
                   return
               
               if not active_cliques:
                   # Found a solution
                   if bin(removed_mask).count('1') < best_size:
                       best_size = bin(removed_mask).count('1')
                       # Reconstruct set
                       best_set = set(i for i in range(n) if (removed_mask >> i) & 1)
                   return
               
               # Lower bound check
               # We need to hit all active_cliques.
               # Max disjoint cliques is a LB on additional hits needed.
               lb = lower_bound_packing(active_cliques)
               if bin(removed_mask).count('1') + lb >= best_size:
                   return
               
               # Select a clique to branch on
               # Choose the one with minimum size? Or max frequency?
               # Minimum size reduces branching factor.
               # Let's pick the first one in the sorted list (smallest).
               # But active_cliques changes.
               # We should pick the clique in active_cliques with smallest size.
               # To do this efficiently, we can just iterate and find min.
               min_clique = None
               min_size = n+1
               for mask in active_cliques:
                   sz = bin(mask).count('1')
                   if sz < min_size:
                       min_size = sz
                       min_clique = mask
               
               if min_clique is None:
                   return
               
               # Branch on each vertex in min_clique
               # Get list of vertices in min_clique
               v_list = [i for i in range(n) if (min_clique >> i) & 1]
               
               # Heuristic: Try vertices with high degree in active cliques first?
               # Or just iterate.
               for v in v_list:
                   v_bit = 1 << v
                   
                   # Check if v is already removed? No, v is in min_clique, 
                   # and min_clique is in active_cliques, so v is not removed.
                   
                   new_removed = removed_mask | v_bit
                   new_candidates = candidates_mask & ~v_bit
                   
                   # Filter active cliques: remove those containing v
                   new_active = []
                   for mask in active_cliques:
                       if not (mask & v_bit):
                           new_active.append(mask)
                   
                   search(new_removed, new_candidates, new_active)
                   
                   # Optional: Update best_size inside loop for tighter pruning?
                   # The nonlocal update happens in the base case.
           
           # Initial call
           all_vertices_mask = (1 << n) - 1
           # active_cliques is list of all clique masks
           search(0, all_vertices_mask, clique_masks)
           
           return sorted(best_set)

       out = []
       for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
           edges = instance_edges(n, p, seed)
           cliques = get_maximal_cliques(n, edges)
           sol = solve_hitting_set(cliques, n)
           out.append(sol)
           
       return out
