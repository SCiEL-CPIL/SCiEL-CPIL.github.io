   def max_clique(candidates_mask, adj_masks):
       # adj_masks[u] is a bitmask of neighbors of u in the current graph
       # candidates_mask is a bitmask of vertices we can use
       best_clique = 0
       
       def expand(P, X, clique):
           nonlocal best_clique
           if P == 0 and X == 0:
               if bin(clique).count('1') > bin(best_clique).count('1'):
                   best_clique = clique
               return
           
           # Pivot selection
           # Pick u from P ∪ X with max degree in P
           U = P | X
           if U == 0:
               return
           
           # Find pivot
           pivot = -1
           max_deg = -1
           temp = U
           while temp:
               lsb = temp & (-temp)
               u = lsb.bit_length() - 1
               temp ^= lsb
               deg = bin(adj_masks[u] & P).count('1')
               if deg > max_deg:
                   max_deg = deg
                   pivot = u
           
           # Recurse on P \ N(pivot)
           # If pivot is -1, N(pivot) is 0? No, if U=0 we returned.
           # If pivot exists, we iterate over P \ N(pivot)
           # Wait, standard BK iterates over P \ N(pivot).
           # If pivot is not in P, N(pivot) might not be defined in P?
           # Actually, we just need to ensure we visit all maximal cliques.
           
           if pivot == -1:
               neighbors = 0
           else:
               neighbors = adj_masks[pivot]
               
           candidates = P & ~neighbors
           
           temp = candidates
           while temp:
               lsb = temp & (-temp)
               u = lsb.bit_length() - 1
               temp ^= lsb
               
               expand(P & adj_masks[u], X & adj_masks[u], clique | (1 << u))
               
               P &= ~(1 << u)
               X |= (1 << u)
               
               # If P becomes 0, we can break? No, need to clear X?
               # Standard BK:
               # P = P - {u}
               # X = X ∪ {u}
               # If P == 0 and X == 0: found maximal clique
               
       expand(candidates_mask, 0, 0)
       return best_clique
   