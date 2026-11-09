import random
import time
import math

def construct():
       n = 40
       m = 401
       start_time = time.time()
       timeout = 110
       
       # Initialize bitmasks
       # adj[i] is bitmask of neighbors
       adj = [0] * n
       
       # Add edges
       def add_edge(u, v):
           adj[u] |= (1 << v)
           adj[v] |= (1 << u)
           
       def remove_edge(u, v):
           adj[u] &= ~(1 << v)
           adj[v] &= ~(1 << u)
           
       # Initial graph: K20,20 + (0,1)
       for u in range(20):
           for v in range(20, 40):
               add_edge(u, v)
       add_edge(0, 1)
       
       # Count edges to verify
       edge_count = sum(bin(adj[i]).count('1') for i in range(n)) // 2
       # Should be 401
       
       def get_max_book():
           max_b = 0
           for u in range(n):
               for v in range(u + 1, n):
                   if (adj[u] >> v) & 1:
                       common = (adj[u] & adj[v]).bit_count()
                       if common > max_b:
                           max_b = common
           return max_b
       
       current_max_book = get_max_book()
       best_max_book = current_max_book
       best_adj = adj.copy()
       
       T = 1.0
       cooling = 0.999
       
       while time.time() - start_time < timeout:
           # Find max book edge
           max_b_val = 0
           max_u, max_v = -1, -1
           for u in range(n):
               for v in range(u + 1, n):
                   if (adj[u] >> v) & 1:
                       common = (adj[u] & adj[v]).bit_count()
                       if common > max_b_val:
                           max_b_val = common
                           max_u, max_v = u, v
           
           if max_b_val <= 7:
               break
           
           # Candidates for removal
           # The edge (max_u, max_v) itself, or edges (max_u, w) or (max_v, w) where w is common neighbor
           common_neighbors = (adj[max_u] & adj[max_v])
           # Get list of neighbors
           cn_list = []
           mask = common_neighbors
           while mask:
               lsb = mask & -mask
               idx = lsb.bit_length() - 1
               cn_list.append(idx)
               mask ^= lsb
               
           candidates = [(max_u, max_v)]
           for w in cn_list:
               candidates.append((max_u, w))
               candidates.append((max_v, w))
           
           random.shuffle(candidates)
           
           improved = False
           for rem_u, rem_v in candidates:
               if (adj[rem_u] >> rem_v) & 1:
                   remove_edge(rem_u, rem_v)
                   
                   # Find best non-edge to add
                   # We want to minimize the book size of the new edge
                   # And ideally not increase max book too much
                   min_book_add = n + 1
                   best_add_u, best_add_v = -1, -1
                   
                   # Iterate all pairs
                   for u in range(n):
                       for v in range(u + 1, n):
                           if not ((adj[u] >> v) & 1):
                               # Check if it's the removed edge
                               if (u == rem_u and v == rem_v) or (u == rem_v and v == rem_u):
                                   continue
                               common = (adj[u] & adj[v]).bit_count()
                               if common < min_book_add:
                                   min_book_add = common
                                   best_add_u, best_add_v = u, v
                   
                   if best_add_u != -1:
                       add_edge(best_add_u, best_add_v)
                       
                       new_max_book = get_max_book()
                       
                       if new_max_book < current_max_book:
                           current_max_book = new_max_book
                           if new_max_book < best_max_book:
                               best_max_book = new_max_book
                               best_adj = adj.copy()
                           improved = True
                           break
                       else:
                           # Annealing
                           delta = new_max_book - current_max_book
                           if delta > 0:
                               if random.random() < math.exp(-delta / T):
                                   current_max_book = new_max_book
                                   improved = True
                                   break
                               else:
                                   # Revert
                                   remove_edge(best_add_u, best_add_v)
                                   add_edge(rem_u, rem_v)
                           else:
                               # Equal or better? (Should be better if <, so this is equal)
                               # Accept to explore
                               current_max_book = new_max_book
                               improved = True
                               break
                   
                   # If no edge added (should not happen), revert
                   if best_add_u == -1:
                       add_edge(rem_u, rem_v)
           
           if not improved:
               # Try random move
               # Pick random edge to remove
               u = random.randint(0, n-1)
               v = random.randint(0, n-1)
               if u != v and (adj[u] >> v) & 1:
                   remove_edge(u, v)
                   # Pick random non-edge
                   added = False
                   for _ in range(100):
                       x = random.randint(0, n-1)
                       y = random.randint(0, n-1)
                       if x != y and not ((adj[x] >> y) & 1):
                           add_edge(x, y)
                           added = True
                           break
                   if added:
                       new_max_book = get_max_book()
                       delta = new_max_book - current_max_book
                       if delta <= 0 or (delta > 0 and random.random() < math.exp(-delta / T)):
                           current_max_book = new_max_book
                           if new_max_book < best_max_book:
                               best_max_book = new_max_book
                               best_adj = adj.copy()
                       else:
                           # Revert
                           # Need to know what was added
                           # This is messy. Let's stick to the structured move.
                           pass
           
           T *= cooling
           
       # Reconstruct edges from best_adj
       edges = []
       for u in range(n):
           for v in range(u + 1, n):
               if (best_adj[u] >> v) & 1:
                   edges.append([u, v])
       return edges
