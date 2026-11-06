   import numpy as np
   import random
   import time

   def construct():
       n = 16
       target_edges = 65
       start_time = time.time()
       time_limit = 140.0
       
       # Initial Graph: K_8,8 + (0,1)
       adj = np.zeros((n, n), dtype=int)
       for u in range(8):
           for v in range(8, 16):
               adj[u, v] = 1
               adj[v, u] = 1
       adj[0, 1] = 1
       adj[1, 0] = 1
       
       def get_score(a):
           degs = a.sum(axis=1)
           max_s = 0
           for i in range(n):
               for j in range(i+1, n):
                   if a[i,j]:
                       for k in range(j+1, n):
                           if a[i,k] and a[j,k]:
                               s = degs[i] + degs[j] + degs[k]
                               if s > max_s: max_s = s
           return max_s
       
       current_score = get_score(adj)
       best_score = current_score
       best_adj = adj.copy()
       
       all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
       
       while time.time() - start_time < time_limit:
           if current_score < best_score:
               best_score = current_score
               best_adj = adj.copy()
               
           # Find worst triangle
           degs = adj.sum(axis=1)
           worst_tri = None
           worst_s = -1
           for i in range(n):
               for j in range(i+1, n):
                   if adj[i,j]:
                       for k in range(j+1, n):
                           if adj[i,k] and adj[j,k]:
                               s = degs[i] + degs[j] + degs[k]
                               if s > worst_s:
                                   worst_s = s
                                   worst_tri = (i, j, k)
                               
           if not worst_tri:
               break
           
           # Moves: Remove edge from worst_tri, add random non-edge
           moved = False
           rem_candidates = [(worst_tri[0], worst_tri[1]), (worst_tri[0], worst_tri[2]), (worst_tri[1], worst_tri[2])]
           random.shuffle(rem_candidates)
           
           for ru, rv in rem_candidates:
               if adj[ru, rv] == 0: continue
               
               adj[ru, rv] = 0
               adj[rv, ru] = 0
               
               # Find non-edge
               # To speed up, we can maintain a list of non-edges, but updating it is O(N^2).
               # Just find one.
               ne = None
               for _ in range(20): # Try a few random
                   u, v = random.choice(all_edges)
                   if adj[u, v] == 0:
                       ne = (u, v)
                       break
               if not ne:
                   # Fallback: scan
                   for u in range(n):
                       for v in range(u+1, n):
                           if adj[u, v] == 0:
                               ne = (u, v)
                               break
                       if ne: break
               
               if not ne:
                   adj[ru, rv] = 1
                   adj[rv, ru] = 1
                   continue
               
               au, av = ne
               adj[au, av] = 1
               adj[av, au] = 1
               
               new_score = get_score(adj)
               
               if new_score < current_score:
                   current_score = new_score
                   moved = True
                   break
               elif new_score == current_score and random.random() < 0.2:
                   current_score = new_score
                   moved = True
                   break
               else:
                   adj[au, av] = 0
                   adj[av, au] = 0
                   adj[ru, rv] = 1
                   adj[rv, ru] = 1
           
           if not moved:
               # Random restart or random walk
               # Try a random swap
               ru, rv = random.choice(all_edges)
               if adj[ru, rv] == 0: continue
               
               adj[ru, rv] = 0
               adj[rv, ru] = 0
               
               ne = None
               for _ in range(20):
                   u, v = random.choice(all_edges)
                   if adj[u, v] == 0:
                       ne = (u, v)
                       break
               if not ne:
                   for u in range(n):
                       for v in range(u+1, n):
                           if adj[u, v] == 0:
                               ne = (u, v)
                               break
                       if ne: break
               
               if not ne:
                   adj[ru, rv] = 1
                   adj[rv, ru] = 1
                   continue
                   
               au, av = ne
               adj[au, av] = 1
               adj[av, au] = 1
               
               new_score = get_score(adj)
               
               if new_score < current_score:
                   current_score = new_score
               else:
                   adj[au, av] = 0
                   adj[av, au] = 0
                   adj[ru, rv] = 1
                   adj[rv, ru] = 1
                   
       edges = []
       for i in range(n):
           for j in range(i+1, n):
               if best_adj[i, j]:
                   edges.append([i, j])
       return {'n': n, 'edges': edges}
   