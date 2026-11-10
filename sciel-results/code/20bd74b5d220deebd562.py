   def construct():
       import time
       from random import Random
       
       deadline = time.time() + 145.0
       
       def instance(n, k, p, seed):
           rng = Random(seed)
           cross_edges = []
           for v in range(k, n):
               for u in range(k):
                   if rng.random() < p:
                       cross_edges.append((u, v))
           return cross_edges
       
       def solve_instance(n, k, p, seed):
           cross_edges = instance(n, k, p, seed)
           
           # N[v] is list of neighbors in C for v in I
           N = {v: set() for v in range(k, n)}
           for u, v in cross_edges:
               N[v].add(u)
               
           cliques = []
           covered_edges_in_C = set()
           
           # Mixed cliques
           for v in range(k, n):
               if N[v]:
                   clique = [v] + list(N[v])
                   cliques.append(clique)
                   # Add edges within N[v] to covered
                   neighbors = list(N[v])
                   for i in range(len(neighbors)):
                       for j in range(i+1, len(neighbors)):
                           covered_edges_in_C.add((min(neighbors[i], neighbors[j]), max(neighbors[i], neighbors[j])))
           
           # Find remaining edges in C
           remaining_edges = set()
           for u in range(k):
               for w in range(u+1, k):
                   if (u, w) not in covered_edges_in_C:
                       remaining_edges.add((u, w))
           
           # Cover remaining edges with pure cliques
           if remaining_edges:
               # Greedy edge clique cover
               rem = set(remaining_edges)
               while rem:
                   if time.time() > deadline:
                       # Fallback: add edges as 2-cliques
                       for e in rem:
                           cliques.append([e[0], e[1]])
                       break
                   
                   # Find a maximal clique in the graph (C, rem)
                   # Vertices involved in rem
                   verts = set()
                   for u, w in rem:
                       verts.add(u)
                       verts.add(w)
                   
                   # Greedy: pick vertex with max degree in rem
                   # Or pick an edge and expand
                   # Simple: find largest clique
                   # Since k is small, we can try to find a large clique
                   # Start with a vertex with max degree
                   best_clique = []
                   if verts:
                       start_v = max(verts, key=lambda x: sum(1 for y in verts if (min(x,y), max(x,y)) in rem))
                       S = {start_v}
                       cand = set(verts) - {start_v}
                       
                       while cand:
                           # Find vertex in cand connected to all in S
                           best_next = None
                           best_deg = -1
                           for w in cand:
                               # Check if w is connected to all in S
                               is_connected = True
                               deg = 0
                               for s in S:
                                   if (min(w,s), max(w,s)) not in rem:
                                       is_connected = False
                                       break
                                   deg += 1 # Not really degree, just count
                               if is_connected:
                                   # Pick one with max connections to cand?
                                   # Or just pick first?
                                   # To maximize clique size, pick one that keeps cand large?
                                   # Greedy: pick w with max degree in cand
                                   deg_cand = 0
                                   for x in cand:
                                       if x == w: continue
                                       if (min(w,x), max(w,x)) in rem:
                                           deg_cand += 1
                                   if deg_cand > best_deg:
                                       best_deg = deg_cand
                                       best_next = w
                           
                           if best_next is None:
                               break
                           
                           S.add(best_next)
                           # Update cand: keep only neighbors of best_next in cand
                           new_cand = set()
                           for w in cand:
                               if w == best_next: continue
                               if (min(best_next, w), max(best_next, w)) in rem:
                                   new_cand.add(w)
                           cand = new_cand
                       
                       if S:
                           clique = list(S)
                           cliques.append(clique)
                           # Remove edges in S from rem
                           for i in range(len(clique)):
                               for j in range(i+1, len(clique)):
                                   rem.discard((min(clique[i], clique[j]), max(clique[i], clique[j])))
                   else:
                       break
           
           # Ensure types
           clean = []
           for clique in cliques:
               clean.append([int(x) for x in clique])
           return clean

       specs = ((30, 10, 0.7, 301), (60, 20, 0.7, 302), (90, 30, 0.7, 303))
       result = []
       
       for (n, k, p, seed) in specs:
           if time.time() > deadline:
               # Fallback
               ce = instance(n, k, p, seed)
               cliques = []
               N = {v: set() for v in range(k, n)}
               for u, v in ce:
                   N[v].add(u)
               for v in range(k, n):
                   if N[v]:
                       cliques.append([v] + list(N[v]))
               # Cover rest
               covered = set()
               for v in range(k, n):
                   if N[v]:
                       ns = list(N[v])
                       for i in range(len(ns)):
                           for j in range(i+1, len(ns)):
                               covered.add((min(ns[i], ns[j]), max(ns[i], ns[j])))
               rem = set()
               for u in range(k):
                   for w in range(u+1, k):
                       if (u,w) not in covered:
                           rem.add((u,w))
               for e in rem:
                   cliques.append([e[0], e[1]])
               result.append([[int(x) for x in c] for c in cliques])
               continue
           
           result.append(solve_instance(n, k, p, seed))
       
       return result
   