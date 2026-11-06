import time

def construct():
       start_time = time.time()
       deadline = 120.0
       
       # 1. Build Grötzsch Graph
       n_core = 11
       edges = []
       # C5
       for i in range(5):
           edges.append([i, (i + 1) % 5])
       # Mycielski
       for i in range(5):
           u = i
           v = (i + 1) % 5
           edges.append([u, 5 + v])
           edges.append([v, 5 + u])
       # Root
       for i in range(5):
           edges.append([10, 5 + i])
           
       # 2. Augment Core
       # Build adjacency
       adj = [set() for _ in range(16)]
       for u, v in edges:
           adj[u].add(v)
           adj[v].add(u)
           
       # Try to add edges to core (0..10)
       added = 0
       for i in range(11):
           if time.time() > deadline - 10:
               break
           for j in range(i + 1, 11):
               if j in adj[i]:
                   continue
               # Check triangle free
               if adj[i].isdisjoint(adj[j]):
                   edges.append([i, j])
                   adj[i].add(j)
                   adj[j].add(i)
                   added += 1
                   
       # 3. Add 5 new vertices (11..15)
       # Identify independent sets in the augmented core
       # We need to find the best assignment for 11..15
       
       # Strategy: Find max independent sets in core.
       # Use the previous logic: One hub, four leaves.
       
       # Find all maximal independent sets?
       # Or just heuristic:
       # The set V' = {5,6,7,8,9} was independent.
       # Is it still independent?
       # We added edges within 0..10.
       # If we added edges involving 5..9, V' might not be independent.
       
       # Let's find the largest independent set in the core.
       # And a second largest disjoint one?
       
       # Simple heuristic:
       # Try to assign neighborhoods from a list of candidate independent sets.
       
       candidates = []
       # Generate independent sets greedily
       # ...
       
       # For now, let's assume the previous structure works if we didn't break it.
       # If we added edges, we need to re-evaluate.
       
       # To be safe and fast, let's just use the 47-edge construction 
       # and see if the core augmentation helps.
       
       # If core augmentation added edges, we might have broken the independence of V'.
       # If V' is not independent, we can't use it as a neighborhood.
       
       # Let's check if V' is still independent.
       V_prime = set([5,6,7,8,9])
       is_indep = True
       for u in V_prime:
           for v in V_prime:
               if u < v and v in adj[u]:
                   is_indep = False
                   break
           if not is_indep:
               break
       
       if is_indep:
           # Use the 47-edge construction
           # Hub 12: {10, 0, 2}
           # Check if {10, 0, 2} is independent
           hub_set = {10, 0, 2}
           if all(x not in adj[y] or x == y for x in hub_set for y in hub_set):
               # It is independent
               edges.append([12, 10])
               edges.append([12, 0])
               edges.append([12, 2])
               
               for v in [11, 13, 14, 15]:
                   for u in V_prime:
                       edges.append([v, u])
               
               # Connect Hub to Leaves
               edges.append([12, 11])
               edges.append([12, 13])
               edges.append([12, 14])
               edges.append([12, 15])
           else:
               # Fallback: just connect to V_prime
               for v in [11, 12, 13, 14, 15]:
                   for u in V_prime:
                       edges.append([v, u])
       else:
           # Fallback
           for v in [11, 12, 13, 14, 15]:
               # Find any independent set
               # Just use {0, 2} if independent?
               # This is getting complex.
               pass
               
       return edges
