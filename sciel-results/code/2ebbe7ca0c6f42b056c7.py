import time
import random
from itertools import combinations

def construct():
         start_time = time.time()
         deadline = start_time + 290  # safety margin
        
         # Fallback: Paley graph on 37
         def get_paley():
             q = 37
             residues = {(x * x) % q for x in range(1, q)}
             return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]
         
         best_n = 37
         best_edges = get_paley()[1]
         
         for n in range(42, 36, -1):
             if time.time() > deadline:
                 break
             
             found = False
             # Try multiple seeds
             for seed in range(1000):
                 if time.time() > deadline:
                     break
                 
                 # Greedy Construction
                 neighbors = [set() for _ in range(n)]
                 edges = []
                 
                 # Candidate edges
                 candidates = [(i, j) for i in range(n) for j in range(i+1, n)]
                 random.seed(seed)
                 random.shuffle(candidates)
                 
                 for u, v in candidates:
                     # Check if adding (u,v) is safe
                     C = neighbors[u] & neighbors[v]
                     safe = True
                     # Check for triangle in C
                     # Optimization: if len(C) < 3, no triangle
                     if len(C) >= 3:
                         C_list = list(C)
                         for a in C_list:
                             if not safe: break
                             for b in C_list:
                                 if a == b: continue
                                 if b in neighbors[a]:
                                     # Check if a and b have common neighbor in C
                                     # neighbors[a] & neighbors[b] & C
                                     if neighbors[a] & neighbors[b] & C:
                                         safe = False
                                         break
                     
                     if safe:
                         neighbors[u].add(v)
                         neighbors[v].add(u)
                         edges.append((u, v))
                 
                 # Repair Step: Try to fix independent sets
                 # We need to ensure no independent set of size 5.
                 # Loop until no independent set of size 5 is found or no more edges can be added
                 max_iterations = 50
                 for _ in range(max_iterations):
                     if time.time() > deadline:
                         break
                     
                     # Find an independent set of size 5
                     # Heuristic: Random sampling or systematic?
                     # Systematic is safer but slower.
                     # Let's try to find one by checking subsets of low-degree vertices?
                     # Or just random sampling of 5 vertices.
                     # If we find one, try to add an edge.
                     
                     found_iset = False
                     # Try random sampling first
                     for _ in range(100):
                         sample = random.sample(range(n), 5)
                         is_indep = True
                         for i in range(5):
                             for j in range(i+1, 5):
                                 if sample[j] in neighbors[sample[i]]:
                                     is_indep = False
                                     break
                             if not is_indep:
                                 break
                         if is_indep:
                             found_iset = True
                             # Try to add an edge within this set
                             added = False
                             for i in range(5):
                                 for j in range(i+1, 5):
                                     u, v = sample[i], sample[j]
                                     if v not in neighbors[u]:
                                         C = neighbors[u] & neighbors[v]
                                         safe = True
                                         if len(C) >= 3:
                                             C_list = list(C)
                                             for a in C_list:
                                                 if not safe: break
                                                 for b in C_list:
                                                     if a == b: continue
                                                     if b in neighbors[a]:
                                                         if neighbors[a] & neighbors[b] & C:
                                                             safe = False
                                                             break
                                         if safe:
                                             neighbors[u].add(v)
                                             neighbors[v].add(u)
                                             edges.append((u, v))
                                             added = True
                                             break
                                 if added:
                                     break
                             break
                     
                     if not found_iset:
                         # No independent set found in sampling.
                         # We should do a full check to be sure.
                         # Full check:
                         valid = True
                         for combo in combinations(range(n), 5):
                             is_indep = True
                             for i in range(5):
                                 for j in range(i+1, 5):
                                     if combo[j] in neighbors[combo[i]]:
                                         is_indep = False
                                         break
                                 if not is_indep:
                                     break
                             if is_indep:
                                 valid = False
                                 break
                         if valid:
                             # Also need to check for K5, but greedy ensures no K5
                             # unless repair added one? Repair checks safety, so no K5.
                             return n, edges
                         else:
                             # Still has independent set, but repair didn't find one in sampling?
                             # Or repair couldn't fix it.
                             break
                     
                     # If found_iset, we loop again to see if it's fixed
                 
                 # If we exited repair loop, check validity one last time
                 # (The full check above handles this)
             
             if found:
                 break
         
         return best_n, best_edges
