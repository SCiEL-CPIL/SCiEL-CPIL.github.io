import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 300  # 300s limit, keep margin

       # 1. Build Paley Graph P(37)
       q = 37
       residues = {(x * x) % q for x in range(1, q)}
       adj = [0] * q
       edges = []
       for i in range(q):
           for j in range(i + 1, q):
               if (j - i) % q in residues:
                   adj[i] |= 1 << j
                   adj[j] |= 1 << i
                   edges.append((i, j))
       
       n = q
       
       # Helper to check if 4 vertices form a K4
       def is_k4(v1, v2, v3, v4):
           return (adj[v1] & (1 << v2)) and (adj[v1] & (1 << v3)) and (adj[v1] & (1 << v4)) and \
                  (adj[v2] & (1 << v3)) and (adj[v2] & (1 << v4)) and \
                  (adj[v3] & (1 << v4))
       
       # Helper to check if 4 vertices form an Independent Set
       def is_ind4(v1, v2, v3, v4):
           return not (adj[v1] & (1 << v2)) and not (adj[v1] & (1 << v3)) and not (adj[v1] & (1 << v4)) and \
                  not (adj[v2] & (1 << v3)) and not (adj[v2] & (1 << v4)) and \
                  not (adj[v3] & (1 << v4))
       
       # Precompute K4s and Ind4s
       k4s = []
       ind4s = []
       for i in range(n):
           for j in range(i+1, n):
               if not (adj[i] & (1 << j)): continue # Optimization? No, need all
               for k in range(j+1, n):
                   if not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                   for l in range(k+1, n):
                       if (adj[i] & (1 << l)) and (adj[j] & (1 << l)) and (adj[k] & (1 << l)):
                           k4s.append((i, j, k, l))
                       if not (adj[i] & (1 << l)) and not (adj[j] & (1 << l)) and not (adj[k] & (1 << l)):
                           ind4s.append((i, j, k, l))
       
       # Function to try adding a vertex
       def try_add_vertex():
           # Returns neighbor mask or None
           # Use local search
           # Random init
           best_n = random.getrandbits(n)
           best_viol = float('inf')
           
           def count_violations(mask):
               viol = 0
               for k4 in k4s:
                   # Check if k4 is subset of mask
                   if (mask & (1 << k4[0])) and (mask & (1 << k4[1])) and (mask & (1 << k4[2])) and (mask & (1 << k4[3])):
                       viol += 1
               for i4 in ind4s:
                   # Check if i4 is subset of complement of mask
                   if not (mask & (1 << i4[0])) and not (mask & (1 << i4[1])) and not (mask & (1 << i4[2])) and not (mask & (1 << i4[3])):
                       viol += 1
               return viol
           
           best_viol = count_violations(best_n)
           if best_viol == 0:
               return best_n
           
           for step in range(1000):
               if time.time() > deadline - 10:
                   return None
               # Pick a random bit to flip
               bit = random.randint(0, n-1)
               new_n = best_n ^ (1 << bit)
               new_viol = count_violations(new_n)
               if new_viol < best_viol:
                   best_n = new_n
                   best_viol = new_viol
                   if best_viol == 0:
                       return best_n
               # Random restart occasionally
               if step % 100 == 0:
                   best_n = random.getrandbits(n)
                   best_viol = count_violations(best_n)
                   if best_viol == 0:
                       return best_n
           return None
       
       # Loop to add vertices
       added = 0
       for _ in range(10): # Try to add up to 10 vertices
           if time.time() > deadline - 30:
               break
           mask = try_add_vertex()
           if mask is None:
               break
           
           # Add vertex n with neighbors defined by mask
           new_adj = [0] * (n + 1)
           for i in range(n):
               new_adj[i] = adj[i]
               if mask & (1 << i):
                   new_adj[i] |= 1 << n
                   new_adj[n] |= 1 << i
                   edges.append((i, n))
           adj = new_adj
           n += 1
           added += 1
           
           # Update k4s and ind4s
           # New K4s involving vertex n-1
           new_k4s = []
           for i in range(n-1):
               if not (adj[n-1] & (1 << i)): continue
               for j in range(i+1, n-1):
                   if not (adj[n-1] & (1 << j)) or not (adj[i] & (1 << j)): continue
                   for k in range(j+1, n-1):
                       if not (adj[n-1] & (1 << k)) or not (adj[i] & (1 << k)) or not (adj[j] & (1 << k)): continue
                       new_k4s.append((i, j, k, n-1))
           k4s.extend(new_k4s)
           
           new_ind4s = []
           for i in range(n-1):
               if adj[n-1] & (1 << i): continue
               for j in range(i+1, n-1):
                   if (adj[n-1] & (1 << j)) or (adj[i] & (1 << j)): continue
                   for k in range(j+1, n-1):
                       if (adj[n-1] & (1 << k)) or (adj[i] & (1 << k)) or (adj[j] & (1 << k)): continue
                       new_ind4s.append((i, j, k, n-1))
           ind4s.extend(new_ind4s)
       
       return n, edges
