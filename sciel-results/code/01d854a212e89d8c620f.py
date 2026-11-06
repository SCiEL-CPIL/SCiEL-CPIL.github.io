import math
import random
import time

def construct():
       max_mod = 128
       moduli = list(range(2, max_mod + 1))
       
       def is_disjoint(r_new, m_new, existing):
           for r_old, m_old in existing:
               g = math.gcd(m_new, m_old)
               if r_new % g == r_old % g:
                   return False
           return True
       
       def try_greedy(order):
           selected = []
           for m in order:
               # Try all residues 0..m-1
               # Optimization: residues are equivalent mod gcds.
               # But m is small, so just iterate.
               found = False
               for r in range(m):
                   if is_disjoint(r, m, selected):
                       selected.append([r, m])
                       found = True
                       break
               # If not found, we could skip m, or try harder?
               # For now, skip.
           return selected
       
       best = []
       start_time = time.time()
       deadline = start_time + 80  # Safety margin
       
       # First, try deterministic order (e.g., descending)
       # Or random orders
       for _ in range(1000):
           if time.time() > deadline:
               break
           random.shuffle(moduli)
           res = try_greedy(moduli)
           if len(res) > len(best):
               best = res
           
       return best
