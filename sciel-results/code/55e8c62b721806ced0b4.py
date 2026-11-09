import time
import random
from itertools import combinations

def construct():
       start_time = time.time()
       deadline = start_time + 55.0
       
       n = 16
       k = 4
       
       all_masks = []
       for combo in combinations(range(n), k):
           mask = 0
           for x in combo:
               mask |= (1 << x)
           all_masks.append(mask)
       
       best_family_size = 0
       best_family = []
       
       while time.time() < deadline:
           random.shuffle(all_masks)
           F = []
           
           for mask in all_masks:
               # Check compatibility with F
               compatible = True
               
               if F:
                   # Group F by intersection with mask
                   groups = {}
                   for i, f_mask in enumerate(F):
                       v = f_mask & mask
                       if v not in groups:
                           groups[v] = []
                       groups[v].append(f_mask)
                   
                   for v, group_masks in groups.items():
                       # Check if any two in group_masks are disjoint outside v
                       # i.e., (a & b) == v
                       # Since a, b contain v, a & b contains v.
                       # We need (a & b) == v.
                       # This is equivalent to ((a & b) & ~v) == 0
                       # Or (a & ~v) & (b & ~v) == 0
                       L = len(group_masks)
                       for i in range(L):
                           a = group_masks[i]
                           a_rem = a & ~v
                           for j in range(i+1, L):
                               b = group_masks[j]
                               b_rem = b & ~v
                               if (a_rem & b_rem) == 0:
                                   compatible = False
                                   break
                           if not compatible:
                               break
                   
                   if not compatible:
                       continue
               
               F.append(mask)
               
           if len(F) > best_family_size:
               best_family_size = len(F)
               best_family = F.copy()
       
       result = []
       for mask in best_family:
           elems = []
           for i in range(n):
               if mask & (1 << i):
                   elems.append(i+1)
           result.append(sorted(elems))
       
       return result
