import time


def construct():
       start_time = time.time()
       deadline = start_time + 55.0

       # Initial set
       S = set(range(1, 129))
       
       # Helper to check if valid
       def is_valid(S_set):
           S_list = sorted(S_set)
           n = len(S_list)
           for i in range(n):
               a = S_list[i]
               for j in range(i+1, n):
                   b = S_list[j]
                   if a + b not in S_set:
                       continue
                   for k in range(j+1, n):
                       c = S_list[k]
                       if a + c in S_set and b + c in S_set:
                           return False, (a,b,c)
           return True, None

       # Greedy removal
       while True:
           if time.time() > deadline:
               break
           
           valid, triple = is_valid(S)
           if valid:
               break
           
           a, b, c = triple
           candidates = {a, b, c, a+b, a+c, b+c}
           candidates = candidates.intersection(S)
           
           best_v = None
           best_score = -1
           
           for v in candidates:
               # Count source degree: pairs (v, w) in S
               src_deg = 0
               for w in S:
                   if v + w in S:
                       src_deg += 1
               
               # Count target degree: pairs (x, y) in S such that x+y=v
               # Note: x < y to avoid double counting? 
               # For counting, we just need existence of pairs.
               # But we want to estimate impact.
               # If v is sum of many pairs, removing v breaks many "sum in S" checks.
               tgt_deg = 0
               # Optimization: only check x < v
               for x in S:
                   if x >= v:
                       break # if S is sorted? S is a set.
                   # We need sorted S for this break
                   pass
               
               # Let's use sorted list for degree calc if needed, or just iterate set
               # Iterating set for tgt_deg is O(N^2) if done naively.
               # N=100. N^2 = 10,000. 6 candidates * 10,000 = 60,000.
               # This is fine.
               
               # Recalculate tgt_deg properly
               tgt_deg = 0
               # We can't easily break from set iteration.
               # Just check all x in S
               for x in S:
                   y = v - x
                   if y in S and x != y: # distinct elements? 
                       # The problem says "three distinct members a,b,c".
                       # The sums are a+b, a+c, b+c.
                       # The pairs forming the sums are distinct elements.
                       # So x != y is required for the pair to be valid?
                       # Yes, a,b,c distinct implies a!=b.
                       tgt_deg += 1
               
               # Double count? x+y=v and y+x=v.
               # Since we iterate all x, we count each pair twice if x!=y.
               # That's fine for comparison.
               
               score = src_deg + tgt_deg
               if score > best_score:
                   best_score = score
                   best_v = v
           
           if best_v is not None:
               S.remove(best_v)
           else:
               # Should not happen
               break

       return list(S)
