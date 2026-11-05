   def _check_conflict(x, S, sum_set):
       """Check if adding x to S (with sum_set) causes any triple sum collision."""
       n = len(S)
       new_sums = set()
       
       # 1. 3x
       val = 3 * x
       if val in sum_set or val in new_sums:
           return True
       new_sums.add(val)
       
       # 2. 2x + s
       for s in S:
           val = 2 * x + s
           if val in sum_set or val in new_sums:
               return True
           new_sums.add(val)
           
           val = x + 2 * s
           if val in sum_set or val in new_sums:
               return True
           new_sums.add(val)
           
       # 3. x + s_i + s_j
       for i in range(n):
           for j in range(i + 1, n):
               val = x + S[i] + S[j]
               if val in sum_set or val in new_sums:
                   return True
               new_sums.add(val)
               
       return False
   