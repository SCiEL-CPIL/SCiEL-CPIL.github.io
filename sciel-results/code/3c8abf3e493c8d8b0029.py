   def is_valid_addition(k, S):
       if not S:
           return True
       
       S_list = list(S)
       n = len(S_list)
       
       # Compute distances from k to all in S
       if D is not None:
           dqs = D[k, S_list]
       else:
           dqs = [get_d(k, s) for s in S_list]
           
       # Check apex at k: all dqs distinct
       if len(set(dqs)) != n:
           return False
       
       # Map point to distance from k for quick lookup?
       # Or just use indices.
       # dqs[i] is distance from k to S_list[i]
       
       for i in range(n):
           for j in range(i + 1, n):
               s1 = S_list[i]
               s2 = S_list[j]
               d_s1s2 = get_d(s1, s2)
               
               d_k_s1 = dqs[i]
               d_k_s2 = dqs[j]
               
               if d_s1s2 == d_k_s1 or d_s1s2 == d_k_s2:
                   return False
                   
       return True
   