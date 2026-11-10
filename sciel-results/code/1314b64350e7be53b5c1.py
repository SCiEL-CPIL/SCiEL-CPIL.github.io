   def construct():
       S = []
       for x in range(1, 513):
           if len(S) >= 48: break
           
           # 1. Check if x is mean of S
           if _is_mean_of_subset(x, S):
               continue
               
           # 2. Check if any a in S becomes mean of S+{x} involving x
           valid = True
           for a in S:
               if _is_mean_involving_x(a, S, x):
                   valid = False
                   break
           
           if valid:
               S.append(x)
               
       return S
   