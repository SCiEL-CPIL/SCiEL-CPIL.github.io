   def construct():
       n, m, p = 2, 3, 9
       # Rank 18
       rank = 18
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       
       # ... fill ...
       return {"U": U, "V": V, "W": W}
   