   def construct():
       n, m, p = 5, 5, 5
       # Rank 94
       rank = 94
       U = [[0.0] * rank for _ in range(25)]
       V = [[0.0] * rank for _ in range(25)]
       W = [[0.0] * rank for _ in range(25)]
       
       col = 0
       
       # 1. A44 B44 using Strassen (49 mults)
       # ... fill U, V, W for cols 0..48
       
       # 2. A44 B41 (16 mults)
       # ...
       
       # 3. A14 B44 (16 mults)
       # ...
       
       # 4. A14 B41 (4 mults)
       # ...
       
       # 5. A11 B11 (1 mult)
       # ...
       
       # 6. A41 B14 (4 mults)
       # ...
       
       # 7. A41 B11 (4 mults)
       # ...
       
       return {"U": U, "V": V, "W": W}
   