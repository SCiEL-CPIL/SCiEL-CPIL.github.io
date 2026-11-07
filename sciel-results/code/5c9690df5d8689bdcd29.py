   def construct():
       n = 10
       sq1 = [[j for j in range(n)] for i in range(n)]
       
       sq2 = [[0] * n for _ in range(n)]
       for i in range(n):
           i5, i2 = i % 5, i % 2
           for j in range(n):
               j5, j2 = j % 5, j % 2
               r5 = (i5 + j5) % 5
               r2 = (i2 + j2) % 2
               sq2[i][j] = (r5 * 6 + r2 * 5) % 10
               
       # Verify orthogonality just in case
       if not _are_orthogonal(sq1, sq2):
           # Fallback to single square
           return [sq1]
           
       return [sq1, sq2]
   