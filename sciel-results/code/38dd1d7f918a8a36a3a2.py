   def verify(U, V, W):
       """Verify that the decomposition is exact."""
       r = len(U[0]) if U else 0
       # Check all (i,j,k) triples
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   lhs = 1  # T[i,j,k] = 1
                   rhs = 0
                   for c in range(r):
                       rhs += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                   if abs(lhs - rhs) > 1e-9:
                       return False
       return True
   