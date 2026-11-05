  for i in range(n):
      for j in range(m):
          for k in range(p):
              total = 0
              for c in range(rank):
                  total += U_mat[i*m+j, c] * V_mat[j*p+k, c] * W_mat[k*n+i, c]
              if total != 1:
                  return False
  