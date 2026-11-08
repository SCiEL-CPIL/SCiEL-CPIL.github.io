# M7: (b-d)(u+v)
# b: U[i_a*m + (j_a+1)] -> U[1]
# d: U[(i_a+1)*m + (j_a+1)] -> U[5]
# u: V[(j_b+1)*p + k_b] -> V[8]
# v: V[(j_b+1)*p + (k_b+1)] -> V[9]
# p: W[k_b*n + i_a] -> W[6]
# s: W[(k_b+1)*n + (i_a+1)] -> W[10]

U[i_a*m + (j_a+1)][col] = 1  # U[1, c7] = 1
U[(i_a+1)*m + (j_a+1)][col] = -1  # U[5, c7] = -1
V[(j_b+1)*p + k_b][col] = 1  # V[8, c7] = 1
V[(j_b+1)*p + (k_b+1)][col] = 1  # V[9, c7] = 1
W[k_b*n + i_a][col] = 1  # W[6, c7] = 1
W[(k_b+1)*n + (i_a+1)][col] = 1  # W[10, c7] = 1
