         def bionj_distance(D, m):
             n = len(D)
             # D is n x n matrix
             # Compute BionJ corrected distances
             # d_ij = D_ij - (1/m) * sum_{k != i,j} 0.5 * (D_ik + D_jk - D_ij)
             # This can be vectorized.
             # Sum_{k != i,j} D_ik = row_sum[i] - D_ij
             # Sum_{k != i,j} D_jk = col_sum[j] - D_ij
             # Sum_{k != i,j} D_ij = (n-2) * D_ij
             # Sum = 0.5 * (row_sum[i] + col_sum[j] - 2*D_ij - (n-2)*D_ij)
             # Sum = 0.5 * (row_sum[i] + col_sum[j] - (n) * D_ij)
             # Correction = Sum / m
             # d_ij = D_ij - Correction
             pass
         