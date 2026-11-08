U_mat[ja][idx] = 1.0          # A[0,ja]
U_mat[m + jb][idx] = 1.0      # A[1,jb]
V_mat[ja * p + ka][idx] = 1.0 # B[ja,ka]
V_mat[jb * p + kb][idx] = 1.0 # B[jb,kb]
