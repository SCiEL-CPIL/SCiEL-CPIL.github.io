val += U_mat[i * m + j][r] * V_mat[j * p + k][r] * W_mat[k * n + i][r]
if abs(val - 1.0) > 1e-10:
    return False, (i, j, k, val)
