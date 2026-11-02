for i in range(n):
    for j in range(m):
        for k in range(p):
            val = 0.0
            for c_idx in range(rank):
                val += U_arr[i*m+j, c_idx] * V_arr[j*p+k, c_idx] * W_arr[k*n+i, c_idx]
            if abs(val - 1.0) > max_err:
                max_err = abs(val - 1.0)
