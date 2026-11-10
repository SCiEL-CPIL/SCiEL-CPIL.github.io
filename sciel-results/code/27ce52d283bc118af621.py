    for i in range(j):
        lo = int(lo_indices[i])
        if lo < i:
            max_prev = S_val[i][lo]
            dp[i, j] = 2 + max_prev
            amax[i, j] = S_arg[i][lo]
        else:
            dp[i, j] = 2
            amax[i, j] = -1
    