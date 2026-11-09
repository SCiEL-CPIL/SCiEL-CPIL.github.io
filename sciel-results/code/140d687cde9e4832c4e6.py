def construct():
    n, m, p = 2, 4, 5
    deadline = time.time() + 110
    
    def decomp_2x2x5_rank10():
        # ... as above
    
    U1, V1, W1 = decomp_2x2x5_rank10()
    U2, V2, W2 = decomp_2x2x5_rank10()
    
    rank = 20
    U_out = np.zeros((2*4, rank), dtype=object)
    V_out = np.zeros((4*5, rank), dtype=object)
    W_out = np.zeros((5*2, rank), dtype=object)
    
    # Block 1: j in {0,1}
    for r in range(10):
        for ij in range(4):
            i = ij // 2
            j_local = ij % 2
            j = j_local
            if U1[ij, r] != 0:
                U_out[i*4+j, r] += U1[ij, r]
        for jk in range(10):
            j_local = jk // 5
            k = jk % 5
            j = j_local
            if V1[jk, r] != 0:
                V_out[j*5+k, r] += V1[jk, r]
        for ki in range(10):
            k = ki // 2
            i = ki % 2
            if W1[ki, r] != 0:
                W_out[k*2+i, r] += W1[ki, r]
    
    # Block 2: j in {2,3}
    for r in range(10):
        r2 = 10 + r
        for ij in range(4):
            i = ij // 2
            j_local = ij % 2
            j = j_local + 2
            if U2[ij, r] != 0:
                U_out[i*4+j, r2] += U2[ij, r]
        for jk in range(10):
            j_local = jk // 5
            k = jk % 5
            j = j_local + 2
            if V2[jk, r] != 0:
                V_out[j*5+k, r2] += V2[jk, r]
        for ki in range(10):
            k = ki // 2
            i = ki % 2
            if W2[ki, r] != 0:
                W_out[k*2+i, r2] += W2[ki, r]
    
    # Verify
    def verify_decomposition(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = Fraction(0)
                    for r in range(rank):
                        total += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if total != 1:
                        return False
        return True
    
    if not verify_decomposition(U_out, V_out, W_out, 20, n, m, p):
        # Fallback to trivial rank-40
        U_out = np.zeros((8, 40), dtype=object)
        V_out = np.zeros((20, 40), dtype=object)
        W_out = np.zeros((10, 40), dtype=object)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    U_out[i * m + j, idx] = Fraction(1)
                    V_out[j * p + k, idx] = Fraction(1)
                    W_out[k * n + i, idx] = Fraction(1)
    
    def convert_to_list(arr):
        res = []
        for row in arr:
            res_row = []
            for val in row:
                if isinstance(val, Fraction):
                    if val.denominator == 1:
                        res_row.append(int(val))
                    elif val.denominator == 2:
                        res_row.append(val.numerator / 2.0)
                    else:
                        res_row.append(float(val))
                else:
                    res_row.append(val)
            res.append(res_row)
        return res
    
    return {"U": convert_to_list(U_out), "V": convert_to_list(V_out), "W": convert_to_list(W_out)}
