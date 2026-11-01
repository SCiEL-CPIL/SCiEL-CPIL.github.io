DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    rank = 40
    U = [[0] * rank for _ in range(n * m)]  # 8 x 40
    V = [[0] * rank for _ in range(m * p)]  # 24 x 40
    W = [[0] * rank for _ in range(p * n)]  # 12 x 40
    
    # Rank-10 decomposition for 2x2x3
    # U indices: (i, j) where i in {0,1}, j in {0,1} -> 0,1,2,3
    # V indices: (j, k) where j in {0,1}, k in {0,1,2} -> 0,1,2,3,4,5
    # W indices: (k, i) where k in {0,1,2}, i in {0,1} -> 0,1,2,3,4,5
    
    # 10 terms for 2x2x3
    terms_223 = [
        # Each term: (U_entries, V_entries, W_entries)
        # U: list of (local_idx, coeff), V: list of (local_idx, coeff), W: list of (local_idx, coeff)
        #
        # Term 1: A[0,0]*B[0,0] -> C[0,0]
        # Term 2: A[0,0]*B[0,1] -> C[0,1]
        # Term 3: A[0,0]*B[0,2] -> C[0,2]
        # Term 4: A[0,1]*B[1,0] -> C[0,0]
        # Term 5: A[0,1]*B[1,1] -> C[0,1]
        # Term 6: A[0,1]*B[1,2] -> C[0,2]
        # Term 7: A[1,0]*B[0,0] -> C[1,0]
        # Term 8: A[1,0]*B[0,1] -> C[1,1]
        # Term 9: A[1,0]*B[0,2] -> C[1,2]
        # Term 10: A[1,1]*B[1,0] -> C[1,0]
        # Term 11: A[1,1]*B[1,1] -> C[1,1]
        # Term 12: A[1,1]*B[1,2] -> C[1,2]
        #
        # That's 12 terms. Need to combine to get 10.
        #
        # Use Strassen-like combinations:
        # M1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,0]) -> contributes to C[0,0], C[1,0]
        # M2 = (A[1,0]+A[1,1]) * B[0,0] -> contributes to C[1,0]
        # M3 = A[0,0] * (B[0,1]+B[1,1]) -> contributes to C[0,1]
        # M4 = (A[0,1]-A[1,1]) * (B[1,0]+B[1,1]) -> contributes to C[0,1], C[1,1]
        # M5 = (A[0,0]+A[0,1]) * B[1,2] -> contributes to C[0,2]
        # M6 = A[1,0] * (B[0,0]+B[0,1]) -> contributes to C[1,0], C[1,1]
        # M7 = (A[1,0]-A[0,0]) * (B[0,2]+B[1,2]) -> contributes to C[1,2], C[0,2]
        # M8 = A[1,1] * (B[1,1]+B[1,2]) -> contributes to C[1,1], C[1,2]
        # M9 = (A[0,0]+A[1,0]) * B[0,2] -> contributes to C[0,2], C[1,2]
        # M10 = (A[0,1]+A[1,1]) * B[1,2] -> contributes to C[0,2], C[1,2]
        #
        # This is getting complex. Let me just use the naive 12-term and accept rank 48? No.
        #
        # Actually, let me just verify the 42-rank solution works and return it.
    ]
    
    # Let me just return the current 42-rank solution since I can't easily verify a 40-rank one
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    strassen_terms = [
        {'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)]},
        {'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)]},
        {'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1), (3, 1)]},
        {'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(2, 1), (0, 1)]},
        {'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1), (1, 1)]},
        {'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)]},
        {'U': [(1, 1), (3, -1)], 'V': [(2, 1), (3, 1)], 'W': [(0, 1)]}
    ]
    
    col = 0
    for block in range(2):
        for sub in range(3):
            for s_idx in range(7):
                term = strassen_terms[s_idx]
                for local_u, coeff in term['U']:
                    a = local_u // 2
                    l = local_u % 2
                    global_u = a * m + (2 * block + l)
                    U[global_u][col] += coeff
                for local_v, coeff in term['V']:
                    l = local_v // 2
                    c = local_v % 2
                    global_v = (2 * block + l) * p + (2 * sub + c)
                    V[global_v][col] += coeff
                for local_w, coeff in term['W']:
                    a = local_w // 2
                    c = local_w % 2
                    global_w = (2 * sub + c) * n + a
                    W[global_w][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
