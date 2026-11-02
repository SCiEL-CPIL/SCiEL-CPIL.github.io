<<<<<<< SEARCH
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    def idx2(i, j):
        return i * 2 + j
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            # U4: 4x4 matrix entries for input A
            for I in range(4):
                for J in range(4):
                    r, c = I // 2, J // 2
                    i_loc, j_loc = I % 2, J % 2
                    coeff = U2[s, idx2(r, c)] * U2[t, idx2(i_loc, j_loc)]
                    U4[I * 4 + J, col] = coeff
            
            # V4: 4x4 matrix entries for input B
            for J in range(4):
                for K in range(4):
                    r, c = J // 2, K // 2
                    j_loc, k_loc = J % 2, K % 2
                    coeff = V2[s, idx2(r, c)] * V2[t, idx2(j_loc, k_loc)]
                    V4[J * 4 + K, col] = coeff
            
            # W4: 4x4 matrix entries for output C[I,K]
            for I in range(4):
                for K in range(4):
                    r, c = I // 2, K // 2
                    i_loc, k_loc = I % 2, K % 2
                    coeff = W2[s, idx2(r, c)] * W2[t, idx2(i_loc, k_loc)]
                    W4[I * 4 + K, col] = coeff
=======
    # Build 4x4x4 rank-49 decomposition via Kronecker product
    # U4[i, col] = U2[s, i] * U2[t, j] where col = s*7+t, i = I*4+J? 
    # Actually, U4 is (16, 49).
    # U2 is (7, 4).
    # We want U4[I*4+J, s*7+t] = U2[s, I] * U2[t, J]
    # This is np.kron(U2.T, U2.T)
    U4 = np.kron(U2.T, U2.T)
    V4 = np.kron(V2.T, V2.T)
    W4 = np.kron(W2.T, W2.T)
>>>>>>> REPLACE
