def construct():
    n, m, p = 4, 4, 7
    
    # Strassen decomposition for 2x2x2
    # U, V, W each have shape (4, 7)
    # Index convention: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    # Define the 7 Strassen terms
    # Each term is (U_vec, V_vec, W_vec) of length 4
    strassen_U = [
        [1,0,0,0],  # c=0
        [0,0,0,1],  # c=1
        [0,1,0,0],  # c=2
        [0,0,1,0],  # c=3
        [1,0,0,1],  # c=4
        [1,0,0,1],  # c=5
        [0,0,1,0],  # c=6
    ]
    strassen_V = [
        [1,0,0,0],
        [0,0,0,1],
        [0,0,0,1],
        [1,0,0,0],
        [1,0,1,0],
        [0,0,0,1],
        [1,0,1,0],
    ]
    strassen_W = [
        [1,0,0,0],
        [0,0,0,1],
        [0,1,0,0],
        [0,0,1,0],
        [1,0,0,1],
        [0,0,1,0],
        [1,0,0,1],
    ]
    
    # Verify: for all i,j,k in {0,1}, sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] should be 1
    # ... (verification code)
    
    # Build 4x4x4 via Kronecker product
    # For each pair (c1, c2), the combined term has:
    # U_4[i*4+j, c1*7+c2] = U_2[i1*2+j1, c1] * U_2[i2*2+j2, c2] where i=i1*2+i2, j=j1*2+j2
    # Similarly for V and W
    
    # Build 4x4x3 via schoolbook
    # Total rank = 49 + 48 = 97
