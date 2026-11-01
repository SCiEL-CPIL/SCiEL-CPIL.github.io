"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    # Rank 7 decomposition for 2x2 matrix multiplication
    # Using a different permutation/ordering of Strassen-like terms
    n, m, p = DIMENSIONS
    rank = 7
    
    # Define the 7 rank-1 tensors
    # Each is a tuple (u_vec, v_vec, w_vec)
    # u_vec corresponds to A (4x1 flattened), v_vec to B (4x1 flattened), w_vec to C (4x1 flattened)
    # Indices: A[i,j], B[j,k], C[i,k]
    # Flatten: A[i*m+j], B[j*p+k], C[k*n+i]
    
    # Term 1: (A00+A11)(B00+B11) contributes to C00, C11
    t0_u = [1, 0, 0, 1]
    t0_v = [1, 0, 0, 1]
    t0_w = [1, 0, 0, 1]
    
    # Term 2: (A10+A11)B00 contributes to C10 (+1), C11 (-1)
    t1_u = [0, 0, 1, 1]
    t1_v = [1, 0, 0, 0]
    t1_w = [0, 1, 0, -1]
    
    # Term 3: A00(B01-B11) contributes to C01 (+1), C11 (+1)
    t2_u = [1, 0, 0, 0]
    t2_v = [0, 1, 0, -1]
    t2_w = [0, 0, 1, 1]
    
    # Term 4: A11(B10-B00) contributes to C00 (+1), C10 (+1)
    t3_u = [0, 0, 0, 1]
    t3_v = [-1, 0, 1, 0]
    t3_w = [1, 1, 0, 0]
    
    # Term 5: (A00+A01)B11 contributes to C00 (-1), C01 (+1)
    t4_u = [1, 1, 0, 0]
    t4_v = [0, 0, 0, 1]
    t4_w = [-1, 0, 1, 0]
    
    # Term 6: (A10-A00)(B00+B01) contributes to C11 (+1)
    t5_u = [-1, 0, 1, 0]
    t5_v = [1, 1, 0, 0]
    t5_w = [0, 0, 0, 1]
    
    # Term 7: (A01-A11)(B10+B11) contributes to C00 (+1)
    t6_u = [0, 1, 0, -1]
    t6_v = [0, 0, 1, 1]
    t6_w = [1, 0, 0, 0]
    
    # Construct U, V, W matrices of shape (4, 7)
    U = [[t0_u[i], t1_u[i], t2_u[i], t3_u[i], t4_u[i], t5_u[i], t6_u[i]] for i in range(4)]
    V = [[t0_v[i], t1_v[i], t2_v[i], t3_v[i], t4_v[i], t5_v[i], t6_v[i]] for i in range(4)]
    W = [[t0_w[i], t1_w[i], t2_w[i], t3_w[i], t4_w[i], t5_w[i], t6_w[i]] for i in range(4)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
