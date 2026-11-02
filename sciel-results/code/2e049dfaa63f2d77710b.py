import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 recursive Strassen decomposition for 4x4 matrix multiplication.
    Directly constructs the 49 rank-1 tensors by explicit enumeration.
    
    Convention:
      C[i,j] = sum_k A[i,k] * B[k,j]
      U index: i*4 + k  (A row i, A col k)
      V index: k*4 + j  (B row k, B col j)
      W index: j*4 + i  (C col j, C row i)  [trace convention]
    """
    t0 = time.time()
    deadline = 140.0

    n = m = p = 4
    rank = 49

    U = np.zeros((16, rank), dtype=np.float64)
    V = np.zeros((16, rank), dtype=np.float64)
    W = np.zeros((16, rank), dtype=np.float64)

    # Recursive Strassen: 4x4 = 2x2 of 2x2 blocks
    # Outer level: 7 Strassen terms, each expanding to 7 inner terms = 49 total
    
    # 2x2 Strassen: for matrices A, B (2x2), compute C = A*B with 7 products
    # Indexing: A[row, col], B[row, col], C[row, col]
    # U2 index: row*2+col, V2 index: row*2+col, W2 index: col*2+row (trace convention)
    
    # Define the 7 Strassen terms for 2x2:
    # Each term: (coeffs on A, coeffs on B, coeffs on C)
    # A coeffs: dict from (row,col) to coeff
    # B coeffs: dict from (row,col) to coeff  
    # C coeffs: dict from (row,col) to coeff
    
    strassen_2x2 = [
        # t0: (A00+A11)*(B00+B11) -> C00+=1, C11+=1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # t1: (A10+A11)*B00 -> C10+=1, C11+=-1
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1, (1,1):-1}),
        # t2: A00*(B01-B11) -> C01+=1, C11+=1
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # t3: A11*(B10-B00) -> C00+=1, C10+=1
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # t4: (A00+A01)*B11 -> C00+=-1, C01+=1
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # t5: (A10-A00)*(B00+B01) -> C11+=1
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # t6: (A01-A11)*(B10+B11) -> C00+=1
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1}),
    ]

    # For 4x4: outer Strassen operates on 2x2 blocks of 2x2 submatrices
    # Outer A block (br, bc) contains A[2*br:2*br+2, 2*bc:2*bc+2]
    # For each outer term t and inner term k, we get one rank-1 tensor
    
    for t in range(7):
        outer_A, outer_B, outer_C = strassen_2x2[t]
        for k in range(7):
            inner_A, inner_B, inner_C = strassen_2x2[k]
            idx = t * 7 + k
            
            # Fill U: A element at global (i, ak)
            # Global A[i, ak] is in block (i//2, ak//2) at local position (i%2, ak%2)
            for (br, bc), oc in outer_A.items():
                for (lr, lc), ic in inner_A.items():
                    i = 2 * br + lr
                    ak = 2 * bc + lc
                    U[i*4 + ak, idx] = oc * ic
            
            # Fill V: B element at global (bk, j)
            # Global B[bk, j] is in block (bk//2, j//2) at local position (bk%2, j%2)
            for (br, bc), oc in outer_B.items():
                for (lr, lc), ic in inner_B.items():
                    bk = 2 * br + lr
                    j = 2 * bc + lc
                    V[bk*4 + j, idx] = oc * ic
            
            # Fill W: C element at global (ci, cj) -> W index is cj*4+ci (trace convention)
            # Global C[ci, cj] is in block (ci//2, cj//2) at local position (ci%2, cj%2)
            for (br, bc), oc in outer_C.items():
                for (lr, lc), ic in inner_C.items():
                    ci = 2 * br + lr
                    cj = 2 * bc + lc
                    W[cj*4 + ci, idx] = oc * ic

    # Verify the decomposition using vectorized einsum
    # U_reshaped: (4, 4, rank) -> U[i, k, r]
    # V_reshaped: (4, 4, rank) -> V[k, j, r]
    # W_reshaped: (4, 4, rank) -> W[j, i, r] (since W index is j*4+i)
    U_r = U.reshape(4, 4, rank)
    V_r = V.reshape(4, 4, rank)
    W_r = W.reshape(4, 4, rank)
    
    # Compute T[i, j, k] = sum_r U[i, k, r] * V[k, j, r] * W[j, i, r]
    # Note: The target tensor for matrix multiplication is T[i, j, k] = 1 for all i, j, k.
    # The indices in the sum are:
    # U depends on i, k
    # V depends on k, j
    # W depends on j, i
    # So we sum over r.
    # Result should be a 4x4x4 tensor of ones.
    
    # np.einsum('ikr,kjr,jir->ijk', U_r, V_r, W_r)
    # Let's check the indices carefully.
    # U_r has shape (4, 4, rank). Let's call indices (i, k, r).
    # V_r has shape (4, 4, rank). Let's call indices (k, j, r).
    # W_r has shape (4, 4, rank). Let's call indices (j, i, r).
    # We want to sum over r.
    # The output should be indexed by i, j, k.
    # 'ikr,kjr,jir->ijk'
    
    T = np.einsum('ikr,kjr,jir->ijk', U_r, V_r, W_r)
    
    max_err = np.max(np.abs(T - 1.0))
    
    if max_err > 1e-8:
        # Fallback to schoolbook rank-64
        U64 = np.zeros((16, 64), dtype=np.float64)
        V64 = np.zeros((16, 64), dtype=np.float64)
        W64 = np.zeros((16, 64), dtype=np.float64)
        idx = 0
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    U64[i*4+k, idx] = 1.0
                    V64[k*4+j, idx] = 1.0
                    W64[j*4+i, idx] = 1.0
                    idx += 1
        return {"U": U64.tolist(), "V": V64.tolist(), "W": W64.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
