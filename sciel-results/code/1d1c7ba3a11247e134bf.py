import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    deadline = time.time() + 110  # safety margin before 120s limit
    
    # Strategy: Strassen for 4x4x4 (rank 49) + schoolbook for 4x4x3 (rank 48) = 97
    rank = 49 + 4 * 4 * 3  # 97
    
    U = [[0] * rank for _ in range(n * m)]   # 16 x 97
    V = [[0] * rank for _ in range(m * p)]   # 28 x 97
    W = [[0] * rank for _ in range(p * n)]   # 28 x 97
    
    col = 0
    
    # Part 1: Strassen 4x4x4 for first 4 columns of B
    # A is 4x4, B1 is 4x4 (first 4 cols of B), C1 is 4x4 (first 4 cols of C)
    # Partition into 2x2 blocks of 2x2 matrices
    # A = [[A11, A12], [A21, A22]], B1 = [[B11, B12], [B21, B22]], C1 = [[C11, C12], [C21, C22]]
    # Strassen terms at block level:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # Each block multiplication is 2x2x2, using Strassen rank 7
    # Total: 7 * 7 = 49
    
    # For each block-level Strassen term, we need to decompose the 2x2x2 multiplication
    # using Strassen's 7-term algorithm for the inner 2x2 matrices
    
    # Indexing conventions:
    # U: A[i,j] -> index i*m + j = i*4 + j
    # V: B[j,k] -> index j*p + k = j*7 + k (for first 4 cols: j*7 + k where k=0..3)
    # W: C[k,i] -> index k*n + i = k*4 + i (for first 4 cols: k*4 + i where k=0..3)
    
    # For a 2x2x2 Strassen decomposition with local indices:
    # A block: rows a0,a1; cols b0,b1 -> U indices: a0*4+b0, a0*4+b1, a1*4+b0, a1*4+b1
    # B block: rows c0,c1; cols d0,d1 -> V indices: c0*7+d0, c0*7+d1, c1*7+d0, c1*7+d1
    # C block: rows e0,e1; cols f0,f1 -> W indices: e0*4+f0, e0*4+f1, e1*4+f0, e1*4+f1
    
    # Strassen 2x2x2 terms (local indices 0,1,2,3 for A, B, C respectively):
    # A: 0=a00, 1=a01, 2=a10, 3=a11
    # B: 0=b00, 1=b01, 2=b10, 3=b11
    # C: 0=c00, 1=c01, 2=c10, 3=c11
    # M1 = (a00+a11)(b00+b11): U=[1,0,1,0], V=[1,0,0,1], W=[1,0,0,1]
    # M2 = (a10+a11)b00: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,-1]
    # M3 = a00(b01-b11): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,0]
    # M4 = a11(b10-b00): U=[0,0,0,1], V=[0,0,1,-1], W=[1,0,1,0]
    # M5 = (a00+a01)b11: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    # M6 = (a10-a00)(b00+b01): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    # M7 = (a01-a11)(b10+b11): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,0]
    
    strassen_2x2 = [
        ([1,0,1,0], [1,0,0,1], [1,0,0,1]),   # M1
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),   # M3
        ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # M5
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # M6
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # M7
    ]
    
    # Block-level Strassen coefficients for 4x4x4:
    # M1 = (A11+A22)*(B11+B22) -> contributes to C11 with +1, C22 with +1
    # M2 = (A21+A22)*B11 -> contributes to C21 with +1, C22 with -1
    # M3 = A11*(B12-B22) -> contributes to C12 with +1, C22 with +1
    # M4 = A22*(B21-B11) -> contributes to C11 with +1, C21 with +1
    # M5 = (A11+A12)*B22 -> contributes to C11 with -1, C12 with +1
    # M6 = (A21-A11)*(B11+B12) -> contributes to C22 with +1
    # M7 = (A12-A22)*(B21+B22) -> contributes to C11 with +1
    
    # For each block-level term, the L and R are linear combinations of 2x2 blocks
    # We need to expand each 2x2x2 Strassen term with the appropriate linear combinations
    
    # Block positions:
    # A11: rows 0-1, cols 0-1
    # A12: rows 0-1, cols 2-3
    # A21: rows 2-3, cols 0-1
    # A22: rows 2-3, cols 2-3
    
    # B11: rows 0-1, cols 0-1
    # B12: rows 0-1, cols 2-3
    # B21: rows 2-3, cols 0-1
    # B22: rows 2-3, cols 2-3
    
    # C11: rows 0-1, cols 0-1
    # C12: rows 0-1, cols 2-3
    # C21: rows 2-3, cols 0-1
    # C22: rows 2-3, cols 2-3
    
    # For a block-level term like (A11+A22)*(B11+B22):
    # The L matrix is A11+A22 (2x2), R matrix is B11+B22 (2x2)
    # When we decompose this 2x2x2 with Strassen, each inner term has
    # U coefficients that are sums of entries from A11 and A22
    
    # Let me define helper to add a 2x2x2 Strassen term with given U,V,W coefficient vectors
    # where U,V,W are length-4 vectors for the local 2x2 blocks
    
    def add_block_term(u_coeff, v_coeff, w_coeff, sign=1):
        """
        u_coeff: list of 4 values for A block entries [a00, a01, a10, a11]
                 These are linear combination coefficients
        v_coeff: list of 4 values for B block entries [b00, b01, b10, b11]
        w_coeff: list of 4 values for C block entries [c00, c01, c10, c11]
        sign: overall sign multiplier
        """
        nonlocal col
        for t in range(7):
            if time.time() > deadline:
                return col
            su, sv, sw = strassen_2x2[t]
            # For each inner Strassen term, the actual U/V/W entries are:
            # U entry for A position p: sum over local A index la where su[la]!=0 of u_coeff[la]*su[la]
            # Actually, u_coeff gives the linear combination for each local A entry
            # The inner Strassen term su gives coefficients for local A entries
            # So the combined coefficient for A local entry la is u_coeff[la] * su[la]
            
            # U: for each local A entry la (0..3), the global U index depends on which block
            # But we need to know which block this term corresponds to
            
            # This approach is getting complex. Let me use a different method.
            # I'll directly compute the rank-1 terms.
            pass
        return col
    
    # Alternative approach: directly enumerate all rank-1 terms
    # For the 4x4x4 Strassen part, there are 49 rank-1 terms
    # Each term is a product of a linear form in A, a linear form in B1, and a linear form in C1
    
    # Let me build this more carefully by constructing the full decomposition matrix
    
    # Actually, let me use a cleaner approach: build the 4x4x4 Strassen decomposition
    # by tensor product of two 2x2x2 Strassen decompositions
    
    # The 4x4x4 matmul tensor T_{i,j,k} = sum_{a,b} delta_{i,a+b} delta_{j,a+c} delta_{k,b+c}
    # where a,b,c are in 0..1 for each "digit" (since 4 = 2*2)
    
    # Using Kronecker product structure:
    # T_4x4x4 = T_2x2x2 ⊗ T_2x2x2 (in appropriate sense)
    
    # If we have Strassen decomposition for 2x2x2: T = sum_r u_r ⊗ v_r ⊗ w_r
    # Then for 4x4x4: T = sum_{r1,r2} (u_r1 ⊗ u_r2) ⊗ (v_r1 ⊗ v_r2) ⊗ (w_r1 ⊗ w_r2)
    
    # This gives rank 7*7 = 49
    
    # Let me implement this tensor product construction
    
    # 2x2x2 Strassen:
    # U2: 4x7, V2: 4x7, W2: 4x7
    U2, V2, W2 = [], [], []
    for t in range(7):
        U2.append(strassen_2x2[t][0])
        V2.append(strassen_2x2[t][1])
        W2.append(strassen_2x2[t][2])
    U2 = np.array(U2, dtype=float).T  # 4x7
    V2 = np.array(V2, dtype=float).T  # 4x7
    W2 = np.array(W2, dtype=float).T  # 4x7
    
    # Tensor product for 4x4x4:
    # U4[i1*2+i2, j1*2+j2, r1*7+r2] = U2[i1,j1,r1] * U2[i2,j2,r2]
    # Wait, the indexing is: A[i,j] where i,j in 0..3
    # i = i1*2 + i2, j = j1*2 + j2 where i1,j1 in 0..1 (high bit), i2,j2 in 0..1 (low bit)
    # Actually for Kronecker: A[i,j] with i = i1 + 2*i2? Let me think...
    
    # For 4x4 = 2x2 block structure:
    # A = [[A11, A12], [A21, A22]]
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # So row index i: high bit i1 = i // 2, low bit i2 = i % 2
    # Col index j: high bit j1 = j // 2, low bit j2 = j % 2
    
    # U4[(i1*2+i2)*4 + (j1*2+j2), r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # Wait, that's not right either. Let me reconsider.
    
    # The Kronecker product of two 2x2x2 tensors:
    # T4[i,j,k] = T2[i1,j1,k1] * T2[i2,j2,k2] where i=i1*2+i2, j=j1*2+j2, k=k1*2+k2
    # This is the tensor product (Kronecker) structure.
    
    # For the decomposition:
    # T4 = sum_{r1,r2} (u_r1 ⊗ u_r2) ⊗ (v_r1 ⊗ v_r2) ⊗ (w_r1 ⊗ w_r2)
    # where u_r1 ⊗ u_r2 has entries (u_r1 ⊗ u_r2)[i,j] = u_r1[i1]*u_r2[i2] * u_r1[j1]*u_r2[j2]
    # Wait no, u is a vector of length 4 (for 2x2 matrix entries).
    
    # Let me be more careful. For 2x2x2:
    # U2 has shape (4, 7): row index is (i,j) flattened as i*2+j
    # V2 has shape (4, 7): row index is (j,k) flattened as j*2+k  
    # W2 has shape (4, 7): row index is (k,i) flattened as k*2+i
    
    # For 4x4x4:
    # U4 has shape (16, 49): row index is (i,j) flattened as i*4+j
    # V4 has shape (16, 49): row index is (j,k) flattened as j*4+k
    # W4 has shape (16, 49): row index is (k,i) flattened as k*4+i
    
    # Kronecker: U4[i*4+j, r1*7+r2] = U2[i1*2+j1, r1] * U2[i2*2+j2, r2]
    # where i = i1*2+i2, j = j1*2+j2
    
    # Let's verify: for i=0,j=0: i1=0,i2=0,j1=0,j2=0
    # U4[0, r1*7+r2] = U2[0, r1] * U2[0, r2]
    # This corresponds to A[0,0] which is in block A11, position (0,0)
    
    # This looks correct!
    
    # Build U4, V4, W4 for 4x4x4
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    i = i1 * 2 + i2
                    j = j1 * 2 + j2
                    idx = i * 4 + j
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            U4[idx, r] = U2[i1 * 2 + j1, r1] * U2[i2 * 2 + j2, r2]
    
    for j1 in range(2):
        for j2 in range(2):
            for k1 in range(2):
                for k2 in range(2):
                    j = j1 * 2 + j2
                    k = k1 * 2 + k2
                    idx = j * 4 + k
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            V4[idx, r] = V2[j1 * 2 + k1, r1] * V2[j2 * 2 + k2, r2]
    
    for k1 in range(2):
        for k2 in range(2):
            for i1 in range(2):
                for i2 in range(2):
                    k = k1 * 2 + k2
                    i = i1 * 2 + i2
                    idx = k * 4 + i
                    for r1 in range(7):
                        for r2 in range(7):
                            r = r1 * 7 + r2
                            W4[idx, r] = W2[k1 * 2 + i1, r1] * W2[k2 * 2 + i2, r2]
    
    # Now fill in the first 49 columns of U, V, W
    for i in range(16):
        for r in range(49):
            U[i][col + r] = U4[i, r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 7 + k][col + r] = V4[j * 4 + k, r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][col + r] = W4[k * 4 + i, r]
    
    col += 49
    
    # Part 2: Schoolbook for 4x4x3 (last 3 columns of B)
    # A is 4x4, B2 is 4x3 (cols 4-6 of B), C2 is 4x3 (cols 4-6 of C)
    # Schoolbook: rank = 4*4*3 = 48
    for i in range(n):
        for j in range(m):
            for k in range(4, 7):  # columns 4,5,6 of B
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Verify dimensions
    assert len(U) == 16
    assert len(V) == 28
    assert len(W) == 28
    assert len(U[0]) == rank
    assert len(V[0]) == rank
    assert len(W[0]) == rank
    
    return {"U": U, "V": V, "W": W}
