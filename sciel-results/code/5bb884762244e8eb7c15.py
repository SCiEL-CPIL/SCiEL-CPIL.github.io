import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Total rank estimate
    # Block 1: 4x4x4 (Strassen) -> 49
    # Block 2: 4x1x4 (Schoolbook) -> 16
    # Block 3: 4x4x2 (Schoolbook) -> 32
    # Block 4: 4x1x2 (Schoolbook) -> 8
    # Total: 105
    
    rank = 105
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add a schoolbook decomposition for a sub-block
    # i_start, i_end, j_start, j_end, k_start, k_end
    def add_schoolbook(i_s, i_e, j_s, j_e, k_s, k_e):
        nonlocal col_offset
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U[u_idx][col_offset] = 1
                    V[v_idx][col_offset] = 1
                    W[w_idx][col_offset] = 1
                    col_offset += 1
                    
    # Helper to add Strassen decomposition for 4x4x4 block
    # This is complex to generate directly. 
    # We can generate the 2x2x2 Strassen factors and then compose them.
    
    # Let's generate the 2x2x2 Strassen factors first.
    # The 7 terms correspond to:
    # M1: (11,11), (11,11), (11,11) ... wait, the indices in the tensor are (i,j,k).
    # For 2x2x2:
    # i in 0,1; j in 0,1; k in 0,1.
    
    # Strassen factors for 2x2:
    # Each factor is a triple of vectors (u, v, w).
    # u is size 4 (2x2), v is size 4, w is size 4.
    
    s2_factors = []
    
    # M1 = (A11+A22)(B11+B22)
    # A part: 1 at (0,0), 1 at (1,1)
    # B part: 1 at (0,0), 1 at (1,1)
    # C part: 1 at (0,0), 1 at (1,1) ... wait.
    # The term is added to C11 and C22.
    # C11 += M1, C22 += M1.
    # So W vector has 1 at (0,0) and 1 at (1,1)?
    # No, W is for C.
    # The decomposition is sum U_r V_r W_r.
    # For M1:
    # U1: 1 at (0,0), 1 at (1,1). (A11, A22)
    # V1: 1 at (0,0), 1 at (1,1). (B11, B22)
    # W1: 1 at (0,0), 1 at (1,1). (C11, C22)
    # This contributes A11 B11 C11 + A11 B22 C11? No.
    # (A11+A22)(B11+B22) = A11 B11 + A11 B22 + A22 B11 + A22 B22.
    # We want to assign this to C11 and C22.
    # C11 gets M1. C22 gets M1.
    # So the term is M1 * (C11 + C22)?
    # The tensor term is U \otimes V \otimes W.
    # If W has 1 at (0,0) and 1 at (1,1), then the contribution is M1 * C11 + M1 * C22.
    # This matches C11 += M1 and C22 += M1.
    
    def make_vec(indices, size):
        v = [0] * size
        for idx in indices:
            v[idx] = 1
        return v
        
    # Indices for 2x2:
    # A: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # B: same
    # C: same
    
    # M1: A(0,3), B(0,3), C(0,3)
    s2_factors.append((make_vec([0,3], 4), make_vec([0,3], 4), make_vec([0,3], 4)))
    
    # M2: (A21+A22) B11
    # A: 1 at (1,0)=2, 1 at (1,1)=3.
    # B: 1 at (0,0)=0.
    # C: C21 += M2. So W: 1 at (1,0)=2.
    s2_factors.append((make_vec([2,3], 4), make_vec([0], 4), make_vec([2], 4)))
    
    # M3: A11 (B12-B22)
    # A: 1 at (0,0)=0.
    # B: 1 at (0,1)=1, -1 at (1,1)=3.
    # C: C12 += M3. So W: 1 at (0,1)=1.
    v3 = [0]*4; v3[1]=1; v3[3]=-1
    s2_factors.append((make_vec([0], 4), v3, make_vec([1], 4)))
    
    # M4: A22 (B21-B11)
    # A: 1 at (1,1)=3.
    # B: 1 at (1,0)=2, -1 at (0,0)=0.
    # C: C11 += M4, C21 += M4.
    # W: 1 at (0,0)=0, 1 at (1,0)=2.
    v4 = [0]*4; v4[2]=1; v4[0]=-1
    s2_factors.append((make_vec([3], 4), v4, make_vec([0,2], 4)))
    
    # M5: (A11+A12) B22
    # A: 1 at (0,0)=0, 1 at (0,1)=1.
    # B: 1 at (1,1)=3.
    # C: C11 -= M5, C12 += M5.
    # W: -1 at (0,0)=0, 1 at (0,1)=1.
    w5 = [0]*4; w5[0]=-1; w5[1]=1
    s2_factors.append((make_vec([0,1], 4), make_vec([3], 4), w5))
    
    # M6: (A21-A11) (B11+B12)
    # A: 1 at (1,0)=2, -1 at (0,0)=0.
    # B: 1 at (0,0)=0, 1 at (0,1)=1.
    # C: C22 += M6.
    # W: 1 at (1,1)=3.
    u6 = [0]*4; u6[2]=1; u6[0]=-1
    s2_factors.append((u6, make_vec([0,1], 4), make_vec([3], 4)))
    
    # M7: (A12-A22) (B21+B22)
    # A: 1 at (0,1)=1, -1 at (1,1)=3.
    # B: 1 at (1,0)=2, 1 at (1,1)=3.
    # C: C11 += M7.
    # W: 1 at (0,0)=0.
    u7 = [0]*4; u7[1]=1; u7[3]=-1
    s2_factors.append((u7, make_vec([2,3], 4), make_vec([0], 4)))
    
    # Now compose for 4x4x4.
    # The 4x4 matrix is divided into 2x2 blocks.
    # Block (0,0) of A is A[0:2, 0:2].
    # We apply the 7 factors to each of the 7 block positions?
    # No, the rank is 7*7=49.
    # The factors are Kronecker products of the 2x2 factors.
    
    # We need to map the 2x2 indices to 4x4 indices.
    # A 4x4 index (i,j) where i,j in 0..3.
    # Block row bi = i//2, block col bj = j//2.
    # Local row li = i%2, local col lj = j%2.
    
    # For each pair of 2x2 factors (r1, r2), we create a 4x4 factor.
    # U_4x4 is size 16.
    # V_4x4 is size 16.
    # W_4x4 is size 16.
    
    for r1 in range(7):
        for r2 in range(7):
            u1, v1, w1 = s2_factors[r1]
            u2, v2, w2 = s2_factors[r2]
            
            u4 = [0] * 16
            v4 = [0] * 16
            w4 = [0] * 16
            
            # Construct Kronecker product
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # Index in 4x4:
                            # i = i1*2 + i2
                            # j = j1*2 + j2
                            # idx = i*4 + j
                            idx = (i1*2 + i2)*4 + (j1*2 + j2)
                            u4[idx] = u1[i1*2+j1] * u2[i2*2+j2]
                            v4[idx] = v1[i1*2+j1] * v2[i2*2+j2]
                            w4[idx] = w1[i1*2+j1] * w2[i2*2+j2]
                            
            # Add to global U, V, W
            # This block corresponds to i in 0..3, j in 0..3, k in 0..3.
            # Global U index: i * 5 + j
            # Global V index: j * 6 + k
            # Global W index: k * 4 + i
            
            for i in range(4):
                for j in range(4):
                    u_idx = i * m + j
                    U[u_idx][col_offset] = u4[i*4+j]
                    
            for j in range(4):
                for k in range(4):
                    v_idx = j * p + k
                    V[v_idx][col_offset] = v4[j*4+k]
                    
            for k in range(4):
                for i in range(4):
                    w_idx = k * n + i
                    W[w_idx][col_offset] = w4[k*4+i]
                    
            col_offset += 1
            
    # Block 2: 4x1x4 (i:0-3, j:4, k:0-3)
    add_schoolbook(0, 4, 4, 5, 0, 4)
    
    # Block 3: 4x4x2 (i:0-3, j:0-3, k:4-5)
    add_schoolbook(0, 4, 0, 4, 4, 6)
    
    # Block 4: 4x1x2 (i:0-3, j:4, k:4-5)
    add_schoolbook(0, 4, 4, 5, 4, 6)
    
    return {"U": U, "V": V, "W": W}
