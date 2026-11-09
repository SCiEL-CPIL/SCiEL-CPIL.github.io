"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _schoolbook_block(n, m, p):
    """
    Returns U, V, W for a schoolbook decomposition of n x m * m x p.
    U shape: (n*m, n*m*p)
    V shape: (m*p, n*m*p)
    W shape: (p*n, n*m*p)
    """
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2_block():
    """
    Returns U, V, W for Strassen's 2x2x2 decomposition.
    Rank 7.
    U shape: (4, 7), V shape: (4, 7), W shape: (4, 7)
    Indices:
    U[i*2+j], V[j*2+k], W[k*2+i]
    """
    rank = 7
    U = [[0] * rank for _ in range(4)]
    V = [[0] * rank for _ in range(4)]
    W = [[0] * rank for _ in range(4)]
    
    # A = [[a, b], [c, d]] -> indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B = [[e, f], [g, h]] -> indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C = [[C00, C01], [C10, C11]] -> W[k*2+i]:
    # C00 -> k=0,i=0 -> idx 0
    # C10 -> k=0,i=1 -> idx 1
    # C01 -> k=1,i=0 -> idx 2
    # C11 -> k=1,i=1 -> idx 3
    
    # M1 = (a+d)(e+h)
    # U: a(0), d(3) -> [1,0,0,1]
    # V: e(0), h(3) -> [1,0,0,1]
    # W: C00(0), C11(3) -> [1,0,0,1] (C11 is at k=1,i=1 -> idx 3)
    # Note: M1 contributes to C00 and C11.
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (c+d)e
    # U: c(2), d(3) -> [0,0,1,1]
    # V: e(0) -> [1,0,0,0]
    # W: C10(1) -> [0,1,0,0] (C10 is at k=0,i=1 -> idx 1)
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1
    
    # M3 = a(f-h)
    # U: a(0) -> [1,0,0,0]
    # V: f(1), h(3) -> [0,1,0,-1]
    # W: C01(2) -> [0,0,1,0] (C01 is at k=1,i=0 -> idx 2)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[2][2] = 1
    
    # M4 = d(g-e)
    # U: d(3) -> [0,0,0,1]
    # V: g(2), e(0) -> [0,0,1,-1] (g is (1,0) idx 2, e is (0,0) idx 0)
    # W: C10(1), C11(3) -> [0,1,0,1]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[1][3] = 1; W[3][3] = 1
    
    # M5 = (a+b)h
    # U: a(0), b(1) -> [1,1,0,0]
    # V: h(3) -> [0,0,0,1]
    # W: C01(2), C11(3) -> [0,0,1,1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[3][4] = 1
    
    # M6 = (c-a)(e+f)
    # U: c(2), a(0) -> [-1,0,1,0]
    # V: e(0), f(1) -> [1,1,0,0]
    # W: C11(3) -> [0,0,0,1]
    U[0][5] = -1; U[2][5] = 1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (b-d)(g+h)
    # U: b(1), d(3) -> [0,1,0,-1]
    # V: g(2), h(3) -> [0,0,1,1]
    # W: C00(0) -> [1,0,0,0]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W, rank

def _combine_blocks(n, m, p, blocks):
    """
    Combine block decompositions into global U, V, W.
    blocks: list of tuples (n_i, m_i, p_i, U_i, V_i, W_i, i_off, j_off, k_off)
    """
    total_rank = sum(b[3] if isinstance(b[3], int) else len(b[3][0]) for b in blocks)
    # Ensure rank is int
    ranks = []
    for b in blocks:
        if isinstance(b[3], int):
            ranks.append(b[3])
        else:
            ranks.append(len(b[3][0]))
    total_rank = sum(ranks)
    
    U_full = [[0] * total_rank for _ in range(n * m)]
    V_full = [[0] * total_rank for _ in range(m * p)]
    W_full = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    for idx, (n_i, m_i, p_i, U_i, V_i, W_i, i_off, j_off, k_off) in enumerate(blocks):
        rank_i = ranks[idx]
        
        # Map U: local (r, c) -> global (i_off+r, j_off+c)
        for r in range(n_i):
            for c in range(m_i):
                global_row = (i_off + r) * m + (j_off + c)
                local_row = r * m_i + c
                for k in range(rank_i):
                    U_full[global_row][col_offset + k] = U_i[local_row][k]
        
        # Map V: local (r, c) -> global (j_off+r, k_off+c)
        for r in range(m_i):
            for c in range(p_i):
                global_row = (j_off + r) * p + (k_off + c)
                local_row = r * p_i + c
                for k in range(rank_i):
                    V_full[global_row][col_offset + k] = V_i[local_row][k]
        
        # Map W: local (r, c) -> global (k_off+r, i_off+c)
        # W index is k*n + i.
        # Local W row is r*n_i + c where r is k_local, c is i_local.
        for r in range(p_i):
            for c in range(n_i):
                global_row = (k_off + r) * n + (i_off + c)
                local_row = r * n_i + c
                for k in range(rank_i):
                    W_full[global_row][col_offset + k] = W_i[local_row][k]
        
        col_offset += rank_i
        
    return U_full, V_full, W_full, total_rank

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify decomposition by checking random matrices.
    """
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # C[i,k] = sum_c (sum_j U[i*m+j, c] A[i,j]) * (sum_l V[j*p+k, c] B[j,l]) ... 
    # Actually, the tensor definition is:
    # T[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
    # And we want T to represent the identity mapping for matmul?
    # No, the standard check is:
    # Let M = A @ B.
    # We can compute M_decomp using the decomposition.
    # The decomposition computes a bilinear map.
    # If we set the "inputs" to A and B, the "output" should be C.
    # The bilinear form for C[i,k] is sum_j A[i,j] B[j,k].
    # The decomposition provides:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] A[i,j]) * ... 
    # Wait, the standard tensor decomposition for matmul is:
    # C[i,k] = sum_c (sum_j U[i,j,c] A[i,j]) (sum_l V[j,l,c] B[j,l]) is NOT correct dimensionally.
    
    # Correct interpretation:
    # The tensor T has indices (i, j, k).
    # T[i,j,k] = 1 if the triple is valid in the sense that it contributes to C[i,k] via A[i,j]B[j,k].
    # Actually, for matmul, the tensor is defined such that:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # This is equivalent to:
    # C[i,k] = sum_{j,c} A[i,j] B[j,k] T[i,j,k,c] is not standard.
    
    # Standard tensor rank definition for matmul:
    # We want to compute C = A * B.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This is a trilinear form? No, it's bilinear in A and B.
    # The tensor T is in R^{(n*m) x (m*p) x (n*p)}.
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    
    # So, we verify that:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[i'*n+k', c] = 1 if (i,j,k) == (i',j',k') else 0.
    # Wait, W index is k*n+i. So W[i'*n+k'] is W[k'*n+i'].
    
    # Let's check a few entries.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Diagonal term should be 1
                val_diag = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(len(U[0])))
                if abs(val_diag - 1.0) > 1e-9:
                    return False
                
                # Off-diagonal terms should be 0
                # We can't check all, but we can check a few random ones
                # Or just trust the construction if diagonal is correct for all.
                # Given the linearity, if the diagonal is correct, the off-diagonal is 0
                # because the basis vectors are orthogonal in the construction.
                # However, Strassen mixes them. We must ensure the mixing doesn't leak.
                # The verification above (random matrix) is safer.
                
    # Let's do the random matrix check which is more robust.
    # C_decomp[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * ... 
    # No, the way to apply the tensor decomposition to matrices is:
    # Let U_c be the c-th column of U.
    # Let V_c be the c-th column of V.
    # Let W_c be the c-th column of W.
    # This doesn't directly give C.
    
    # The correct application:
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) ... NO.
    
    # The tensor T is defined by T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
    # We want T to be the "identity" tensor for matmul.
    # T[(i,j), (j',k), (i',k')] = 1 iff i=i', j=j', k=k'.
    
    # So, we just need to verify that for all i,j,k,i',j',k':
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[i'*n+k', c] = 1 if (i,j,k)==(i',j',k') else 0.
    
    # This is O((n*m*p)^3 * rank) which is 75^3 * 64 ~ 2.7M * 64 = 172M ops.
    # This might be too slow in Python loops.
    
    # Let's use NumPy for the random matrix check, which is faster.
    # If T is the identity tensor, then for any A, B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # Can we compute C from U,V,W?
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * ... 
    # This form is not standard.
    
    # Standard form:
    # C = sum_c W_c * (U_c^T A) * (V_c^T B) ... no.
    
    # Let's stick to the tensor entry verification but optimized.
    # We only need to check that the "diagonal" is 1 and "off-diagonal" is 0.
    # We can sample.
    
    # Actually, the random matrix check is:
    # Compute C_true = A @ B.
    # Compute C_decomp using the tensor?
    # The tensor decomposition *defines* the bilinear map.
    # If the tensor is the matmul tensor, then the map is matmul.
    # How to evaluate the map from U,V,W?
    # The map is: (A, B) -> C
    # C[i,k] = sum_c W[k*n+i, c] * (sum_j A[i,j] U[i*m+j, c]) * (sum_l B[j,l] V[j*p+l, c]) ... 
    # This is not right.
    
    # The correct evaluation of a tensor decomposition of matmul:
    # C[i,k] = sum_c (sum_j A[i,j] U[i*m+j, c]) * (sum_l B[j,l] V[j*p+l, c]) ... 
    # This implies U depends on i,j and c.
    # And V depends on j,l and c.
    # And W depends on i,k and c.
    
    # Let's check the dimensions.
    # Term c:
    # A_part_c = sum_j A[i,j] U[i*m+j, c]  (This is a scalar for fixed i,c)
    # B_part_c = sum_l B[j,l] V[j*p+l, c]  (This depends on j, which is summed out in A_part? No.)
    
    # The standard Strassen formula:
    # M1 = (A00+A11)(B00+B11)
    # This corresponds to:
    # U for M1: extracts A00, A11.
    # V for M1: extracts B00, B11.
    # W for M1: adds M1 to C00 and C11.
    
    # So, for a fixed c:
    # A_c = sum_j A[i,j] U[i*m+j, c]  -> This is a vector of length n? No.
    # For fixed c, U_c is a vector of length n*m.
    # A is n*m.
    # A_c_val = A_flat dot U_c? No.
    
    # Let's look at the indices.
    # U[i*m+j, c].
    # For a fixed c, let u_c be the vector of U[:,c].
    # Let v_c be the vector of V[:,c].
    # Let w_c be the vector of W[:,c].
    
    # The contribution of term c to C is:
    # C_c = outer(A_flat dot u_c, B_flat dot v_c) * w_c? No.
    
    # The bilinear form is:
    # C[i,k] = sum_c w_c[k*n+i] * (sum_j A[i,j] u_c[i*m+j]) * (sum_l B[j,l] v_c[j*p+l])
    # This is not separable like that.
    
    # Correct separation:
    # C[i,k] = sum_c w_c[k*n+i] * (sum_j A[i,j] u_c[i*m+j]) ... 
    # Wait, the j index is in both A and V.
    
    # Actually, the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c].
    # This T is the matmul tensor.
    # This means:
    # C[i,k] = sum_j A[i,j] B[j,k] T[i,j,k] ... no.
    
    # The definition is:
    # The matmul tensor IS the tensor T.
    # So if we verify T[i,j,k] = delta(i,j,k) (in the sense of the identity tensor),
    # then the decomposition is correct.
    
    # Let's verify T[i,j,k] = 1 if (i,j,k) is a valid triple, 0 otherwise?
    # No, the matmul tensor is NOT the identity tensor on (i,j,k).
    # The matmul tensor maps A, B to C.
    # C[i,k] = sum_j A[i,j] B[j,k].
    # This can be written as:
    # C[i,k] = sum_{j,c} A[i,j] B[j,k] T[i,j,k,c] ... no.
    
    # The tensor T is in R^{n*m x m*p x n*p}.
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k', else 0.
    
    # So we MUST verify that:
    # sum_c U[i*m+j, c] V[j'*p+k, c] W[i'*n+k', c] = 1 if (i,j,k)==(i',j',k') else 0.
    
    # This is the only correct verification.
    # We will sample random (i,j,k) and (i',j',k') to check this.
    
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check diagonal
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
        if abs(val - 1.0) > 1e-9:
            return False
            
    # Check off-diagonal
    for _ in range(100):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        i2 = np.random.randint(n)
        j2 = np.random.randint(m)
        k2 = np.random.randint(p)
        if (i,j,k) == (i2,j2,k2):
            continue
        val = np.sum(U_arr[i*m+j] * V_arr[j2*p+k2] * W_arr[k2*n+i2])
        if abs(val) > 1e-9:
            return False
            
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110
    
    # Partition:
    # n=3 -> 1+2
    # m=5 -> 2+3
    # p=5 -> 2+3
    
    blocks = []
    
    # Block 1: n=1, m=2, p=2 at (0, 0, 0) -> Strassen? No, Strassen is 2x2x2.
    # We can use Strassen for 2x2x2 blocks.
    # Let's use Strassen for the 2x2x2 part of the 2x3x3 block?
    # No, let's stick to the plan:
    # 1. 1x3x3 (Schoolbook, rank 9) at (0, 2, 2) -> i=0, j=2, k=2
    # 2. 1x2x3 (Schoolbook, rank 6) at (0, 0, 2) -> i=0, j=0, k=2
    # 3. 2x2x2 (Strassen, rank 7) at (1, 0, 0) -> i=1, j=0, k=0
    # 4. 2x3x3 (Schoolbook, rank 18) at (1, 2, 2) -> i=1, j=2, k=2
    # 5. 2x2x3 (Schoolbook, rank 12) at (1, 0, 2) -> i=1, j=0, k=2
    # 6. 2x3x2 (Schoolbook, rank 12) at (1, 2, 0) -> i=1, j=2, k=0
    
    # Wait, the indices must cover the full 3x5x5.
    # i in 0..2, j in 0..4, k in 0..4.
    
    # Block 1: i=0, j=0..1, k=0..1 -> 1x2x2. Rank 4 (Schoolbook) or 4 (Strassen is 2x2x2, so not applicable).
    # Let's use 2x2x2 Strassen for the bottom-right 2x2x2 block of the 2x3x3 area?
    
    # Let's refine the blocks to be contiguous and cover the space.
    # i=0:
    #   j=0..1, k=0..1: 1x2x2 (Schoolbook, rank 4)
    #   j=0..1, k=2..4: 1x2x3 (Schoolbook, rank 6)
    #   j=2..4, k=0..1: 1x3x2 (Schoolbook, rank 6)
    #   j=2..4, k=2..4: 1x3x3 (Schoolbook, rank 9)
    # i=1..2:
    #   j=0..1, k=0..1: 2x2x2 (Strassen, rank 7)
    #   j=0..1, k=2..4: 2x2x3 (Schoolbook, rank 12)
    #   j=2..4, k=0..1: 2x3x2 (Schoolbook, rank 12)
    #   j=2..4, k=2..4: 2x3x3 (Schoolbook, rank 18)
    
    # Total Rank = 4 + 6 + 6 + 9 + 7 + 12 + 12 + 18 = 74.
    # This is only 1 better than schoolbook (75).
    
    # Can we do better?
    # Use Strassen for 2x2x2.
    # Can we use Laderman for 3x3x3? Rank 21 vs 27.
    # If we use Laderman for the 1x3x3? No, Laderman is 3x3x3.
    # We have a 1x3x3 block. Rank 9.
    # If we split i=0..2 into 3 blocks of 1x5x5?
    # 1x5x5 rank 25. 3 * 25 = 75.
    
    # Let's try to use Laderman for the 3x3x3 sub-block at i=0..2, j=0..2, k=0..2.
    # This block is 3x3x3. Rank 21 (Laderman) vs 27 (Schoolbook).
    # Remaining blocks:
    # i=0..2, j=0..2, k=3..4: 3x3x2. Rank 18.
    # i=0..2, j=3..4, k=0..2: 3x2x3. Rank 18.
    # i=0..2, j=3..4, k=3..4: 3x2x2. Rank 12.
    # Total Rank = 21 + 18 + 18 + 12 = 69.
    
    # This is better! 69 < 75.
    # We need a correct Laderman implementation.
    
    # I will implement a verified Laderman 3x3x3 decomposition.
    
    def _laderman_3x3():
        # Laderman's 21 bilinear forms.
        # Source: "A non-recursive algorithm for multiplying 3x3 matrices" (1976)
        # I will use a known correct set.
        # A = [[a00, a01, a02], [a10, a11, a12], [a20, a21, a22]]
        # B = [[b00, b01, b02], [b10, b11, b12], [b20, b21, b22]]
        
        U = []
        V = []
        W = []
        
        # Term 0: a00*b00 -> C00
        U.append([1,0,0, 0,0,0, 0,0,0])
        V.append([1,0,0, 0,0,0, 0,0,0])
        W.append([1,0,0, 0,0,0, 0,0,0]) # C00 at k=0,i=0 -> idx 0
        
        # Term 1: a00*b11 -> C01
        U.append([1,0,0, 0,0,0, 0,0,0])
        V.append([0,0,0, 0,1,0, 0,0,0])
        W.append([0,0,0, 1,0,0, 0,0,0]) # C01 at k=1,i=0 -> idx 3
        
        # Term 2: a00*b22 -> C02
        U.append([1,0,0, 0,0,0, 0,0,0])
        V.append([0,0,0, 0,0,0, 0,0,1])
        W.append([0,0,0, 0,0,0, 0,0,1]) # C02 at k=2,i=0 -> idx 6
        
        # Term 3: (a01+a02)*b10 -> C00
        U.append([0,1,1, 0,0,0, 0,0,0])
        V.append([0,0,0, 1,0,0, 0,0,0])
        W.append([1,0,0, 0,0,0, 0,0,0])
        
        # Term 4: (a01+a02)*b20 -> C00
        U.append([0,1,1, 0,0,0, 0,0,0])
        V.append([0,0,0, 0,0,0, 1,0,0])
        W.append([1,0,0, 0,0,0, 0,0,0])
        
        # Term 5: (a10+a20)*b01 -> C10
        U.append([0,0,0, 1,0,0, 1,0,0])
        V.append([0,1,0, 0,0,0, 0,0,0])
        W.append([0,1,0, 0,0,0, 0,0,0]) # C10 at k=0,i=1 -> idx 1
        
        # Term 6: (a10+a20)*b02 -> C20
        U.append([0,0,0, 1,0,0, 1,0,0])
        V.append([0,0,1, 0,0,0, 0,0,0])
        W.append([0,0,1, 0,0,0, 0,0,0]) # C20 at k=0,i=2 -> idx 2
        
        # Term 7: (a11+a21)*b10 -> C10
        U.append([0,0,0, 0,1,0, 0,1,0])
        V.append([0,0,0, 1,0,0, 0,0,0])
        W.append([0,1,0, 0,0,0, 0,0,0])
        
        # Term 8: (a11+a21)*b20 -> C10
        U.append([0,0,0, 0,1,0, 0,1,0])
        V.append([0,0,0, 0,0,0, 1,0,0])
        W.append([0,1,0, 0,0,0, 0,0,0])
        
        # Term 9: (a12+a22)*b10 -> C20
        U.append([0,0,0, 0,0,1, 0,0,1])
        V.append([0,0,0, 1,0,0, 0,0,0])
        W.append([0,0,1, 0,0,0, 0,0,0])
        
        # Term 10: (a12+a22)*b20 -> C20
        U.append([0,0,0, 0,0,1, 0,0,1])
        V.append([0,0,0, 0,0,0, 1,0,0])
        W.append([0,0,1, 0,0,0, 0,0,0])
        
        # Term 11: (a01+a11+b00+b10)*b11 -> C01
        # This term is not purely bilinear in A and B separately?
        # Laderman's formula involves sums of A and B.
        # M12 = (a01+a11+b00+b10) * b11
        # This is not of the form (u^T A)(v^T B).
        # It is (a01+a11)b11 + (b00+b10)b11.
        # The second part (b00+b10)b11 is quadratic in B.
        # Laderman's algorithm is NOT a tensor decomposition in the standard sense?
        # No, Laderman's algorithm IS a tensor decomposition.
        # The terms are bilinear.
        # Let's check the standard Laderman terms.
        # M12 = (a01 + a11) * b11 + b00*b11 + b10*b11.
        # This is not a single rank-1 tensor term.
        # Laderman's algorithm uses 21 multiplications, but some are not simple products of linear forms?
        # Actually, Laderman's algorithm is a bilinear algorithm.
        # The 21 terms are bilinear.
        
        # I will skip Laderman due to complexity and risk of error.
        # I will stick to the 74 rank solution (Strassen 2x2x2).
        # Or better, just return schoolbook if I can't guarantee a lower rank quickly.
        # But 74 is an improvement.
        
        return None

    # Let's use the 74 rank solution.
    # Blocks:
    # 1. 1x2x2 at (0,0,0) Schoolbook
    # 2. 1x2x3 at (0,0,2) Schoolbook
    # 3. 1x3x2 at (0,2,0) Schoolbook
    # 4. 1x3x3 at (0,2,2) Schoolbook
    # 5. 2x2x2 at (1,0,0) Strassen
    # 6. 2x2x3 at (1,0,2) Schoolbook
    # 7. 2x3x2 at (1,2,0) Schoolbook
    # 8. 2x3x3 at (1,2,2) Schoolbook
    
    blocks = []
    
    # 1
    U1, V1, W1, r1 = _schoolbook_block(1, 2, 2)
    blocks.append((1, 2, 2, U1, V1, W1, 0, 0, 0))
    
    # 2
    U2, V2, W2, r2 = _schoolbook_block(1, 2, 3)
    blocks.append((1, 2, 3, U2, V2, W2, 0, 0, 2))
    
    # 3
    U3, V3, W3, r3 = _schoolbook_block(1, 3, 2)
    blocks.append((1, 3, 2, U3, V3, W3, 0, 2, 0))
    
    # 4
    U4, V4, W4, r4 = _schoolbook_block(1, 3, 3)
    blocks.append((1, 3, 3, U4, V4, W4, 0, 2, 2))
    
    # 5
    U5, V5, W5, r5 = _strassen_2x2_block()
    blocks.append((2, 2, 2, U5, V5, W5, 1, 0, 0))
    
    # 6
    U6, V6, W6, r6 = _schoolbook_block(2, 2, 3)
    blocks.append((2, 2, 3, U6, V6, W6, 1, 0, 2))
    
    # 7
    U7, V7, W7, r7 = _schoolbook_block(2, 3, 2)
    blocks.append((2, 3, 2, U7, V7, W7, 1, 2, 0))
    
    # 8
    U8, V8, W8, r8 = _schoolbook_block(2, 3, 3)
    blocks.append((2, 3, 3, U8, V8, W8, 1, 2, 2))
    
    U_full, V_full, W_full, total_rank = _combine_blocks(n, m, p, blocks)
    
    if _verify_decomposition(U_full, V_full, W_full, n, m, p):
        return {"U": U_full, "V": V_full, "W": W_full}
    else:
        # Fallback to schoolbook
        U_s, V_s, W_s, _ = _schoolbook_block(n, m, p)
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
