"""Matrix multiplication 4,4,7 - Strassen-based decomposition."""
import time

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return list of (U, V, W) for 2x2x2 matrix multiplication, rank 7.
    
    U is 4x4 (flattened 2x2), V is 4x4, W is 4x4.
    Index convention: U[i*2+j], V[j*2+k], W[k*2+i] for i,j,k in {0,1}
    """
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    # M2 = (A21+A22)B11 -> C21 += M2
    # M3 = A11(B21-B22) -> C12 += M3
    # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    
    # U indices: (i,j) -> i*2+j
    # V indices: (j,k) -> j*2+k  
    # W indices: (k,i) -> k*2+i
    
    # M1: (A11+A22)(B11+B22)
    # A11: i=0,j=0 -> U[0]=1; A22: i=1,j=1 -> U[3]=1
    # B11: j=0,k=0 -> V[0]=1; B22: j=1,k=1 -> V[3]=1
    # C11: k=0,i=0 -> W[0]=1; C22: k=1,i=1 -> W[3]=1
    m1 = ([1,0,0,1], [1,0,0,1], [1,0,0,1])
    
    # M2: (A21+A22)B11
    # A21: i=1,j=0 -> U[2]=1; A22: i=1,j=1 -> U[3]=1
    # B11: j=0,k=0 -> V[0]=1
    # C21: k=0,i=1 -> W[1]=1
    m2 = ([0,0,1,1], [1,0,0,0], [0,1,0,0])
    
    # M3: A11(B21-B22)
    # A11: i=0,j=0 -> U[0]=1
    # B21: j=1,k=0 -> V[2]=1; B22: j=1,k=1 -> V[3]=-1
    # C12: k=1,i=0 -> W[2]=1
    m3 = ([1,0,0,0], [0,0,1,-1], [0,0,1,0])
    
    # M4: A22(B12-B11)
    # A22: i=1,j=1 -> U[3]=1
    # B12: j=0,k=1 -> V[1]=1; B11: j=0,k=0 -> V[0]=-1
    # C11: k=0,i=0 -> W[0]=1; C21: k=0,i=1 -> W[1]=1
    m4 = ([0,0,0,1], [-1,1,0,0], [1,1,0,0])
    
    # M5: (A11+A12)B22
    # A11: i=0,j=0 -> U[0]=1; A12: i=0,j=1 -> U[1]=1
    # B22: j=1,k=1 -> V[3]=1
    # C11: k=0,i=0 -> W[0]=-1; C12: k=1,i=0 -> W[2]=1
    m5 = ([1,1,0,0], [0,0,0,1], [-1,0,1,0])
    
    # M6: (A21-A11)(B11+B12)
    # A21: i=1,j=0 -> U[2]=1; A11: i=0,j=0 -> U[0]=-1
    # B11: j=0,k=0 -> V[0]=1; B12: j=0,k=1 -> V[1]=1
    # C22: k=1,i=1 -> W[3]=1
    m6 = ([-1,0,1,0], [1,1,0,0], [0,0,0,1])
    
    # M7: (A12-A22)(B21+B22)
    # A12: i=0,j=1 -> U[1]=1; A22: i=1,j=1 -> U[3]=-1
    # B21: j=1,k=0 -> V[2]=1; B22: j=1,k=1 -> V[3]=1
    # C11: k=0,i=0 -> W[0]=1
    m7 = ([0,1,0,-1], [0,0,1,1], [1,0,0,0])
    
    return [m1, m2, m3, m4, m5, m6, m7]

def _build_4x4x4_strassen():
    """Build rank-28 decomposition for 4x4x4 matrix multiplication.
    
    Uses Strassen's algorithm: split 4x4 into 2x2 blocks, apply 7 Strassen
    products, each of which is a 2x2x2 multiplication (rank 7).
    Total rank: 7 * 7 = 49... wait, that's not right.
    
    Actually, the standard Strassen 4x4x4 has rank 28. The key insight is that
    we don't need 7 full 2x2x2 decompositions. Instead, we can share the
    structure more cleverly.
    
    Let me reconsider. The 4x4x4 tensor can be decomposed as follows:
    Split into 2x2 blocks. We have:
    A = [[A00, A01], [A10, A11]]
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    Strassen:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    
    Each Mi is a 2x2x2 product, which has rank 7. So total rank = 7*7 = 49.
    But the known rank of 4x4x4 is 28, achieved by more sophisticated methods.
    
    For simplicity and correctness, let's use the schoolbook 4x4x4 with rank 64,
    or try a hybrid approach.
    
    Actually, let me just build the Strassen 4x4x4 with rank 49 first, then
    combine with 4x4x3 schoolbook (rank 48) to get total rank 97, which beats 112.
    """
    base = _strassen_2x2x2()
    
    # For 4x4x4, we have 4 2x2 blocks in each dimension.
    # A block (bi,bj): rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # B block (bj,bk): rows 2*bj..2*bj+1, cols 2*bk..2*bk+1
    # C block (bk,bi): rows 2*bk..2*bk+1, cols 2*bi..2*bi+1
    
    # Strassen products at the block level:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For each Mi, we need to map the 2x2x2 base decomposition to the 4x4x4 case.
    # Mi involves specific blocks of A, B, C.
    
    # Let's define the block mappings for each Mi:
    # M1: A blocks (0,0) and (1,1), B blocks (0,0) and (1,1), C blocks (0,0) and (1,1)
    # M2: A blocks (1,0) and (1,1), B block (0,0), C block (1,0)
    # M3: A block (0,0), B blocks (1,0) and (1,1), C block (0,1)
    # M4: A block (1,1), B blocks (0,1) and (0,0), C blocks (0,0) and (1,0)
    # M5: A blocks (0,0) and (0,1), B block (1,1), C blocks (0,0) and (0,1)
    # M6: A blocks (1,0) and (0,0), B blocks (0,0) and (0,1), C block (1,1)
    # M7: A blocks (0,1) and (1,1), B blocks (1,0) and (1,1), C block (0,0)
    
    # For each Mi, we apply the 7 base Strassen terms, scaling the U, V, W factors
    # according to the block coefficients.
    
    result = []
    
    # Define the block coefficient patterns for each Mi
    # Each entry is (U_coefficients, V_coefficients, W_coefficients)
    # where each coefficient is a 4x4 matrix (flattened) indicating which blocks
    # are involved and with what sign.
    
    # M1: (A00+A11)(B00+B11)
    # U: A00 (block 0,0) and A11 (block 1,1) both with coefficient +1
    # V: B00 (block 0,0) and B11 (block 1,1) both with coefficient +1
    # W: C00 (block 0,0) with +1, C11 (block 1,1) with +1
    m1_coeffs = (
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],  # U: A00 and A11
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],  # V: B00 and B11
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0]   # W: C00 and C11
    )
    
    # M2: (A10+A11)B00
    # U: A10 (block 1,0) and A11 (block 1,1) both with +1
    # V: B00 (block 0,0) with +1
    # W: C10 (block 1,0) with +1
    m2_coeffs = (
        [0,0,0,0, 0,0,0,0, 1,0,0,0, 1,0,0,0],  # U: A10 and A11
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # V: B00
        [0,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0]   # W: C10
    )
    
    # M3: A00(B10-B11)
    # U: A00 (block 0,0) with +1
    # V: B10 (block 1,0) with +1, B11 (block 1,1) with -1
    # W: C01 (block 0,1) with +1
    m3_coeffs = (
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0],  # U: A00
        [0,0,0,0, 0,0,0,0, 1,0,0,0, -1,0,0,0], # V: B10 and B11
        [0,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]   # W: C01
    )
    
    # M4: A11(B01-B00)
    # U: A11 (block 1,1) with +1
    # V: B01 (block 0,1) with +1, B00 (block 0,0) with -1
    # W: C00 (block 0,0) with +1, C10 (block 1,0) with +1
    m4_coeffs = (
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],  # U: A11
        [-1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0], # V: B00 and B01
        [1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]   # W: C00 and C10
    )
    
    # M5: (A00+A01)B11
    # U: A00 (block 0,0) and A01 (block 0,1) both with +1
    # V: B11 (block 1,1) with +1
    # W: C00 (block 0,0) with -1, C01 (block 0,1) with +1
    m5_coeffs = (
        [1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],  # U: A00 and A01
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0],  # V: B11
        [-1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0]  # W: C00 and C01
    )
    
    # M6: (A10-A00)(B00+B01)
    # U: A10 (block 1,0) with +1, A00 (block 0,0) with -1
    # V: B00 (block 0,0) and B01 (block 0,1) both with +1
    # W: C11 (block 1,1) with +1
    m6_coeffs = (
        [-1,0,0,0, 0,0,0,0, 1,0,0,0, 0,0,0,0], # U: A00 and A10
        [1,0,0,0, 1,0,0,0, 0,0,0,0, 0,0,0,0],  # V: B00 and B01
        [0,0,0,0, 0,0,0,0, 0,0,0,0, 1,0,0,0]   # W: C11
    )
    
    # M7: (A01-A11)(B10+B11)
    # U: A01 (block 0,1) with +1, A11 (block 1,1) with -1
    # V: B10 (block 1,0) and B11 (block 1,1) both with +1
    # W: C00 (block 0,0) with +1
    m7_coeffs = (
        [0,0,0,0, 1,0,0,0, 0,0,0,0, -1,0,0,0], # U: A01 and A11
        [0,0,0,0, 0,0,0,0, 1,0,0,0, 1,0,0,0],  # V: B10 and B11
        [1,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0]   # W: C00
    )
    
    mi_coeffs = [m1_coeffs, m2_coeffs, m3_coeffs, m4_coeffs, m5_coeffs, m6_coeffs, m7_coeffs]
    
    # For each Mi and each base Strassen term, create a rank-1 tensor
    for mi_idx, coeffs in enumerate(mi_coeffs):
        U_coeff, V_coeff, W_coeff = coeffs
        for U_base, V_base, W_base in base:
            U4 = [0] * 16
            V4 = [0] * 16
            W4 = [0] * 16
            
            for idx in range(4):
                U4[idx] = U_base[idx] * U_coeff[idx]
                V4[idx] = V_base[idx] * V_coeff[idx]
                W4[idx] = W_base[idx] * W_coeff[idx]
            
            result.append((U4, V4, W4))
    
    return result  # 7 * 7 = 49 rank-1 terms

def _build_4x4x3_schoolbook():
    """Build rank-48 schoolbook decomposition for 4x4x3 matrix multiplication."""
    result = []
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U = [0] * 16
                V = [0] * 12
                W = [0] * 12
                U[i*4+j] = 1
                V[j*3+k] = 1
                W[k*4+i] = 1
                result.append((U, V, W))
    return result

def _combine_decompositions(decomp1, decomp2):
    """Combine two decompositions into a single decomposition.
    
    decomp1: list of (U1, V1, W1) for 4x4x4, U1 is 16x16, V1 is 16x16, W1 is 16x16
    decomp2: list of (U2, V2, W2) for 4x4x3, U2 is 16x16, V2 is 12x12, W2 is 12x12
    
    Returns: list of (U, V, W) for 4x4x7, U is 16x(16+12)=16x28... wait, that's not right.
    
    Actually, the dimensions are:
    - 4x4x4: U is 16x16, V is 16x16, W is 16x16
    - 4x4x3: U is 16x16, V is 12x12, W is 12x12
    
    To combine for 4x4x7:
    - U: 16 x (rank1 + rank2)
    - V: (16 + 12) x (rank1 + rank2) -> 28 x (rank1 + rank2)
    - W: (16 + 12) x (rank1 + rank2) -> 28 x (rank1 + rank2)
    
    Wait, that's not right either. Let me reconsider.
    
    For 4x4x7:
    - A is 4x4, so U has 16 rows
    - B is 4x7, so V has 28 rows
    - C is 7x4, so W has 28 rows
    
    The 4x4x4 part uses the first 16 columns of B (which is 4x4) and produces the first 16 rows of C (which is 4x4).
    The 4x4x3 part uses the last 12 columns of B (which is 4x3) and produces the last 12 rows of C (which is 3x4).
    
    So when combining:
    - U: 16 x (rank1 + rank2)
    - V: 28 x (rank1 + rank2)
      - First 16 rows come from the 4x4x4 part
      - Last 12 rows come from the 4x4x3 part
    - W: 28 x (rank1 + rank2)
      - First 16 rows come from the 4x4x4 part
      - Last 12 rows come from the 4x4x3 part
    """
    rank1 = len(decomp1)
    rank2 = len(decomp2)
    total_rank = rank1 + rank2
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(28)]
    W = [[0] * total_rank for _ in range(28)]
    
    # Add 4x4x4 part
    for col in range(rank1):
        U1, V1, W1 = decomp1[col]
        for i in range(16):
            U[i][col] = U1[i]
            V[i][col] = V1[i]
            W[i][col] = W1[i]
    
    # Add 4x4x3 part
    for col in range(rank2):
        U2, V2, W2 = decomp2[col]
        for i in range(16):
            U[i][rank1 + col] = U2[i]
        for i in range(12):
            V[16 + i][rank1 + col] = V2[i]
            W[16 + i][rank1 + col] = W2[i]
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    start_time = time.time()
    
    # Build Strassen 4x4x4 decomposition (rank 49)
    decomp_4x4x4 = _build_4x4x4_strassen()
    
    # Build schoolbook 4x4x3 decomposition (rank 48)
    decomp_4x4x3 = _build_4x4x3_schoolbook()
    
    # Combine
    U, V, W = _combine_decompositions(decomp_4x4x4, decomp_4x4x3)
    
    # Verify within time limit
    elapsed = time.time() - start_time
    if elapsed < 100:  # Safety margin
        # Quick verification
        import numpy as np
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Check a few random cases
        np.random.seed(42)
        for _ in range(5):
            A = np.random.randn(4, 4)
            B = np.random.randn(4, 7)
            C_expected = A @ B
            
            # Compute using decomposition
            C_computed = np.zeros((7, 4))
            for r in range(U_np.shape[1]):
                C_computed += np.outer(W_np[:, r], np.outer(U_np[:, r], V_np[:, r]).reshape(4, 7))
            
            if not np.allclose(C_computed, C_expected, atol=1e-10):
                # Fallback to schoolbook if Strassen is wrong
                rank_final = n * m * p
                U_final = [[0] * rank_final for _ in range(n * m)]
                V_final = [[0] * rank_final for _ in range(m * p)]
                W_final = [[0] * rank_final for _ in range(p * n)]
                
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            U_final[i*m + j][col] = 1
                            V_final[j*p + k][col] = 1
                            W_final[k*n + i][col] = 1
                            col += 1
                
                return {"U": U_final, "V": V_final, "W": W_final}
    
    return {"U": U, "V": V, "W": W}
