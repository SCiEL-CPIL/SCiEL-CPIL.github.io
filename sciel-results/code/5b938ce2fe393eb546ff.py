"""Matrix multiplication 4x4 * 4x5 via Strassen + schoolbook for extra column.
Rank 65 = 49 (Strassen for 4x4*4x4) + 16 (schoolbook for 4x4*4x1).
"""
import time
import numpy as np
from fractions import Fraction
import sys

def _strassen_2x2():
    """Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U (4x7), V (4x7), W (4x7).
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Standard Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    #
    # U[i*2+j, s] = coefficient of A[i,j] in product s
    # V[j*2+k, s] = coefficient of B[j,k] in product s
    # W[k*2+i, s] = coefficient of C[i,k] from product s
    
    # Product 0: M1 = (a00+a11)(b00+b11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = -1; W[1, 0] = -1; W[2, 0] = 1
    # C00 += M1, C11 -= M1, C01 -= M1, C10 += M1
    # W[k*2+i]: W[0*2+0]=W[0]=1 (C00), W[1*2+1]=W[3]=-1 (C11), W[0*2+1]=W[2]=-1 (C10), W[1*2+0]=W[1]=1 (C01)
    # Wait, W[k*2+i] where k is row of B/col of C, i is row of A/row of C
    # C[i,k] gets contribution. W[k*2+i] means for output C[i,k], the factor is W[k*2+i]
    # So W[0*2+0]=W[0] for C[0,0], W[0*2+1]=W[1] for C[1,0], W[1*2+0]=W[2] for C[0,1], W[1*2+1]=W[3] for C[1,1]
    # C00 = M1 + M4 - M5 - M7: W[0] gets +1 from M1
    # C01 = M3 + M5: W[2] gets -1 from M1? No...
    # Let me redo: W[k*2+i] is the factor for C[i,k]
    # C[i,k] = sum_s U[i*2+j, s] * V[j*2+k, s] * W[k*2+i, s]
    # For M1: contributes to C00, C01, C10, C11
    # C00 += M1: W[0*2+0] = W[0] += 1
    # C01 += M1: W[1*2+0] = W[2] += 1  (C[0,1], k=1, i=0)
    # C10 += M1: W[0*2+1] = W[1] += 1  (C[1,0], k=0, i=1)
    # C11 += M1: W[1*2+1] = W[3] += 1
    # But from Strassen: C00 = M1+M4-M5-M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M7
    # So M1 contributes: C00+=1, C01+=0, C10+=0, C11+=1
    # Thus W[0]=1, W[2]=0, W[1]=0, W[3]=1 for product 0
    
    # Let me restart with correct Strassen formulas
    U = np.zeros((4, 7), dtype=np.int64)
    V = np.zeros((4, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # M1 = (a00+a11)(b00+b11)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1  # C00
    W[3,0]=1  # C11
    
    # M2 = (a10+a11)b00
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1  # C10
    W[3,1]=-1 # C11
    
    # M3 = a00(b10-b11)
    U[0,2]=1
    V[2,2]=1; V[3,2]=-1
    W[1,2]=1  # C01
    W[3,2]=1  # C11
    
    # M4 = a11(b01-b00)
    U[3,3]=1
    V[1,3]=1; V[0,3]=-1
    W[0,3]=1  # C00
    W[1,3]=1  # C10
    
    # M5 = (a00+a01)b11
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1 # C00
    W[2,4]=1  # C01
    
    # M6 = (a10-a00)(b00+b01)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[0,5]=1  # C00
    W[1,5]=1  # C10
    
    # M7 = (a01-a11)(b10+b11)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=-1 # C00
    W[3,6]=1  # C11
    
    return U, V, W

def _build_f4_4x4_4x4():
    """Placeholder - not used. Strassen is used instead."""
    raise NotImplementedError("Not used")

def _build_strassen_4x4_4x4():
    """Build Strassen-based decomposition for 4x4 * 4x4 -> 4x4 with rank 49.
    
    Uses 7 Strassen 2x2 multiplications at each of 7 block positions.
    Block positions: (0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)
    Wait, 4x4 split into 2x2 blocks gives 4 block positions, not 7.
    
    Actually, for 4x4*4x4, we split into 2x2 blocks:
    A = [[A00, A01], [A10, A11]] where each is 2x2
    B = [[B00, B01], [B10, B11]]
    C = [[C00, C01], [C10, C11]]
    
    C00 = A00*B00 + A10*B10
    C01 = A00*B01 + A10*B11
    C10 = A01*B00 + A11*B10
    C11 = A01*B01 + A11*B11
    
    Each C_ij is a 2x2 matrix product sum. Using Strassen for each pair:
    C00: 7 mults for A00*B00, 7 for A10*B10 -> 14
    C01: 7 mults for A00*B01, 7 for A10*B11 -> 14
    C10: 7 mults for A01*B00, 7 for A11*B10 -> 14
    C11: 7 mults for A01*B01, 7 for A11*B11 -> 14
    Total: 56, not 49.
    
    The standard Strassen for 4x4 uses 49 = 7^2 mults by treating the 4x4
    as a 2x2 block matrix and applying Strassen at the block level:
    M1 = (A00+A11)(B00+B11)
    M2 = (A10+A11)B00
    M3 = A00(B10-B11)
    M4 = A11(B01-B00)
    M5 = (A00+A01)B11
    M6 = (A10-A00)(B00+B01)
    M7 = (A01-A11)(B10+B11)
    
    Each M is a 2x2*2x2 = 4 scalar mults, but using Strassen recursively: 7 mults each.
    Total: 7 * 7 = 49.
    """
    n = m = p = 4
    rank = 49
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    sU, sV, sW = _strassen_2x2()
    
    # Block-level Strassen: 7 products, each a 2x2*2x2 computed with 7 sub-products
    # Block A indices: (br, bc) -> rows br*2:(br+1)*2, cols bc*2:(bc+1)*2
    # A block (br, bc) has elements A[br*2+i, bc*2+j] for i,j in 0,1
    # U index: i*m+j where i is row of A, j is col of A
    # For A block (br, bc), element (i,j) -> U[(br*2+i)*4 + (bc*2+j), col]
    
    # B block (br, bc): B[br*2+i, bc*2+j], V index: j*p+k = (bc*2+j)*4 + (br*2+i)
    # Wait, V[j*p+k] where j is col of A / row of B, k is col of B
    # For B block (br, bc), element at row br*2+i, col bc*2+j
    # V index: (bc*2+j)*4 + (br*2+i)  -- NO
    # V is indexed by [j, k] where j is the shared index (col of A, row of B)
    # For B, row = j (shared), col = k
    # B block (br, bc): rows are br*2+i (this is j), cols are bc*2+j (this is k)
    # So V[j*p+k] = V[(br*2+i)*4 + (bc*2+j)]
    
    # W index: [k, i] where k is col of C, i is row of C
    # C block (br, bc): rows br*2+i (this is i), cols bc*2+j (this is k)
    # W[k*n+i] = W[(bc*2+j)*4 + (br*2+i)]
    
    # Block-level Strassen products:
    # Each product involves specific A blocks, B blocks, and contributes to C blocks
    # 
    # M1 = (A00+A11)(B00+B11) -> contributes to C00, C11
    # M2 = (A10+A11)B00 -> contributes to C10, C11
    # M3 = A00(B10-B11) -> contributes to C01, C11
    # M4 = A11(B01-B00) -> contributes to C00, C10
    # M5 = (A00+A01)B11 -> contributes to C00, C01
    # M6 = (A10-A00)(B00+B01) -> contributes to C00, C10
    # M7 = (A01-A11)(B10+B11) -> contributes to C00, C11
    #
    # C00 = M1 + M4 - M5 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M7
    
    # For each of the 7 block-level products, we compute a 2x2*2x2 using Strassen
    # The 2x2*2x2 has 7 sub-products, each with specific U, V, W factors
    
    # A block (br, bc) -> for sub-product s, the 2x2 matrix elements:
    # U[2x2 index xi, s] where xi = i*2+j for the 2x2 matrix
    # This maps to A[(br*2+i), (bc*2+j)] -> U[(br*2+i)*4 + (bc*2+j), col]
    
    # B block (br, bc) -> for sub-product s:
    # V[2x2 index yi, s] where yi = i*2+j
    # This maps to B[(br*2+i), (bc*2+j)] -> V[(bc*2+j)*4 + (br*2+i), col]
    # Wait, V[j*p+k] where j is row of B, k is col of B
    # B row = br*2+i, B col = bc*2+j
    # V[(br*2+i)*4 + (bc*2+j), col]
    
    # C block (br, bc) -> for sub-product s:
    # W[2x2 index zi, s] where zi = i*2+j
    # This maps to C[(br*2+i), (bc*2+j)] -> W[(bc*2+j)*4 + (br*2+i), col]
    # W[k*n+i] where k is col of C, i is row of C
    # C row = br*2+i, C col = bc*2+j
    # W[(bc*2+j)*4 + (br*2+i), col]
    
    # Block-level Strassen specification:
    # (a_blocks, b_blocks, c_blocks, c_signs)
    # a_blocks: list of (block_row, block_col, sign) for A
    # b_blocks: list of (block_row, block_col, sign) for B
    # c_blocks: list of (block_row, block_col, sign) for C
    
    block_products = [
        # M1 = (A00+A11)(B00+B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10+A11)B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00(B10-B11)
        ([(0,0,1)], [(1,0,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11(B01-B00)
        ([(1,1,1)], [(0,1,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00+A01)B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10-A00)(B00+B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(0,0,1),(1,0,1)]),
        # M7 = (A01-A11)(B10+B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,-1),(1,1,1)]),
    ]
    
    for pi, (a_blocks, b_blocks, c_blocks) in enumerate(block_products):
        for s in range(7):
            col = pi * 7 + s
            
            # A contribution
            for (br, bc, sign) in a_blocks:
                for xi in range(4):
                    u_val = sU[xi, s]
                    if u_val == 0: continue
                    i2, j2 = xi // 2, xi % 2
                    row = (br*2 + i2) * 4 + (bc*2 + j2)
                    U[row, col] += u_val * sign
            
            # B contribution
            for (br, bc, sign) in b_blocks:
                for yi in range(4):
                    v_val = sV[yi, s]
                    if v_val == 0: continue
                    i2, j2 = yi // 2, yi % 2
                    row = (br*2 + i2) * 4 + (bc*2 + j2)
                    V[row, col] += v_val * sign
            
            # C contribution
            for (br, bc, sign) in c_blocks:
                for zi in range(4):
                    w_val = sW[zi, s]
                    if w_val == 0: continue
                    i2, j2 = zi // 2, zi % 2
                    row = (bc*2 + j2) * 4 + (br*2 + i2)
                    W[row, col] += w_val * sign
    
    return U, V, W

def _build_extra_column():
    """Schoolbook for 4x4 * 4x1 -> 4x1, rank 16."""
    n, m, p = 4, 4, 1
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, rank):
    """Verify: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k"""
    # Build tensors for fast verification
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U_t = U.reshape(n, m, rank)
    V_t = V.reshape(m, p, rank)
    W_t = W.reshape(p, n, rank)
    
    # T[i,j,k] = sum_c U_t[i,j,c] * V_t[j,k,c] * W_t[k,i,c]
    # Use einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    return np.allclose(T, 1.0, atol=1e-9)

def _schoolbook(n, m, p):
    """Standard schoolbook decomposition, rank = n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _build_f4_decomposition():
    """Attempt F(4) rank-25 decomposition for 4x4*4x4.
    
    F(4) algorithm:
    - Split 4x4 matrices into 2x2 blocks
    - Use 5 evaluation points: t = 0, 1, 2, inf, -1
    - At each point, form 2x2 block matrices and multiply
    - Interpolate to recover the 4x4 product
    
    The key insight: C_ij for the full 4x4 product can be expressed as
    a linear combination of 20 scalar products (5 points x 4 entries per 2x2 product),
    but due to the polynomial structure, only 25 independent scalar multiplications
    are needed (using Strassen at each point: 5 x 7 = 35, with 10 dependencies).
    
    For simplicity and correctness, I'll use the direct 5-point construction
    with schoolbook 2x2*2x2 at each point, giving 5*4 = 20 scalar mults,
    but the interpolation weights make some of these cancel, and the total
    number of distinct scalar multiplications in the tensor decomposition is 25.
    
    Actually, let me just use the proven Strassen approach (rank 49) combined
    with schoolbook for the extra column (rank 16), total 65.
    """
    # Fall back to Strassen + schoolbook
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U_e, V_e, W_e = _build_extra_column()
    
    n, m, p = 4, 4, 5
    total_rank = 49 + 16
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    U[:, :49] = U_s
    for j in range(m):
        for k in range(4):
            V[j*p+k, :49] = V_s[j*4+k, :]
    for k in range(4):
        for i in range(n):
            W[k*n+i, :49] = W_s[k*4+i, :]
    
    U[:, 49:] = U_e
    for j in range(m):
        V[j*p+4, 49:] = V_e[j*1+0, :]
    for i in range(n):
        W[4*n+i, 49:] = W_e[0*n+i, :]
    
    return U, V, W, total_rank

def construct():
    n, m, p = 4, 4, 5
    deadline = time.time() + 130
    
    # Build Strassen 4x4x4 (rank 49) + schoolbook 4x4x1 (rank 16) = rank 65
    U_s, V_s, W_s = _build_strassen_4x4_4x4()
    U_e, V_e, W_e = _build_extra_column()
    
    total_rank = 49 + 16
    U = np.zeros((n*m, total_rank), dtype=np.int64)
    V = np.zeros((m*p, total_rank), dtype=np.int64)
    W = np.zeros((p*n, total_rank), dtype=np.int64)
    
    # Strassen part: 4x4 * 4x4 -> 4x4 (first 4 columns of B)
    U[:, :49] = U_s
    for j in range(4):
        for k in range(4):
            V[j*p+k, :49] = V_s[j*4+k, :]
    
    for k in range(4):
        for i in range(n):
            W[k*n+i, :49] = W_s[k*4+i, :]
    
    # Schoolbook part: 4x4 * 4x1 -> 4x1 (5th column of B)
    U[:, 49:] = U_e
    for j in range(m):
        V[j*p+4, 49:] = V_e[j*1+0, :]
    for i in range(n):
        W[4*n+i, 49:] = W_e[0*n+i, :]
    
    # Verify
    if _verify(U, V, W, n, m, p, total_rank):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook (rank 80)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
