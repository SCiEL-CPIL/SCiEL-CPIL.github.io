import time
import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START

def _verify_fast(U, V, W, n, m, p):
    """
    Verify T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    """
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
        return False
    
    U3 = U.reshape(n, m, rank)
    V3 = V.reshape(m, p, rank)
    W3 = W.reshape(p, n, rank)
    
    T_computed = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
    T_expected = np.ones((n, m, p))
    return np.allclose(T_computed, T_expected, atol=1e-9)


def _strassen_2x2_2x2():
    """
    Returns U, V, W (each 4x7) for rank-7 Strassen decomposition of 2x2x2 matmul.
    U index: row*2+col of A (4 entries)
    V index: row*2+col of B (4 entries)
    W index: col*2+row of C (4 entries) -- trace convention: W[k*n+i] for 2x2 means W[k*2+i]
    
    Correct Strassen formulas:
    p1 = (a00+a11)(b00+b11)
    p2 = (a10+a11)(b00)
    p3 = a00(b01-b11)
    p4 = a11(b10-b00)
    p5 = (a00+a01)(b11)
    p6 = (a10-a00)(b00+b01)
    p7 = (a01-a11)(b10+b11)
    
    c00 = p1 - p4 + p5 + p7
    c01 = p2 + p5 - p6 - p7
    c10 = p1 + p3 - p4 - p5
    c11 = p3 + p6 + p4 - p1
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (a00+a11)(b00+b11)
    # U: a00 and a11 -> rows 0,3; V: b00 and b11 -> rows 0,3; W: c00 += 1, c11 -= 1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=-1  # c00 at (0,0), c11 at (1,1) -> W[0*2+0]=1, W[1*2+1]=-1
    
    # P2: (a10+a11)(b00)
    # U: a10 and a11 -> rows 2,3; V: b00 -> row 0; W: c01 += 1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[1,1]=1  # c01 at (0,1) -> W[1*2+0]=1... wait
    
    # Let me re-index W carefully.
    # W[k*n+i] where k=col of C, i=row of C
    # c00: i=0,j=0 -> W[0*2+0]=W[0]
    # c01: i=0,j=1 -> W[1*2+0]=W[2]
    # c10: i=1,j=0 -> W[0*2+1]=W[1]
    # c11: i=1,j=1 -> W[1*2+1]=W[3]
    
    # Let me redo with correct W indices:
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1: (a00+a11)(b00+b11)
    # A: a00(0,0), a11(1,1) -> U[0], U[3]
    # B: b00(0,0), b11(1,1) -> V[0], V[3]
    # C: c00(0,0) += 1, c11(1,1) -= 1 -> W[0*2+0]=W[0]+=1, W[1*2+1]=W[3]-=1
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=-1
    
    # P2: (a10+a11)(b00)
    # A: a10(1,0), a11(1,1) -> U[2], U[3]
    # B: b00(0,0) -> V[0]
    # C: c01(0,1) += 1 -> W[1*2+0]=W[2]+=1
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1
    
    # P3: a00(b01-b11)
    # A: a00(0,0) -> U[0]
    # B: b01(0,1), b11(1,1) -> V[1], V[3]
    # C: c10(1,0) += 1, c11(1,1) += 1 -> W[0*2+1]=W[1]+=1, W[1*2+1]=W[3]+=1
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # P4: a11(b10-b00)
    # A: a11(1,1) -> U[3]
    # B: b10(1,0), b00(0,0) -> V[2], V[0]
    # C: c00(0,0) -= 1, c10(1,0) -= 1, c11(1,1) += 1 -> W[0]-=1, W[1]-=1, W[3]+=1
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=-1; W[1,3]=-1; W[3,3]=1
    
    # P5: (a00+a01)(b11)
    # A: a00(0,0), a01(0,1) -> U[0], U[1]
    # B: b11(1,1) -> V[3]
    # C: c00(0,0) += 1, c01(0,1) += 1, c10(1,0) -= 1 -> W[0]+=1, W[2]+=1, W[1]-=1
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=1; W[2,4]=1; W[1,4]=-1
    
    # P6: (a10-a00)(b00+b01)
    # A: a10(1,0), a00(0,0) -> U[2], U[0]
    # B: b00(0,0), b01(0,1) -> V[0], V[1]
    # C: c01(0,1) -= 1, c11(1,1) += 1 -> W[2]-=1, W[3]+=1
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[2,5]=-1; W[3,5]=1
    
    # P7: (a01-a11)(b10+b11)
    # A: a01(0,1), a11(1,1) -> U[1], U[3]
    # B: b10(1,0), b11(1,1) -> V[2], V[3]
    # C: c00(0,0) += 1, c01(0,1) -= 1, c11(1,1) -= 1 -> W[0]+=1, W[2]-=1, W[3]-=1
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[2,6]=-1; W[3,6]=-1
    
    return U, V, W


def _build_4x4_4x7_strassen():
    """
    Build rank-97 decomposition using Strassen for 4x4x4 part and schoolbook for 4x4x3 part.
    
    For 4x4x4: use 2x2 block Strassen (rank 7) applied to each of the 4 column blocks of B
    that correspond to 4x4x4 structure. Actually, 4x4x4 means B is 4x4, so we use
    Strassen on the 2x2 block structure: 7 rank-7 terms = 49 total.
    
    For 4x4x3: schoolbook, 4*4*3 = 48 terms.
    
    Total: 97.
    """
    n, m, p = 4, 4, 7
    
    U2, V2, W2 = _strassen_2x2_2x2()
    
    # Verify the 2x2 Strassen works
    T2 = np.zeros((2,2,2))
    for r in range(7):
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    T2[i,j,k] += U2[i*2+j,r] * V2[j*2+k,r] * W2[k*2+i,r]
    
    if not np.allclose(T2, np.ones((2,2,2)), atol=1e-9):
        return None
    
    total_rank = 49 + 48
    
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    def add_strassen_block(a_blocks, b_blocks, c_blocks, term_start):
        """
        a_blocks: list of (row_offset, col_offset, sign) for 2x2 A sub-blocks
        b_blocks: list of (row_offset, col_offset, sign) for 2x2 B sub-blocks
        c_blocks: list of (row_offset, col_offset, sign) for 2x2 C sub-blocks
        """
        for r in range(7):
            for ar0, ac0, as_ in a_blocks:
                for ii in range(2):
                    for jj in range(2):
                        val = U2[ii*2+jj, r] * as_
                        if val != 0:
                            gi = ar0 + ii  # global row in A
                            gj = ac0 + jj  # global col in A
                            U[gi*4 + gj, term_start+r] += val
            
            for br0, bc0, bs_ in b_blocks:
                for ii in range(2):
                    for jj in range(2):
                        val = V2[ii*2+jj, r] * bs_
                        if val != 0:
                            gi = br0 + ii  # global row in B
                            gj = bc0 + jj  # global col in B
                            V[gi*7 + gj, term_start+r] += val
            
            for cr0, cc0, cs_ in c_blocks:
                for ii in range(2):
                    for jj in range(2):
                        val = W2[ii*2+jj, r] * cs_
                        if val != 0:
                            gi = cr0 + ii  # global row in C
                            gj = cc0 + jj  # global col in C
                            # W index = k*n + i = col*4 + row
                            W[gj*4 + gi, term_start+r] += val
    
    # Strassen for 4x4 * 4x4 (B columns 0-3, C columns 0-3)
    # Using 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where Bij is 2x2 (B is 4x4, so rows 0-3, cols 0-3)
    # C = [[C11, C12], [C21, C22]] where Cij is 2x2 (C is 4x4, so rows 0-3, cols 0-3)
    
    # Block positions in global indices:
    # A11: rows 0-1, cols 0-1 -> (0,0)
    # A12: rows 0-1, cols 2-3 -> (0,2)
    # A21: rows 2-3, cols 0-1 -> (2,0)
    # A22: rows 2-3, cols 2-3 -> (2,2)
    
    # B11: rows 0-1, cols 0-1 -> (0,0)
    # B12: rows 0-1, cols 2-3 -> (0,2)
    # B21: rows 2-3, cols 0-1 -> (2,0)
    # B22: rows 2-3, cols 2-3 -> (2,2)
    
    # C11: rows 0-1, cols 0-1 -> (0,0)
    # C12: rows 0-1, cols 2-3 -> (0,2)
    # C21: rows 2-3, cols 0-1 -> (2,0)
    # C22: rows 2-3, cols 2-3 -> (2,2)
    
    # M1: (A11+A22)(B11+B22)
    # C11 += M1, C11 -= M4... let me use the correct formulas:
    # c00 = p1 - p4 + p5 + p7
    # c01 = p2 + p5 - p6 - p7
    # c10 = p1 + p3 - p4 - p5
    # c11 = p3 + p6 + p4 - p1
    
    # P1: (A11+A22)(B11+B22)
    add_strassen_block([(0,0,1),(2,2,1)], [(0,0,1),(2,2,1)], [(0,0,1),(2,2,-1)], 0)
    # P2: (A21+A22)(B11)
    add_strassen_block([(2,0,1),(2,2,1)], [(0,0,1)], [(0,2,1)], 7)
    # P3: A11(B12-B22)
    add_strassen_block([(0,0,1)], [(0,2,1),(2,2,-1)], [(2,0,1),(2,2,1)], 14)
    # P4: A22(B21-B11)
    add_strassen_block([(2,2,1)], [(2,0,1),(0,0,-1)], [(0,0,-1),(2,0,-1),(2,2,1)], 21)
    # P5: (A11+A12)(B22)
    add_strassen_block([(0,0,1),(0,2,1)], [(2,2,1)], [(0,0,1),(0,2,1),(2,0,-1)], 28)
    # P6: (A21-A11)(B11+B12)
    add_strassen_block([(2,0,1),(0,0,-1)], [(0,0,1),(0,2,1)], [(0,2,-1),(2,2,1)], 35)
    # P7: (A12-A22)(B21+B22)
    add_strassen_block([(0,2,1),(2,2,-1)], [(2,0,1),(2,2,1)], [(0,0,1),(0,2,-1),(2,2,-1)], 42)
    
    # Schoolbook for 4x4 * 4x3 (B columns 4,5,6 -> C columns 4,5,6)
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[i*4 + j, col] = 1
                V[j*7 + k, col] = 1
                W[k*4 + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _build_schoolbook():
    """Fallback: schoolbook rank 112"""
    n, m, p = 4, 4, 7
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120
    
    # Try the rank-97 Strassen-based decomposition
    try:
        result = _build_4x4_4x7_strassen()
        if result is not None and _verify_fast(result["U"], result["V"], result["W"], n, m, p):
            return result
    except Exception:
        pass
    
    if time.time() > deadline:
        return _build_schoolbook()
    
    # Fallback: schoolbook rank 112
    return _build_schoolbook()

# EVOLVE-BLOCK-END
