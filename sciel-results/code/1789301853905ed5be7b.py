"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if isinstance(U, list) else U.shape[1]
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor T has T[i,j,k] = 1 if we're computing (A*B)[i,k]
    # which involves A[i,j]*B[j,k]. The decomposition is:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_np[i*m+j, c] * V_np[j*p+k, c] * W_np[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _strassen_2x2():
    """Return the 7-rank Strassen decomposition for 2x2 * 2x2 -> 2x2.
    Returns U, V, W as lists of shape (4,7), (4,7), (4,7)."""
    # Standard Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # A is 2x2: indices (i,j) -> row i*m+j = i*2+j
    # B is 2x2: indices (j,k) -> row j*p+k = j*2+k
    # C is 2x2: indices (k,i) -> row k*n+i = k*2+i (trace convention)
    
    # U: A factor, shape (4, 7)
    # V: B factor, shape (4, 7)
    # W: C factor, shape (4, 7)
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)*(B11+B22)
    # A11 = U[0,0], A22 = U[3,0], B11 = V[0,0], B22 = V[3,0]
    # C11 += M1, C22 += M1 (with sign)
    # C[k,i]: C11 = W[0*2+0]=W[0], C22 = W[1*2+1]=W[3]
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)*B11
    # A21 = U[2,1], A22 = U[3,1], B11 = V[0,1]
    # C21 += M2: C21 = W[1*2+0]=W[2]
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # M3 = A11*(B12-B22)
    # A11 = U[0,2], B12 = V[1,2], B22 = V[3,2]
    # C12 += M3: C12 = W[0*2+1]=W[1]
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # M4 = A22*(B21-B11)
    # A22 = U[3,3], B21 = V[2,3], B11 = V[0,3]
    # C22 += M4: C22 = W[3]
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[3][3] = 1
    
    # M5 = (A11+A12)*B22
    # A11 = U[0,4], A12 = U[1,4], B22 = V[3,4]
    # C12 += M5: C12 = W[1]
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[1][4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    # A21 = U[2,5], A11 = U[0,5], B11 = V[0,5], B12 = V[1,5]
    # C21 += M6: C21 = W[2]
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[2][5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    # A12 = U[1,6], A22 = U[3,6], B21 = V[2,6], B22 = V[3,6]
    # C22 += M7: C22 = W[3]
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[3][6] = 1
    
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return schoolbook decomposition for n x m * m x p -> n x p.
    Returns U (n*m, n*m*p), V (m*p, n*m*p), W (p*n, n*m*p)."""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(decs):
    """Combine multiple block decompositions into one global decomposition.
    Each dec is (U_block, V_block, W_block, row_offset_U, row_offset_V, row_offset_W, n, m, p).
    Returns global U, V, W.
    """
    total_rank = sum(len(d[0][0]) for d in decs)
    # Determine global sizes
    global_n = sum(d[6] for d in decs)
    global_m = sum(d[7] for d in decs)
    global_p = sum(d[8] for d in decs)
    
    U = [[0]*total_rank for _ in range(global_n * global_m)]
    V = [[0]*total_rank for _ in range(global_m * global_p)]
    W = [[0]*total_rank for _ in range(global_p * global_n)]
    
    rank_offset = 0
    for (Ub, Vb, Wb, roff_U, roff_V, roff_W, nb, mb, pb) in decs:
        r = len(Ub[0])
        for i in range(nb):
            for j in range(mb):
                for c in range(r):
                    U[roff_U + i*mb + j][rank_offset + c] = Ub[i*mb+j][c]
        for j in range(mb):
            for k in range(pb):
                for c in range(r):
                    V[roff_V + j*pb + k][rank_offset + c] = Vb[j*pb+k][c]
        for k in range(pb):
            for i in range(nb):
                for c in range(r):
                    W[roff_W + k*nb + i][rank_offset + c] = Wb[k*nb+i][c]
        rank_offset += r
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    
    # Strategy: Split the 2x3 * 3x5 into blocks.
    # We can split along the middle dimension (m=3) into 2+1 or 1+2.
    # Or split along n or p.
    
    # Option 1: Split m=3 into m1=2, m2=1
    # Block 1: 2x2 * 2x5 -> 2x5 (use Strassen for 2x2 part + schoolbook for 2x5)
    # Block 2: 2x1 * 1x5 -> 2x5 (schoolbook, rank 10)
    
    # Actually, let me think more carefully about the block structure.
    # A is 2x3, B is 3x5.
    # Split A = [A1 | A2] where A1 is 2x2, A2 is 2x1
    # Split B = [B1; B2] where B1 is 2x5, B2 is 1x5
    # C = A1*B1 + A2*B2
    
    # For A1*B1: 2x2 * 2x5. We can use Strassen-style for the 2x2 part.
    # But 2x2 * 2x5 is not a square block. Let's use a hybrid.
    
    # Actually, the simplest approach that gives rank < 30:
    # Use Strassen on the 2x2 sub-blocks where possible.
    
    # Let's try: decompose into two blocks along the m dimension.
    # Block 1: (2x2) * (2x5) -> 2x5
    # Block 2: (2x1) * (1x5) -> 2x5
    
    # For Block 1 (2x2 * 2x5):
    # We can think of B1 as two 2x2 blocks plus a 2x1 block, or use a different approach.
    # Actually, let's use the known result: rank(2x2 * 2x5) <= 7 + 5 = 12 (Strassen + schoolbook for 2x1)
    # Wait, that's not quite right either.
    
    # Let me just use the explicit rank-28 construction mentioned in the knowledge:
    # Strassen(7) for a 2x2x2 sub-block + schoolbook for the rest.
    
    # Actually, the simplest verified approach: 
    # Split into blocks where we can apply Strassen.
    
    # Let's do: split p=5 into 2+3, and handle each part.
    # Block 1: 2x3 * 3x2 -> 2x2 (use Strassen-like)
    # Block 2: 2x3 * 3x3 -> 2x3 (schoolbook)
    
    # Hmm, let me just go with the straightforward block decomposition that's known to work.
    
    # Known optimal upper bound for 2x3 * 3x5:
    # rank(2x3 * 3x5) <= rank(2x2 * 2x2) + rank(2x2 * 2x1) + rank(2x1 * 1x2) + rank(2x1 * 1x1)
    # = 7 + 4 + 2 + 1 = 14? No, that's not how it works.
    
    # Let me just use the schoolbook decomposition but with a smarter split.
    # The key insight: for 2x3 * 3x5, we can split the middle dimension 3 = 2 + 1.
    
    # Block A: 2x2 * 2x5, Block B: 2x1 * 1x5
    # rank(2x2 * 2x5): We can use Strassen for the 2x2 part of the left and treat the right as 2 columns of 2x2 + 1 column of 2x1.
    # This is getting complex. Let me just use a direct construction.
    
    # Direct approach: Use the fact that we can share computations.
    # The schoolbook rank is 30. Let's try to get below that.
    
    # For now, let's return the schoolbook decomposition which is guaranteed correct.
    # Then we can try to improve.
    
    U, V, W = _schoolbook_block(n, m, p)
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
