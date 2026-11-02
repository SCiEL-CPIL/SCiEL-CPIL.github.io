import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def _verify(U, V, W, n=2, m=4, p=8, trials=50, tol=1e-9):
    """Verify that the decomposition correctly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for _ in range(trials):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for c in range(rank):
            factor_A = np.zeros(n)
            for i in range(n):
                factor_A[i] = sum(A[i, j] * U[i * m + j][c] for j in range(m))
            factor_B = np.zeros(p)
            for k in range(p):
                factor_B[k] = sum(V[j * p + k][c] * B[j, k] for j in range(m))
            for i in range(n):
                for k in range(p):
                    C_computed[i, k] += factor_A[i] * factor_B[k] * W[k * n + i][c]

        if np.max(np.abs(C_computed - C_expected)) > tol:
            return False
    return True

def _build_schoolbook(n=2, m=4, p=8):
    """Schoolbook decomposition, rank n*m*p."""
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
    return U, V, W

def _build_rank56(n=2, m=4, p=8):
    """
    Rank 56 construction using Strassen on 2x2x2 blocks.
    Splits B into two 4x4 blocks.
    Each 4x4 block is split into 2x2 sub-blocks.
    Total 8 Strassen blocks of 2x2x2.
    8 * 7 = 56.
    """
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients
    # m1 = (a+d)(e+h) -> U: a:1, d:1; V: e:1, h:1; W: C00:1, C11:1
    # m2 = (c+d)e   -> U: c:1, d:1; V: e:1; W: C10:1, C11:-1
    # m3 = a(f-h)   -> U: a:1; V: f:1, h:-1; W: C01:1, C11:1
    # m4 = d(g-e)   -> U: d:1; V: g:1, e:-1; W: C00:1, C10:1
    # m5 = (a+b)h   -> U: a:1, b:1; V: h:1; W: C00:-1, C01:1
    # m6 = (c-a)(e+f) -> U: c:1, a:-1; V: e:1, f:1; W: C11:1
    # m7 = (b-d)(g+h) -> U: b:1, d:-1; V: g:1, h:1; W: C00:1, C11:1
    
    # Local indices:
    # A: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
    # B: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
    # C: (0,0), (0,1), (1,0), (1,1)
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # m1
        U[i_base*4 + j_base + 0][col] += 1  # a
        U[i_base*4 + j_base + 1][col] += 1  # d? No, d is (1,1)
        # Wait, my indexing:
        # A row i_base, col j_base.
        # a = A[i_base, j_base]
        # b = A[i_base, j_base+1]
        # c = A[i_base+1, j_base]
        # d = A[i_base+1, j_base+1]
        
        # Correction:
        # U index for A[i,j] is i*m + j. Here m=4.
        # i_global = i_base + i_local
        # j_global = j_base + j_local
        
        # m1: (a+d)(e+h)
        U[(i_base+0)*4 + (j_base+0)][col] += 1 # a
        U[(i_base+1)*4 + (j_base+1)][col] += 1 # d
        V[(j_base+0)*8 + (k_base+0)][col] += 1 # e
        V[(j_base+1)*8 + (k_base+1)][col] += 1 # h
        W[(k_base+0)*2 + (i_base+0)][col] += 1 # C00
        W[(k_base+1)*2 + (i_base+1)][col] += 1 # C11
        col += 1
        
        # m2: (c+d)e
        U[(i_base+1)*4 + (j_base+0)][col] += 1 # c
        U[(i_base+1)*4 + (j_base+1)][col] += 1 # d
        V[(j_base+0)*8 + (k_base+0)][col] += 1 # e
        W[(k_base+0)*2 + (i_base+1)][col] += 1 # C10
        W[(k_base+1)*2 + (i_base+1)][col] += -1 # C11
        col += 1
        
        # m3: a(f-h)
        U[(i_base+0)*4 + (j_base+0)][col] += 1 # a
        V[(j_base+0)*8 + (k_base+1)][col] += 1 # f
        V[(j_base+1)*8 + (k_base+1)][col] += -1 # h
        W[(k_base+1)*2 + (i_base+0)][col] += 1 # C01
        W[(k_base+1)*2 + (i_base+1)][col] += 1 # C11
        col += 1
        
        # m4: d(g-e)
        U[(i_base+1)*4 + (j_base+1)][col] += 1 # d
        V[(j_base+1)*8 + (k_base+0)][col] += 1 # g
        V[(j_base+0)*8 + (k_base+0)][col] += -1 # e
        W[(k_base+0)*2 + (i_base+0)][col] += 1 # C00
        W[(k_base+0)*2 + (i_base+1)][col] += 1 # C10
        col += 1
        
        # m5: (a+b)h
        U[(i_base+0)*4 + (j_base+0)][col] += 1 # a
        U[(i_base+0)*4 + (j_base+1)][col] += 1 # b
        V[(j_base+1)*8 + (k_base+1)][col] += 1 # h
        W[(k_base+0)*2 + (i_base+0)][col] += -1 # C00
        W[(k_base+1)*2 + (i_base+0)][col] += 1 # C01
        col += 1
        
        # m6: (c-a)(e+f)
        U[(i_base+1)*4 + (j_base+0)][col] += 1 # c
        U[(i_base+0)*4 + (j_base+0)][col] += -1 # a
        V[(j_base+0)*8 + (k_base+0)][col] += 1 # e
        V[(j_base+0)*8 + (k_base+1)][col] += 1 # f
        W[(k_base+1)*2 + (i_base+1)][col] += 1 # C11
        col += 1
        
        # m7: (b-d)(g+h)
        U[(i_base+0)*4 + (j_base+1)][col] += 1 # b
        U[(i_base+1)*4 + (j_base+1)][col] += -1 # d
        V[(j_base+1)*8 + (k_base+0)][col] += 1 # g
        V[(j_base+1)*8 + (k_base+1)][col] += 1 # h
        W[(k_base+0)*2 + (i_base+0)][col] += 1 # C00
        W[(k_base+1)*2 + (i_base+1)][col] += 1 # C11
        col += 1

    # We have 2 blocks of B (cols 0-3 and 4-7)
    # For each B block, we have 4 Strassen sub-blocks (2x2 A cols x 2x2 B rows)
    # A cols: 0-1 (j_base=0), 2-3 (j_base=2)
    # B rows: 0-1 (j_base=0), 2-3 (j_base=2)
    # B cols: 0-1 (k_base=0), 2-3 (k_base=2)
    
    for b_block in range(2): # 0: cols 0-3, 1: cols 4-7
        k_offset = b_block * 4
        for j_A_block in range(2): # 0: A cols 0-1, 1: A cols 2-3
            j_A_offset = j_A_block * 2
            for j_B_block in range(2): # 0: B rows 0-1, 1: B rows 2-3
                j_B_offset = j_B_block * 2
                for k_B_block in range(2): # 0: B cols 0-1, 1: B cols 2-3
                    k_B_offset = k_B_block * 2
                    
                    i_base = 0
                    j_base = j_A_offset
                    k_base = k_offset + k_B_offset
                    # Note: j in V index is the row of B, which corresponds to col of A.
                    # In our Strassen block, j_local is the inner dimension.
                    # So j_base for V should be j_B_offset?
                    # Yes, the inner dimension j is shared.
                    # A col index = j_global. B row index = j_global.
                    # So j_base must be consistent.
                    # In the loop above, I used j_base for both A col and B row.
                    # j_A_offset is A col offset. j_B_offset is B row offset.
                    # They MUST be the same for the multiplication to make sense.
                    # Wait.
                    # C[i,k] = sum_j A[i,j] B[j,k].
                    # If we split j into blocks, we sum over those blocks.
                    # Block 1: j in 0,1. A[:, 0:2] * B[0:2, :].
                    # Block 2: j in 2,3. A[:, 2:4] * B[2:4, :].
                    # So j_A_offset must equal j_B_offset.
                    
                    # My loop structure:
                    # j_A_block and j_B_block are independent in the loop above?
                    # No, I wrote:
                    # for j_A_block in range(2):
                    #    for j_B_block in range(2):
                    # This allows j_A=0, j_B=2. That would be A[:, 0:2] * B[2:4, :].
                    # This term is NOT part of the standard multiplication unless we are doing a block sum.
                    # Standard multiplication sums A_L B_L + A_R B_R.
                    # It does NOT include A_L B_R.
                    # So I should only iterate where j_A_offset == j_B_offset.
                    
                    if j_A_offset == j_B_offset:
                        add_strassen_block(0, j_A_offset, k_base)

    assert col == rank
    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Try Rank 56
    try:
        U, V, W = _build_rank56(n, m, p)
        if _verify(U, V, W, n, m, p, trials=10, tol=1e-9):
            print("Rank-56 verified")
            return {"U": U, "V": V, "W": W}
    except Exception as e:
        print(f"Rank 56 failed: {e}")
        
    # Fallback to Schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    if _verify(U, V, W, n, m, p, trials=10, tol=1e-9):
        print("Schoolbook verified")
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
