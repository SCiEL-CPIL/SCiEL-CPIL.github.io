import time
import numpy as np

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    t_start = time.time()
    deadline = t_start + 145  # 5 seconds safety margin
    
    n, m, p = DIMENSIONS
    n_rows_U = n * m  # 4
    n_rows_V = m * p  # 6
    n_rows_W = p * n  # 6
    
    # Target tensor: T[u, v, w] where u = i*m+j, v = j*p+k, w = k'*n+i'
    # Non-zero when i=i' and j=j' and k=k'
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Try rank 11 first (Strassen for 2x2 block + schoolbook for 3rd column)
    # U: (4, 11), V: (6, 11), W: (6, 11)
    
    # Strassen 2x2x2 decomposition (rank 7)
    # A indices: 00, 01, 10, 11 -> U rows 0,1,2,3
    # B indices: 00, 01, 10, 11 -> V rows 0,1,3,4 (column 0,1,2,3 of B)
    # C indices: 00, 01, 10, 11 -> W rows 0,2,1,3 (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    # Wait, W index is k*n+i, so:
    # C[0,0] -> k=0,i=0 -> w=0
    # C[0,1] -> k=0,i=1 -> w=1
    # C[1,0] -> k=1,i=0 -> w=2
    # C[1,1] -> k=1,i=1 -> w=3
    
    # Strassen matrices for 2x2x2:
    # M1: A=[1,0,0,1], B=[1,0,0,1], C=[1,0,0,0]
    # M2: A=[0,1,1,0], B=[1,0,0,0], C=[0,0,1,0]
    # M3: A=[1,0,0,0], B=[0,1,0,-1], C=[0,1,0,0]
    # M4: A=[0,0,0,1], B=[0,0,1,-1], C=[-1,0,0,0]
    # M5: A=[1,1,0,0], B=[0,0,0,1], C=[1,0,0,0]
    # M6: A=[0,1,-1,0], B=[1,1,0,0], C=[0,0,0,1]
    # M7: A=[0,1,0,-1], B=[0,1,1,1], C=[0,1,0,0]
    
    # Let me use the standard Strassen formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For our problem, B is 2x3, so B has columns 0,1,2
    # Strassen uses B columns 0 and 1 (B00,B01,B10,B11)
    # Schoolbook handles column 2 (B02, B12)
    
    # U rows: i*m+j: A00=0, A01=1, A10=2, A11=3
    # V rows: j*p+k: B00=0, B01=1, B02=2, B10=3, B11=4, B12=5
    # W rows: k*n+i: C00=0, C01=1, C10=2, C11=3, C20=4, C21=5
    
    # Rank 11 construction:
    # Columns 0-6: Strassen for 2x2 block (using B cols 0,1 -> C cols 0,1)
    # Columns 7-8: Schoolbook for 3rd column (B col 2 -> C col 2)
    
    U = np.zeros((4, 11), dtype=float)
    V = np.zeros((6, 11), dtype=float)
    W = np.zeros((6, 11), dtype=float)
    
    # Strassen part (columns 0-6)
    # M1: A=[1,0,0,1], B=[1,0,0,1], C=[1,0,0,0]
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[4,0] = 1
    W[0,0] = 1
    
    # M2: A=[0,1,1,0], B=[1,0,0,0], C=[0,0,1,0]
    U[1,1] = 1; U[2,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    # M3: A=[1,0,0,0], B=[0,1,0,-1], C=[0,1,0,0]
    U[0,2] = 1
    V[1,2] = 1; V[4,2] = -1
    W[1,2] = 1
    
    # M4: A=[0,0,0,1], B=[0,0,1,-1], C=[-1,0,0,0]
    U[3,3] = 1
    V[3,3] = 1; V[0,3] = -1
    W[0,3] = -1
    
    # M5: A=[1,1,0,0], B=[0,0,0,1], C=[1,0,0,0]
    U[0,4] = 1; U[1,4] = 1
    V[4,4] = 1
    W[0,4] = 1
    
    # M6: A=[0,1,-1,0], B=[1,1,0,0], C=[0,0,0,1]
    U[1,5] = 1; U[2,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    # M7: A=[0,1,0,-1], B=[0,1,1,1], C=[0,1,0,0]
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1; V[4,6] = 1
    W[1,6] = 1
    
    # Schoolbook part (columns 7-8) for 3rd column of B
    # A00 * B02 -> C02: U row 0, V row 2, W row 4
    U[0,7] = 1
    V[2,7] = 1
    W[4,7] = 1
    
    # A10 * B12 -> C12: U row 2, V row 5, W row 5
    U[2,8] = 1
    V[5,8] = 1
    W[5,8] = 1
    
    # Wait, that's only 2 more, giving rank 9? No, 7 + 2 = 9.
    # But I need rank 11. Let me reconsider.
    
    # Actually, for the 3rd column, we need:
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2]
    
    # So we need 4 terms for schoolbook on 3rd column, but we can potentially share.
    # Let me just do full schoolbook for the 3rd column:
    # A00*B02 -> C02: U row 0, V row 2, W row 4
    # A01*B12 -> C02: U row 1, V row 5, W row 4
    # A10*B02 -> C12: U row 2, V row 2, W row 5
    # A11*B12 -> C12: U row 3, V row 5, W row 5
    
    # That's 4 more terms, total rank = 7 + 4 = 11. Let me redo.
    
    # Reset
    U = np.zeros((4, 11), dtype=float)
    V = np.zeros((6, 11), dtype=float)
    W = np.zeros((6, 11), dtype=float)
    
    # Strassen part (columns 0-6)
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[4,0] = 1
    W[0,0] = 1
    
    U[1,1] = 1; U[2,1] = 1
    V[0,1] = 1
    W[2,1] = 1
    
    U[0,2] = 1
    V[1,2] = 1; V[4,2] = -1
    W[1,2] = 1
    
    U[3,3] = 1
    V[3,3] = 1; V[0,3] = -1
    W[0,3] = -1
    
    U[0,4] = 1; U[1,4] = 1
    V[4,4] = 1
    W[0,4] = 1
    
    U[1,5] = 1; U[2,5] = -1
    V[0,5] = 1; V[1,5] = 1
    W[3,5] = 1
    
    U[1,6] = 1; U[3,6] = -1
    V[1,6] = 1; V[3,6] = 1; V[4,6] = 1
    W[1,6] = 1
    
    # Schoolbook part (columns 7-10) for 3rd column
    # A00*B02 -> C02
    U[0,7] = 1
    V[2,7] = 1
    W[4,7] = 1
    
    # A01*B12 -> C02
    U[1,8] = 1
    V[5,8] = 1
    W[4,8] = 1
    
    # A10*B02 -> C12
    U[2,9] = 1
    V[2,9] = 1
    W[5,9] = 1
    
    # A11*B12 -> C12
    U[3,10] = 1
    V[5,10] = 1
    W[5,10] = 1
    
    # Verify
    def verify(U, V, W, n, m, p):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    expected = 1.0
                    actual = 0.0
                    for r in range(rank):
                        actual += U[u, r] * V[v, r] * W[w, r]
                    if abs(actual - expected) > 1e-9:
                        return False
        # Check that all other entries are zero
        for u in range(n * m):
            for v in range(m * p):
                for w in range(p * n):
                    # Determine if this is a valid (i,j,k) triple
                    i = u // m
                    j = u % m
                    j_v = v // p
                    k = v % p
                    k_w = w // n
                    i_w = w % n
                    if i != i_w or j != j_v or k != k_w:
                        actual = 0.0
                        for r in range(rank):
                            actual += U[u, r] * V[v, r] * W[w, r]
                        if abs(actual) > 1e-9:
                            return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to schoolbook rank 12
    rank = 12
    U = np.zeros((4, 12), dtype=float)
    V = np.zeros((6, 12), dtype=float)
    W = np.zeros((6, 12), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
