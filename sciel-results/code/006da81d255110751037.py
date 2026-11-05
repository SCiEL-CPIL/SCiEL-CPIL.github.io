import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    start_time = time.time()
    deadline = start_time + 110.0

    def time_left():
        return deadline - time.time()

    # Build target tensor T[i,j,k] = 1 if output (i,k) gets contribution from A[i,j]*B[j,k]
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0

    def check_decomposition(U, V, W, T, tol=1e-9):
        """Check if U,V,W exactly decompose T"""
        T_rec = np.einsum('ir,vr,wr->ivw', U, V, W, optimize=True)
        return np.max(np.abs(T_rec - T)) < tol

    # Rank-28 decomposition using Strassen + schoolbook
    def build_rank_28():
        rank = 28
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0

        def add_strassen(i1, i2, j1, j2, k1, k2):
            """Add 7 Strassen terms for 2x2 * 2x2 block"""
            nonlocal col
            # M1 = (A00 + A11)(B00 + B11)
            U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M2 = (A10 + A11) B00
            U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
            V[j1*p+k1, col] = 1
            W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
            col += 1
            # M3 = A00 (B01 - B11)
            U[i1*m+j1, col] = 1
            V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
            W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
            col += 1
            # M4 = A11 (B10 - B00)
            U[i2*m+j2, col] = 1
            V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
            W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
            col += 1
            # M5 = (A00 + A01) B11
            U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
            V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
            col += 1
            # M6 = (A10 - A00)(B00 + B01)
            U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
            V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
            W[k2*n+i2, col] = 1
            col += 1
            # M7 = (A01 - A11)(B10 + B11)
            U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
            V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
            W[k1*n+i1, col] = 1
            col += 1

        def add_outer(i, j, k):
            """Add one schoolbook term"""
            nonlocal col
            U[i*m+j, col] = 1
            V[j*p+k, col] = 1
            W[k*n+i, col] = 1
            col += 1

        # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (7 terms)
        add_strassen(0, 1, 0, 1, 0, 1)
        # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> Strassen (7 terms)
        add_strassen(0, 1, 0, 1, 2, 3)
        # Remaining schoolbook terms:
        # j=2, k in {0,1}: 2 terms
        for i in range(n):
            for k in range(2):
                add_outer(i, 2, k)
        # j in {0,1}, k=4: 4 terms
        for j in range(2):
            for i in range(n):
                add_outer(i, j, 4)
        # j=2, k in {2,3,4}: 6 terms
        for i in range(n):
            for k in range(2, 5):
                add_outer(i, 2, k)

        assert col == rank, f"Expected rank {rank}, got {col}"
        return U, V, W

    # Try to find rank-27 by searching for a better decomposition
    def try_find_lower_rank():
        """Try various strategies to find rank < 28"""
        if time_left() < 30:
            return None
        
        # Strategy: Use a 2x2x3 block with rank 8 instead of schoolbook 12
        # For multiplying 2x2 by 2x3, there's an 8-term algorithm
        # Let's try partitioning differently
        
        # Alternative partition:
        # Block A: i in {0,1}, j in {0,1}, k in {0,1} -> Strassen (7)
        # Block B: i in {0,1}, j in {0,1}, k in {2,3,4} -> 2x2x3, try rank 8
        # Block C: i in {0,1}, j=2, k in {0,1,2,3,4} -> schoolbook (10)
        # Total: 7 + 8 + 10 = 25 if 2x2x3 rank 8 works!
        
        # Actually, let me verify: the 2x2x3 matrix multiplication tensor
        # C[i,k] = sum_j A[i,j] * B[j,k] for i in {0,1}, j in {0,1}, k in {0,1,2}
        # This is the tensor for multiplying 2x2 by 2x3.
        # The minimal rank is known to be 8.
        
        # Let me construct an 8-term decomposition for 2x2 * 2x3
        def build_2x2x3_rank8(i1, i2, j1, j2, k1, k2, k3):
            """8-term decomposition for 2x2 * 2x3"""
            terms = []
            
            # We can think of this as:
            # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
            # 
            # One approach: use Strassen-like ideas
            # Let's define:
            # M1 = (A00 + A11)(B00 + B10)  -> contributes to C00, C10, C01, C11
            # M2 = (A10 + A11)B00          -> contributes to C10, C11
            # M3 = A00(B01 - B11)          -> contributes to C01, C02
            # M4 = A11(B10 - B00)          -> contributes to C00, C10
            # M5 = (A00 + A01)B11          -> contributes to C01, C02
            # M6 = (A10 - A00)(B00 + B01)  -> contributes to C11, C12
            # M7 = (A01 - A11)(B10 + B11)  -> contributes to C00, C01
            # M8 = A10 * B12               -> contributes to C12
            # 
            # This is getting complex. Let me use a known 8-term algorithm.
            # 
            # Actually, for 2x2 * 2x3, we can use:
            # Treat it as two 2x2 * 2x1 problems plus one 2x1 * 2x1
            # Or use the following 8-term scheme:
            
            # Let me use a systematic approach with linear algebra
            # The tensor has 2*2*3 = 12 non-zero entries
            # We need 8 factors
            
            # Known 8-term algorithm for 2x2 * 2x3:
            # M1 = (A00 + A11)(B00 + B10)
            # M2 = (A10 + A11)B00  
            # M3 = A00(B01 - B11)
            # M4 = A11(B10 - B00)
            # M5 = (A00 + A01)B11
            # M6 = (A10 - A00)(B00 + B01)
            # M7 = (A01 - A11)(B10 + B11)
            # M8 = A10 * B12
            
            # Let me verify this works...
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C02 = M3 (from A00*B02? No, B has only cols 0,1 in this subproblem)
            
            # I think I'm overcomplicating this. Let me just use schoolbook for now
            # and see if the overall structure gives rank 28, then try to optimize.
            
            return None  # Placeholder
        
        # For now, let's stick with the rank-28 solution which is verified
        return None

    # Build the rank-28 decomposition
    U28, V28, W28 = build_rank_28()
    
    if check_decomposition(U28, V28, W28, T):
        # Verify and return
        return {"U": U28.tolist(), "V": V28.tolist(), "W": W28.tolist()}
    
    # Fallback to schoolbook rank 30
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
