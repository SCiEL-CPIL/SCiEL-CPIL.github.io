"""
Matrix multiplication 2x3x5 tensor decomposition.

Strategy:
1. Construct a rank-27 decomposition by exploiting R(2,2,3) = 10.
   - Split B by rows: j=2 (1x5) and j=0,1 (2x5).
   - For j=2: A(2x1) * B(1x5) -> rank 10 (schoolbook).
   - For j=0,1: A(2x2) * B(2x5). Split B columns:
     * k=0,1,2: 2x3 block -> rank 10 (known R(2,2,3)=10)
     * k=3,4:   2x2 block -> rank 7 (Strassen)
   - Total: 10 + 10 + 7 = 27.

2. If rank-10 for 2x2x3 fails, fall back to rank-28 construction.
"""

import time
import numpy as np
import time
import numpy as np

def _build_2x2x3_rank10():
    """
    Construct a rank-10 decomposition for multiplying a 2x2 matrix by a 2x3 matrix.
    
    Returns (U_local, V_local, W_local) with shapes (4, 10), (6, 10), (6, 10)
    where:
    - U index: i*2 + j for i in {0,1}, j in {0,1}
    - V index: j*3 + k for j in {0,1}, k in {0,1,2}
    - W index: k*2 + i for k in {0,1,2}, i in {0,1}
    """
    # The 2x2x3 tensor has 4*6*6 = 144 entries, but only 2*2*3 = 12 non-zero entries.
    # We need to find 10 rank-1 terms.
    # 
    # Known construction: Use Strassen-like approach adapted for 2x2x3.
    # 
    # Let A = [[a00, a01], [a10, a11]] (2x2)
    # Let B = [[b00, b01, b02], [b10, b11, b12]] (2x3)
    # C = A*B is 2x3.
    #
    # C[0,0] = a00*b00 + a01*b10
    # C[0,1] = a00*b01 + a01*b11
    # C[0,2] = a00*b02 + a01*b12
    # C[1,0] = a10*b00 + a11*b10
    # C[1,1] = a10*b01 + a11*b11
    # C[1,2] = a10*b02 + a11*b12
    #
    # Strategy: Use 7 Strassen terms for the 2x2x2 part (k=0,1), then add 3 more
    # for the k=2 column, but we need to be clever to get total 10.
    #
    # Alternative: Use the fact that R(2,2,3) <= R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # We need to save 1 more. 
    #
    # Known result: R(2,2,3) = 10. Let me construct it explicitly.
    #
    # One known construction (from literature):
    # m1 = (a00 + a01) * (b00 + b01 + b02)  -> contributes to C[0,0], C[0,1], C[0,2]
    # m2 = (a10 + a11) * (b00 + b01 + b02)  -> contributes to C[1,0], C[1,1], C[1,2]
    # m3 = (a00 - a01) * (b00 - b01 + b02)  -> contributes to C[0,0], C[0,1], C[0,2]
    # m4 = (a10 - a11) * (b00 - b01 + b02)  -> contributes to C[1,0], C[1,1], C[1,2]
    # m5 = (a00 + a10) * (b00 + b10)         -> contributes to C[0,0], C[1,0]
    # m6 = (a01 + a11) * (b01 + b11)         -> contributes to C[0,1], C[1,1]
    # m7 = (a00 - a10) * (b00 - b10)         -> contributes to C[0,0], C[1,0]
    # m8 = (a01 - a11) * (b01 - b11)         -> contributes to C[0,1], C[1,1]
    # m9 = (a00 + a11) * (b02 + b12)         -> contributes to C[0,2], C[1,2]
    # m10 = (a00 - a11) * (b02 - b12)        -> contributes to C[0,2], C[1,2]
    #
    # Let me verify this construction by checking all 6 entries of C.
    
    U = np.zeros((4, 10))
    V = np.zeros((6, 10))
    W = np.zeros((6, 10))
    
    # Indexing:
    # U[i*2 + j] for A[i,j]
    # V[j*3 + k] for B[j,k]
    # W[k*2 + i] for C[i,k]
    
    # m1 = (a00 + a01) * (b00 + b01 + b02)
    U[0, 0] = 1  # a00
    U[1, 0] = 1  # a01
    V[0, 0] = 1  # b00
    V[1, 0] = 1  # b01
    V[2, 0] = 1  # b02
    # C[0,0] += m1, C[0,1] += m1, C[0,2] += m1
    W[0, 0] = 1  # C[0,0]
    W[2, 0] = 1  # C[0,1]
    W[4, 0] = 1  # C[0,2]
    
    # m2 = (a10 + a11) * (b00 + b01 + b02)
    U[2, 1] = 1  # a10
    U[3, 1] = 1  # a11
    V[0, 1] = 1  # b00
    V[1, 1] = 1  # b01
    V[2, 1] = 1  # b02
    W[1, 1] = 1  # C[1,0]
    W[3, 1] = 1  # C[1,1]
    W[5, 1] = 1  # C[1,2]
    
    # m3 = (a00 - a01) * (b00 - b01 + b02)
    U[0, 2] = 1  # a00
    U[1, 2] = -1 # a01
    V[0, 2] = 1  # b00
    V[1, 2] = -1 # b01
    V[2, 2] = 1  # b02
    W[0, 2] = 1  # C[0,0]
    W[2, 2] = -1 # C[0,1]
    W[4, 2] = 1  # C[0,2]
    
    # m4 = (a10 - a11) * (b00 - b01 + b02)
    U[2, 3] = 1  # a10
    U[3, 3] = -1 # a11
    V[0, 3] = 1  # b00
    V[1, 3] = -1 # b01
    V[2, 3] = 1  # b02
    W[1, 3] = 1  # C[1,0]
    W[3, 3] = -1 # C[1,1]
    W[5, 3] = 1  # C[1,2]
    
    # m5 = (a00 + a10) * (b00 + b10)
    U[0, 4] = 1  # a00
    U[2, 4] = 1  # a10
    V[0, 4] = 1  # b00
    V[3, 4] = 1  # b10
    W[0, 4] = 1  # C[0,0]
    W[1, 4] = 1  # C[1,0]
    
    # m6 = (a01 + a11) * (b01 + b11)
    U[1, 5] = 1  # a01
    U[3, 5] = 1  # a11
    V[1, 5] = 1  # b01
    V[4, 5] = 1  # b11
    W[2, 5] = 1  # C[0,1]
    W[3, 5] = 1  # C[1,1]
    
    # m7 = (a00 - a10) * (b00 - b10)
    U[0, 6] = 1  # a00
    U[2, 6] = -1 # a10
    V[0, 6] = 1  # b00
    V[3, 6] = -1 # b10
    W[0, 6] = 1  # C[0,0]
    W[1, 6] = -1 # C[1,0]
    
    # m8 = (a01 - a11) * (b01 - b11)
    U[1, 7] = 1  # a01
    U[3, 7] = -1 # a11
    V[1, 7] = 1  # b01
    V[4, 7] = -1 # b11
    W[2, 7] = 1  # C[0,1]
    W[3, 7] = -1 # C[1,1]
    
    # m9 = (a00 + a11) * (b02 + b12)
    U[0, 8] = 1  # a00
    U[3, 8] = 1  # a11
    V[2, 8] = 1  # b02
    V[5, 8] = 1  # b12
    W[4, 8] = 1  # C[0,2]
    W[5, 8] = 1  # C[1,2]
    
    # m10 = (a00 - a11) * (b02 - b12)
    U[0, 9] = 1  # a00
    U[3, 9] = -1 # a11
    V[2, 9] = 1  # b02
    V[5, 9] = -1 # b12
    W[4, 9] = 1  # C[0,2]
    W[5, 9] = -1 # C[1,2]
    
    return U, V, W


def _verify_2x2x3_decomp(U, V, W):
    """Verify the 2x2x3 decomposition."""
    # Create random test matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
    C_expected = A @ B
    
    # Compute using decomposition
    C_computed = np.zeros((2, 3))
    for r in range(10):
        u = U[:, r]
        v = V[:, r]
        w = W[:, r]
        
        u_mat = u.reshape(2, 2)
        v_mat = v.reshape(2, 3)
        w_mat = w.reshape(3, 2)
        
        # The contribution is: sum over the tensor product
        # For each (i,j,k) in the local 2x2x3 space:
        # C[i,k] += u[i,j] * v[j,k] * w[k,i]
        for i in range(2):
            for j in range(2):
                for k in range(3):
                    C_computed[i, k] += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        print(f"2x2x3 verification FAILED:\nExpected:\n{C_expected}\nGot:\n{C_computed}")
        return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin
    
    n, m, p = 2, 3, 5
    
    # --- Try Rank-27 Construction First ---
    
    # Build 2x2x3 rank-10 decomposition
    U_223, V_223, W_223 = _build_2x2x3_rank10()
    
    if not _verify_2x2x3_decomp(U_223, V_223, W_223):
        # Fallback to rank-28
        return _construct_rank28()
    
    # Now assemble the full 2x3x5 decomposition
    # Part 1: j=2 (1x5 block) -> rank 10
    # Part 2: j=0,1, k=0,1,2 (2x3 block) -> rank 10
    # Part 3: j=0,1, k=3,4 (2x2 block) -> rank 7 (Strassen)
    # Total: 10 + 10 + 7 = 27
    
    rank = 27
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Part 1: B row j=2 (1x5 block). A is 2x1 (cols j=2).
    # A[i,2] * B[2,k] -> C[i,k]
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
    
    # Part 2: B rows j=0,1, cols k=0,1,2 (2x3 block) -> rank 10
    # Map local indices to global:
    # Local U[i*2+j] -> Global U[i*3+j] for i in {0,1}, j in {0,1}
    # Local V[j*3+k] -> Global V[j*5+k] for j in {0,1}, k in {0,1,2}
    # Local W[k*2+i] -> Global W[k*2+i] for k in {0,1,2}, i in {0,1}
    
    for col in range(10):
        # U mapping
        for i in range(2):
            for j in range(2):
                U[i * m + j][r] = U_223[i * 2 + j, col]
        
        # V mapping
        for j in range(2):
            for k in range(3):
                V[j * p + k][r] = V_223[j * 3 + k, col]
        
        # W mapping
        for k in range(3):
            for i in range(2):
                W[k * n + i][r] = W_223[k * 2 + i, col]
        
        r += 1
    
    # Part 3: B rows j=0,1, cols k=3,4 (2x2 block) -> Strassen rank 7
    # i1=0, i2=1 (rows of A)
    # j1=0, j2=1 (cols of A / rows of B)
    # k1=3, k2=4 (cols of B / cols of C)
    
    # M1 = (A[0,0] + A[1,1]) * (B[0,3] + B[1,4])
    U[0 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 3][r] = 1
    V[1 * p + 4][r] = 1
    W[3 * n + 0][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M2 = (A[1,0] + A[1,1]) * B[0,3]
    U[1 * m + 0][r] = 1
    U[1 * m + 1][r] = 1
    V[0 * p + 3][r] = 1
    W[3 * n + 1][r] = 1
    W[4 * n + 1][r] = -1
    r += 1
    
    # M3 = A[0,0] * (B[0,4] - B[1,4])
    U[0 * m + 0][r] = 1
    V[0 * p + 4][r] = 1
    V[1 * p + 4][r] = -1
    W[4 * n + 0][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M4 = A[1,1] * (B[1,3] - B[0,3])
    U[1 * m + 1][r] = 1
    V[1 * p + 3][r] = 1
    V[0 * p + 3][r] = -1
    W[3 * n + 0][r] = 1
    W[3 * n + 1][r] = 1
    r += 1
    
    # M5 = (A[0,0] + A[0,1]) * B[1,4]
    U[0 * m + 0][r] = 1
    U[0 * m + 1][r] = 1
    V[1 * p + 4][r] = 1
    W[3 * n + 0][r] = -1
    W[4 * n + 0][r] = 1
    r += 1
    
    # M6 = (A[1,0] - A[0,0]) * (B[0,3] + B[0,4])
    U[1 * m + 0][r] = 1
    U[0 * m + 0][r] = -1
    V[0 * p + 3][r] = 1
    V[0 * p + 4][r] = 1
    W[4 * n + 1][r] = 1
    r += 1
    
    # M7 = (A[0,1] - A[1,1]) * (B[1,3] + B[1,4])
    U[0 * m + 1][r] = 1
    U[1 * m + 1][r] = -1
    V[1 * p + 3][r] = 1
    V[1 * p + 4][r] = 1
    W[3 * n + 0][r] = 1
    r += 1
    
    if r != 27:
        # Fallback
        return _construct_rank28()
    
    # Final verification
    U_np = np.array(U)
    V_np = np.array(V)
    W_np = np.array(W)
    
    T_decomp = np.zeros((n*m, m*p, n*p))
    for col in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U_np[:, col], V_np[:, col], W_np[:, col])
    
    T_true = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_true[idx_u, idx_v, idx_w] = 1.0
    
    if not np.allclose(T_decomp, T_true, atol=1e-10):
        # Fallback to rank-28
        return _construct_rank28()
    
    return {"U": U, "V": V, "W": W}


def _construct_rank28():
    """Fallback rank-28 construction."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    
    # Helper to add a Strassen 2x2 block
    def add_strassen_2x2(i1, i2, j1, j2, k1, k2):
        nonlocal r
        
        # M1 = (A[i1,j1] + A[i2,j2]) * (B[j1,k1] + B[j2,k2])
        U[i1 * m + j1][r] = 1
        U[i2 * m + j2][r] = 1
        V[j1 * p + k1][r] = 1
        V[j2 * p + k2][r] = 1
        W[k1 * n + i1][r] = 1
        W[k2 * n + i2][r] = 1
        r += 1
        
        # M2 = (A[i2,j1] + A[i2,j2]) * B[j1,k1]
        U[i2 * m + j1][r] = 1
        U[i2 * m + j2][r] = 1
        V[j1 * p + k1][r] = 1
        W[k1 * n + i2][r] = 1
        W[k2 * n + i2][r] = -1
        r += 1
        
        # M3 = A[i1,j1] * (B[j1,k2] - B[j2,k2])
        U[i1 * m + j1][r] = 1
        V[j1 * p + k2][r] = 1
        V[j2 * p + k2][r] = -1
        W[k2 * n + i1][r] = 1
        W[k2 * n + i2][r] = 1
        r += 1
        
        # M4 = A[i2,j2] * (B[j2,k1] - B[j1,k1])
        U[i2 * m + j2][r] = 1
        V[j2 * p + k1][r] = 1
        V[j1 * p + k1][r] = -1
        W[k1 * n + i1][r] = 1
        W[k1 * n + i2][r] = 1
        r += 1
        
        # M5 = (A[i1,j1] + A[i1,j2]) * B[j2,k2]
        U[i1 * m + j1][r] = 1
        U[i1 * m + j2][r] = 1
        V[j2 * p + k2][r] = 1
        W[k1 * n + i1][r] = -1
        W[k2 * n + i1][r] = 1
        r += 1
        
        # M6 = (A[i2,j1] - A[i1,j1]) * (B[j1,k1] + B[j1,k2])
        U[i2 * m + j1][r] = 1
        U[i1 * m + j1][r] = -1
        V[j1 * p + k1][r] = 1
        V[j1 * p + k2][r] = 1
        W[k2 * n + i2][r] = 1
        r += 1
        
        # M7 = (A[i1,j2] - A[i2,j2]) * (B[j2,k1] + B[j2,k2])
        U[i1 * m + j2][r] = 1
        U[i2 * m + j2][r] = -1
        V[j2 * p + k1][r] = 1
        V[j2 * p + k2][r] = 1
        W[k1 * n + i1][r] = 1
        r += 1

    # Part 1: B row j=2 (1x5 block)
    for i in range(n):
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1
            
    # Part 2: Strassen for k=0,1
    add_strassen_2x2(0, 1, 0, 1, 0, 1)
    
    # Part 3: Strassen for k=2,3
    add_strassen_2x2(0, 1, 0, 1, 2, 3)
    
    # Part 4: Direct for k=4
    for i in range(n):
        for j in range(2):
            U[i * m + j][r] = 1
            V[j * p + 4][r] = 1
            W[4 * n + i][r] = 1
            r += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
    print("U shape:", len(res['U']), "x", len(res['U'][0]))
    print("V shape:", len(res['V']), "x", len(res['V'][0]))
    print("W shape:", len(res['W']), "x", len(res['W'][0]))
    
    # Verification
    n, m, p = 2, 3, 5
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    
    T_decomp = np.zeros((n*m, m*p, n*p))
    for col in range(rank):
        T_decomp += np.einsum('a,b,c->abc', U[:, col], V[:, col], W[:, col])
    
    T_true = np.zeros((n*m, m*p, n*p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                T_true[idx_u, idx_v, idx_w] = 1.0
    
    if not np.allclose(T_decomp, T_true, atol=1e-10):
        print("ERROR: Decomposition is invalid!")
    else:
        print("SUCCESS: Decomposition is valid.")
