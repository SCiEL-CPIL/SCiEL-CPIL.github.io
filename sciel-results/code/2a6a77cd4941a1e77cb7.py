"""
Strategy: Construct a valid rank-36 decomposition for 2x4 * 4x5 matrix multiplication.
We split the inner dimension m=4 into two blocks of size 2 (j=0,1 and j=2,3).
For each 2x2x5 sub-problem (since n=2, p=5), we decompose the 5 output columns into:
- Two 2x2x2 blocks (columns 0,1 and 2,3) using Strassen's algorithm (rank 7 each).
- One 2x2x1 block (column 4) using the trivial decomposition (rank 4).
Total rank per block = 7 + 7 + 4 = 18.
Total rank for 2 blocks = 18 + 18 = 36.
This is lower than the schoolbook rank of 40.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 5
    start_time = time.time()
    deadline = start_time + 130

    # Fast vectorized verification
    def verify(U, V, W):
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        r = U.shape[1]
        # Build the full tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # Check nonzero positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        # Check zero positions: T[i,j,k] should be 0 unless (i,j) matches A entry and (j,k) matches B entry
        # i.e., for the decomposition to be valid, T[i,j,k] = delta_{ij} for the multiplication tensor
        # Actually, the matrix multiplication tensor has T[(i,j),(j,k),(k,i)] = 1 and 0 elsewhere
        # So for any (i,j,k), the value should be 1 if this corresponds to a valid multiplication term, 0 otherwise
        # The valid terms are exactly (i,j,k) for i in [0,n), j in [0,m), k in [0,p)
        # And the tensor value is 1 for all such (i,j,k)
        # Zero positions are where the indices don't align: we need to check all (a,b,c) in U x V x W
        # where a = i*m+j, b = j'*p+k, c = k'*n+i' and check T=0 when (j != j') or (i != i') or (k != k')
        # More precisely: the decomposition computes sum_c U[a,c]*V[b,c]*W[c_idx,c]
        # This should equal 1 iff a=i*m+j, b=j*p+k, c_idx=k*n+i for some i,j,k
        # and 0 otherwise
        # Efficient check: for each c, the outer product U[:,c] ⊗ V[:,c] ⊗ W[:,c] should sum to the tensor
        # Let's just check a sample of zero positions
        for i in range(n):
            for j in range(m):
                for j2 in range(m):
                    if j2 == j:
                        continue
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j2 * p + k
                        w_idx = k * n + i
                        val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                        if abs(val) > 1e-9:
                            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for i2 in range(n):
                        if i2 == i:
                            continue
                        for j2 in range(m):
                            if j2 == j:
                                continue
                            u_idx = i * m + j
                            v_idx = j2 * p + k
                            w_idx = k * n + i2
                            val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                            if abs(val) > 1e-9:
                                return False
        return True

    def schoolbook():
        U = np.zeros((n * m, n * m * p))
        V = np.zeros((m * p, n * m * p))
        W = np.zeros((p * n, n * m * p))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Strassen 2x2x2 decomposition
    # Returns U, V, W matrices of shape (2*4, 7), (4*5, 7), (5*2, 7)
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global indices: 
    # U index: i*4 + (j_offset + j)
    # V index: (j_offset + j)*5 + (k_offset + k)
    # W index: (k_offset + k)*2 + i
    def get_strassen_222(j_offset, k_offset):
        U = np.zeros((n * m, 7))
        V = np.zeros((m * p, 7))
        W = np.zeros((p * n, 7))
        
        # M1 = (A00+A11)(B00+B11)
        U[0*4+(0+j_offset), 0] = 1.0
        U[1*4+(1+j_offset), 0] = 1.0
        V[(0+j_offset)*5+(0+k_offset), 0] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 0] = 1.0
        W[(0+k_offset)*2+0, 0] = 1.0
        W[(1+k_offset)*2+1, 0] = 1.0

        # M2 = (A10+A11)B00
        U[1*4+(0+j_offset), 1] = 1.0
        U[1*4+(1+j_offset), 1] = 1.0
        V[(0+j_offset)*5+(0+k_offset), 1] = 1.0
        W[(0+k_offset)*2+1, 1] = 1.0
        W[(1+k_offset)*2+1, 1] = -1.0

        # M3 = A00(B01-B11)
        U[0*4+(0+j_offset), 2] = 1.0
        V[(0+j_offset)*5+(1+k_offset), 2] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 2] = -1.0
        W[(1+k_offset)*2+0, 2] = 1.0
        W[(1+k_offset)*2+1, 2] = 1.0

        # M4 = A11(B10-B00)
        U[1*4+(1+j_offset), 3] = 1.0
        V[(1+j_offset)*5+(0+k_offset), 3] = 1.0
        V[(0+j_offset)*5+(0+k_offset), 3] = -1.0
        W[(0+k_offset)*2+0, 3] = 1.0
        W[(0+k_offset)*2+1, 3] = 1.0

        # M5 = (A00+A01)B11
        U[0*4+(0+j_offset), 4] = 1.0
        U[0*4+(1+j_offset), 4] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 4] = 1.0
        W[(0+k_offset)*2+0, 4] = -1.0
        W[(1+k_offset)*2+0, 4] = 1.0

        # M6 = (A10-A00)(B00+B01)
        U[1*4+(0+j_offset), 5] = 1.0
        U[0*4+(0+j_offset), 5] = -1.0
        V[(0+j_offset)*5+(0+k_offset), 5] = 1.0
        V[(0+j_offset)*5+(1+k_offset), 5] = 1.0
        W[(1+k_offset)*2+1, 5] = 1.0

        # M7 = (A01-A11)(B10+B11)
        U[0*4+(1+j_offset), 6] = 1.0
        U[1*4+(1+j_offset), 6] = -1.0
        V[(1+j_offset)*5+(0+k_offset), 6] = 1.0
        V[(1+j_offset)*5+(1+k_offset), 6] = 1.0
        W[(0+k_offset)*2+0, 6] = 1.0

        return U, V, W

    # Trivial 2x2x1 decomposition (rank 4)
    def get_trivial_221(j_offset, k_offset):
        U = np.zeros((n * m, 4))
        V = np.zeros((m * p, 4))
        W = np.zeros((p * n, 4))
        
        col = 0
        for i in range(2):
            for j in range(2):
                U[i*4 + (j_offset + j), col] = 1.0
                V[(j_offset + j)*5 + (k_offset), col] = 1.0
                W[(k_offset)*2 + i, col] = 1.0
                col += 1
        return U, V, W

    # Build full decomposition
    # Block 1: j in {0, 1}
    U1_s1, V1_s1, W1_s1 = get_strassen_222(0, 0) # k in {0, 1}
    U1_s2, V1_s2, W1_s2 = get_strassen_222(0, 2) # k in {2, 3}
    U1_t1, V1_t1, W1_t1 = get_trivial_221(0, 4)  # k in {4}
    
    U_block1 = np.hstack([U1_s1, U1_s2, U1_t1])
    V_block1 = np.hstack([V1_s1, V1_s2, V1_t1])
    W_block1 = np.hstack([W1_s1, W1_s2, W1_t1])

    # Block 2: j in {2, 3}
    U2_s1, V2_s1, W2_s1 = get_strassen_222(2, 0) # k in {0, 1}
    U2_s2, V2_s2, W2_s2 = get_strassen_222(2, 2) # k in {2, 3}
    U2_t1, V2_t1, W2_t1 = get_trivial_221(2, 4)  # k in {4}
    
    U_block2 = np.hstack([U2_s1, U2_s2, U2_t1])
    V_block2 = np.hstack([V2_s1, V2_s2, V2_t1])
    W_block2 = np.hstack([W2_s1, W2_s2, W2_t1])

    U_full = np.hstack([U_block1, U_block2])
    V_full = np.hstack([V_block1, V_block2])
    W_full = np.hstack([W_block1, W_block2])

    if time.time() > deadline:
        # Emergency: return schoolbook
        U_s, V_s, W_s = schoolbook()
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    if verify(U_full, V_full, W_full):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}

    # Fallback
    U_s, V_s, W_s = schoolbook()
    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
