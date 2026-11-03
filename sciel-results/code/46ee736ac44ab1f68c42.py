import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def _strassen_2x2x2():
    """Return 7 (u, v, w) tuples for the 2x2x2 matrix multiplication tensor.
    Indices: (i,j) in 0..3 flattened as i*2+j for U, V, W.
    """
    # Standard Strassen decomposition for 2x2 matrices
    # T1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
    # T2: u=(0,0,1,1), v=(1,0,0,0), w=(0,0,1,1)
    # T3: u=(1,0,0,0), v=(0,1,0,-1), w=(0,1,0,-1)
    # T4: u=(0,0,0,1), v=(0,0,1,-1), w=(0,0,1,-1)
    # T5: u=(1,1,0,0), v=(0,0,0,1), w=(1,1,0,0)
    # T6: u=(0,1,-1,0), v=(1,1,0,0), w=(0,1,1,-1)
    # T7: u=(0,1,0,-1), v=(0,0,1,1), w=(0,1,-1,1)
    return [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,-1]),
        ([0,0,0,1], [0,0,1,-1], [0,0,1,-1]),
        ([1,1,0,0], [0,0,0,1], [1,1,0,0]),
        ([0,1,-1,0], [1,1,0,0], [0,1,1,-1]),
        ([0,1,0,-1], [0,0,1,1], [0,1,-1,1]),
    ]

def _verify_2x2x2(terms):
    """Verify that the given Strassen terms correctly decompose 2x2x2 multiplication."""
    # Build the target tensor: T[i,j,k] = 1 if i==k else 0 (for C[i,k] = sum_j A[i,j] B[j,k])
    # Actually, the multiplication tensor for 2x2 is:
    # sum_{i,j,k} C_ik A_ij B_jk
    # So the tensor T has T[i,j,k] = 1 iff i==k (no, that's not right)
    # The tensor is: T[i,j,k] = delta_{i,k}... no.
    # C_ik = sum_j A_ij B_jk
    # So the trilinear form is sum_{i,j,k} C_ik A_ij B_jk
    # The tensor T_{i,j,k} = 1 if the (i,j) entry of A, (j,k) entry of B, and (i,k) entry of C are all used
    # Actually T[i,j,k] = 1 for all i,j,k where i is row of A/C, j is col of A/row of B, k is col of B/C
    # Wait, let me think again. The tensor for matrix multiplication A(n,m) * B(m,p) = C(n,p) is:
    # T[i,j,k] = 1 for i in 0..n-1, j in 0..m-1, k in 0..p-1
    # where i indexes rows of A and C, j indexes cols of A and rows of B, k indexes cols of B and C.
    # For 2x2x2: T is a 2x2x2 tensor with T[i,j,k] = 1 for all i,j,k.
    
    # The rank-1 decomposition is: T = sum_r u_r (o) v_r (o) w_r
    # where (u_r (o) v_r (o) w_r)[i,j,k] = u_r[i] * v_r[j] * w_r[k]
    
    # Let's verify by checking that sum_r u_r[i] * v_r[j] * w_r[k] = 1 for all i,j,k in 0,1
    # Wait no, that's not right either. The tensor for matrix multiplication is:
    # T[i,j,k] = delta_{i,k}? No.
    
    # Let me be very explicit. For 2x2x2:
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = [[c00, c01], [c10, c11]]
    # c00 = a00*b00 + a01*b10
    # c01 = a00*b01 + a01*b11
    # c10 = a10*b00 + a11*b10
    # c11 = a10*b01 + a11*b11
    
    # The trilinear form is: sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k] * C[i,k]
    # where T[i,j,k] = 1 for all i,j,k in {0,1}.
    # No wait, that's not right. Let me think of it as:
    # The tensor T is such that T (o) A (o) B = C (flattened)
    # T is a 3-way tensor of shape (2*2, 2*2, 2*2) = (4,4,4)
    # T[a_idx, b_idx, c_idx] where a_idx = i*2+j, b_idx = j*2+k, c_idx = i*2+k
    # T[a_idx, b_idx, c_idx] = 1 iff a_idx, b_idx, c_idx are consistent with the multiplication
    
    # For 2x2x2:
    # a_idx ranges 0..3 (i*2+j for i,j in 0,1)
    # b_idx ranges 0..3 (j*2+k for j,k in 0,1)
    # c_idx ranges 0..3 (i*2+k for i,k in 0,1)
    
    # T[a_idx, b_idx, c_idx] = 1 iff there exist i,j,k such that:
    # a_idx = i*2+j, b_idx = j*2+k, c_idx = i*2+k
    # And the contribution is A[i,j] * B[j,k] to C[i,k]
    
    # So for each (i,j,k), we set T[i*2+j, j*2+k, i*2+k] = 1
    
    T = np.zeros((4,4,4), dtype=int)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                T[i*2+j, j*2+k, i*2+k] = 1
    
    # Now check: sum_r U[:,r] (o) V[:,r] (o) W[:,r] should equal T
    result = np.zeros((4,4,4), dtype=int)
    for u, v, w in terms:
        result += np.outer(np.outer(u, v), w).reshape(4,4,4)
    
    if not np.array_equal(result, T):
        return False
    return True

def _build_4x4x4_strassen():
    """Build rank-49 decomposition for 4x4x4 using Kronecker product of 2x2x2 Strassen."""
    # The 4x4x4 multiplication can be decomposed as 2x2 blocks of 2x2 matrices.
    # A = [[A00, A01], [A10, A11]] where each Aij is 2x2
    # B = [[B00, B01], [B10, B11]] where each Bjk is 2x2
    # C = A*B = [[A00*B00 + A01*B10, A00*B01 + A01*B11],
    #             [A10*B00 + A11*B10, A10*B01 + A11*B11]]
    
    # Each block product (e.g., A00*B00) is a 2x2x2 multiplication, which has rank 7.
    # So each block product needs 7 rank-1 terms.
    # There are 8 block products, so total rank = 8 * 7 = 56? No wait.
    
    # Actually, the Kronecker product approach:
    # If we have a rank-r decomposition for 2x2x2, then the Kronecker product gives
    # a rank-r^2 decomposition for 4x4x4.
    # Rank 7 for 2x2x2 -> rank 49 for 4x4x4.
    
    strassen_2 = _strassen_2x2x2()
    
    U = []
    V = []
    W = []
    
    for r1, (u1, v1, w1) in enumerate(strassen_2):
        for r2, (u2, v2, w2) in enumerate(strassen_2):
            # The Kronecker product of the rank-1 terms
            u = np.kron(u1, u2)
            v = np.kron(v1, v2)
            w = np.kron(w1, w2)
            U.append(list(u))
            V.append(list(v))
            W.append(list(w))
    
    return U, V, W

def _build_4x4x3_schoolbook():
    """Build rank-48 schoolbook decomposition for 4x4x3."""
    n, m, p = 4, 4, 3
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    return U, V, W

def _verify_full(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    rank = len(U[0])
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    
    # Build the target tensor
    T = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Compute the decomposition
    result = np.zeros((n * m, m * p, p * n), dtype=float)
    for r in range(rank):
        result += np.outer(np.outer(U[:, r], V[:, r]), W[:, r]).reshape(n * m, m * p, p * n)
    
    return np.allclose(result, T, atol=1e-6)

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Try the optimized decomposition: Strassen for 4x4x4 + schoolbook for 4x4x3
    try:
        U1, V1, W1 = _build_4x4x4_strassen()
        U2, V2, W2 = _build_4x4x3_schoolbook()
        
        U = U1 + U2
        V = V1 + V2
        W = W1 + W2
        
        # Verify
        if _verify_full(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
        
    except Exception as e:
        pass
    
    # Fallback to schoolbook
    rank = n * m * p
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = [0] * (n * m)
                v = [0] * (m * p)
                w = [0] * (p * n)
                u[i * m + j] = 1
                v[j * p + k] = 1
                w[k * n + i] = 1
                U.append(u)
                V.append(v)
                W.append(w)
    
    return {"U": U, "V": V, "W": W}
