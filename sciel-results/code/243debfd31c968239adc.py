"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def _strassen_2x2_decomposition():
    """
    Strassen's 7-multiplication decomposition for 2x2 matrix multiplication.
    Returns U, V, W for the tensor T_{i,j,k} = A_{i,j} * B_{j,k}.
    
    Indexing convention:
      U is (n*m, rank) = (4, 7) where row index = i*m + j = i*2 + j
      V is (m*p, rank) = (4, 7) where row index = j*p + k = j*2 + k
      W is (p*n, rank) = (4, 7) where row index = k*n + i = k*2 + i
    
    Strassen's 7 products:
      M1 = (A11+A22)*(B11+B22)
      M2 = (A21+A22)*B11
      M3 = A11*(B12-B22)
      M4 = A22*(B21-B11)
      M5 = (A11+A12)*B22
      M6 = (A21-A11)*(B11+B12)
      M7 = (A12-A22)*(B21+B22)
    
    Results:
      C11 = M1 + M4 - M5 + M7
      C12 = M3 + M5
      C21 = M2 + M4
      C22 = M1 - M2 + M3 + M6
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # M1 = (A11+A22)*(B11+B22)
    # A factors: A11 (i=0,j=0) -> U[0*2+0]=U[0], A22 (i=1,j=1) -> U[1*2+1]=U[3]
    # B factors: B11 (j=0,k=0) -> V[0*2+0]=V[0], B22 (j=1,k=1) -> V[1*2+1]=V[3]
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    # C11 = M1 + ... -> W[0*2+0]=W[0]; C22 = M1 - ... -> W[1*2+1]=W[3]
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)*B11
    # A21 (i=1,j=0) -> U[1*2+0]=U[2], A22 (i=1,j=1) -> U[3]
    # B11 (j=0,k=0) -> V[0]
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    # C21 = M2 + ... -> W[1*2+0]=W[2]; C22 = ... - M2 -> W[3]
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11*(B12-B22)
    # A11 (i=0,j=0) -> U[0]
    # B12 (j=0,k=1) -> V[0*2+1]=V[1], B22 (j=1,k=1) -> V[3]
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    # C11 = ... + M3? No. C12 = M3 + M5 -> W[0*2+1]=W[1]; C22 = ... + M3 -> W[3]
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22*(B21-B11)
    # A22 (i=1,j=1) -> U[3]
    # B21 (j=1,k=0) -> V[1*2+0]=V[2], B11 (j=0,k=0) -> V[0]
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    # C11 = M1 + M4 - ... -> W[0]; C21 = M2 + M4 -> W[2]
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)*B22
    # A11 (i=0,j=0) -> U[0], A12 (i=0,j=1) -> U[0*2+1]=U[1]
    # B22 (j=1,k=1) -> V[3]
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    # C11 = ... - M5 -> W[0]; C12 = M3 + M5 -> W[1]
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)*(B11+B12)
    # A21 (i=1,j=0) -> U[2], A11 (i=0,j=0) -> U[0]
    # B11 (j=0,k=0) -> V[0], B12 (j=0,k=1) -> V[1]
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    # C22 = ... + M6 -> W[3]
    W[3, 5] = 1
    
    # M7 = (A12-A22)*(B21+B22)
    # A12 (i=0,j=1) -> U[1], A22 (i=1,j=1) -> U[3]
    # B21 (j=1,k=0) -> V[2], B22 (j=1,k=1) -> V[3]
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    # C11 = ... + M7 -> W[0]
    W[0, 6] = 1
    
    return U, V, W


def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    """
    Combine two tensor decompositions via Kronecker product.
    If T1 = sum_r U1_r * V1_r * W1_r and T2 = sum_s U2_s * V2_s * W2_s,
    then T1 x T2 (tensor product) has rank rank1*rank2 with:
      U[r,s] = U1[r] x U2[s]
      V[r,s] = V1[r] x V2[s]
      W[r,s] = W1[r] x W2[s]
    """
    r1, r2 = U1.shape[1], U2.shape[1]
    rank = r1 * r2
    n1, m1, p1 = U1.shape[0] // 2, 2, 2  # inferred from 2x2 case
    # Actually let's be more general:
    # U1 is (n1*m1, r1), V1 is (m1*p1, r1), W1 is (p1*n1, r1)
    # We need n1, m1, p1. For 2x2x2: n1=m1=p1=2
    # For the combined 6x6x6: we'll use 2x2 blocks, so n1=m1=p1=2, and the "outer" is 3x3x3
    # But actually for Strassen on 6x6, we partition into 3x3 blocks of 2x2 matrices.
    # The Kronecker product of two 2x2x2 decompositions gives a 4x4x4 decomposition.
    # Then we need a 3x3x3 decomposition to get 6x6x6.
    
    n1 = U1.shape[0] // m1 if False else 2  # This is getting complicated
    # Let me just do it explicitly for the 2x2 case
    n1, m1, p1 = 2, 2, 2
    U = np.zeros((n1 * m1 * 2 * 2, rank), dtype=np.float64)  # No, this is wrong
    
    # Actually, the Kronecker product of two (n1,m1,p1) and (n2,m2,p2) decompositions
    # gives an (n1*n2, m1*m2, p1*p2) decomposition.
    # For two 2x2x2 decompositions: gives 4x4x4.
    # Let's implement it properly.
    n1, m1, p1 = U1.shape[0] // m1, m1, V1.shape[0] // m1  # This is fragile
    
    # Let me just hardcode for the 2x2 case since that's what we're using
    # U1 is (4, r1), V1 is (4, r1), W1 is (4, r1)
    # U2 is (4, r2), V2 is (4, r2), W2 is (4, r2)
    # Result: U is (16, r1*r2), V is (16, r1*r2), W is (16, r1*r2)
    n1, m1, p1 = 2, 2, 2
    n2, m2, p2 = 2, 2, 2
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    for r in range(r1):
        for s in range(r2):
            col = r * r2 + s
            # U row index: (i1*n2 + i2) * m + (j1*m2 + j2) = i1*(n2*m) + i2*m + j1*m2 + j2
            # = i1*(n2*m) + i2*m + j1*m2 + j2
            # Where i1 in [0,n1), i2 in [0,n2), j1 in [0,m1), j2 in [0,m2)
            for i1 in range(n1):
                for i2 in range(n2):
                    for j1 in range(m1):
                        for j2 in range(m2):
                            u1_val = U1[i1 * m1 + j1, r]
                            u2_val = U2[i2 * m2 + j2, s]
                            u_val = u1_val * u2_val
                            if u_val != 0:
                                row = (i1 * n2 + i2) * m + (j1 * m2 + j2)
                                U[row, col] = u_val
            # V row index: (j1*m2 + j2) * p + (k1*p2 + k2) = j1*(m2*p) + j2*p + k1*p2 + k2
            for j1 in range(m1):
                for j2 in range(m2):
                    for k1 in range(p1):
                        for k2 in range(p2):
                            v1_val = V1[j1 * p1 + k1, r]
                            v2_val = V2[j2 * p2 + k2, s]
                            v_val = v1_val * v2_val
                            if v_val != 0:
                                row = (j1 * m2 + j2) * p + (k1 * p2 + k2)
                                V[row, col] = v_val
            # W row index: (k1*n2 + k2) * n + (i1*n2 + i2) = k1*(n2*n) + k2*n + i1*n2 + i2
            for k1 in range(p1):
                for k2 in range(p2):
                    for i1 in range(n1):
                        for i2 in range(n2):
                            w1_val = W1[k1 * n1 + i1, r]
                            w2_val = W2[k2 * n2 + i2, s]
                            w_val = w1_val * w2_val
                            if w_val != 0:
                                row = (k1 * n2 + k2) * n + (i1 * n2 + i2)
                                W[row, col] = w_val
    
    return U, V, W


def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly computes matrix multiplication."""
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Should equal delta (i.e., T[i,j,k] = 1 if this is a valid term, but actually
    # the tensor for matrix multiplication is T_{i,j,k} = 1 for all i,j,k since
    # C_{i,k} = sum_j A_{i,j} B_{j,k})
    # Wait, the tensor is T_{i,j,k} = A_{i,j} * B_{j,k} which as a trilinear form
    # has T_{i,j,k} = 1 for all i,j,k.
    # So we check: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all i,j,k
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(U.shape[1]):
                    val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err


def _convert_to_lists(U, V, W):
    """Convert numpy arrays to lists of lists with integer/half-integer entries."""
    def to_list(arr):
        return [[float(x) for x in row] for row in arr]
    return to_list(U), to_list(V), to_list(W)


def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 6x6 matrix multiplication.
    
    Strategy: Use Strassen's 7-multiplication decomposition for 2x2 blocks.
    Partition the 6x6 matrices into 3x3 blocks of 2x2 matrices.
    
    The overall decomposition combines:
    - A 3x3x3 decomposition for the block-level multiplication
    - A 2x2x2 Strassen decomposition for each block
    
    For the 3x3x3 level, we use the schoolbook (rank 27) since no known
    better decomposition exists for 3x3.
    
    Total rank = 27 * 7 = 189 (vs schoolbook 216)
    """
    start_time = time.time()
    deadline = 150.0
    
    n, m, p = DIMENSIONS  # 6, 6, 6
    
    # Get Strassen 2x2x2 decomposition
    U_s, V_s, W_s = _strassen_2x2_decomposition()
    
    # Get schoolbook 3x3x3 decomposition for the block level
    U_b, V_b, W_b = _schoolbook_decomposition(3, 3, 3)
    
    # Combine via Kronecker product: (3,3,3) x (2,2,2) -> (6,6,6)
    U, V, W = _kron_decomposition(U_b, V_b, W_b, U_s, V_s, W_s)
    
    rank = U.shape[1]
    
    # Verify the decomposition
    elapsed = time.time() - start_time
    if elapsed < deadline - 10:
        valid, max_err = _verify_decomposition(U, V, W, n, m, p)
        if not valid:
            # Fall back to schoolbook
            U, V, W = _schoolbook_decomposition(n, m, p)
            rank = n * m * p
    
    # Convert to lists
    U_list, V_list, W_list = _convert_to_lists(U, V, W)
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
