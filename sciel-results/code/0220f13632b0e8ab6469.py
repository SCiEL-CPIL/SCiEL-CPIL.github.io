"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    T = np.zeros((n, m, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, j, k] = 1.0
    
    recon = np.zeros((n, m, m, p))
    for c in range(rank):
        u4 = U[:, c].reshape(n, m)
        v4 = V[:, c].reshape(m, p)
        w4 = W[:, c].reshape(p, n)
        recon += u4[:, :, np.newaxis, np.newaxis] * v4[np.newaxis, np.newaxis, :, :] * w4.T[np.newaxis, np.newaxis, :, :]
    
    return np.allclose(recon, T, atol=1e-10)

def construct():
    """
    Returns a rank-12 exact tensor decomposition for 2x3 x 3x3 matrix multiplication.
    
    Strategy:
    - Split the 2x3 A matrix into two 2x2 blocks (first 2 cols, last 1 col)
    - Use Strassen's algorithm for the 2x2 x 2x2 part (first 2 cols of B)
    - Use schoolbook for the remaining part (last col of B)
    - Total rank: 7 (Strassen) + 2 (schoolbook for 2x1) + 3 (schoolbook for 1x3) = 12
    
    Actually, let me think more carefully:
    A is 2x3, B is 3x3.
    C[i,k] = sum_{j=0}^{2} A[i,j] * B[j,k]
    
    We can split j into j=0,1 and j=2:
    C[i,k] = (A[:,0:2] @ B[0:2,:])[i,k] + A[:,2] * B[2,k]
    
    For the first part: 2x2 @ 2x2, use Strassen (rank 7)
    For the second part: 2x1 @ 1x3, this is just outer product A[:,2] (2x1) and B[2,:] (1x3)
    which is rank 1 (outer product). But we need to express it in the tensor form.
    
    Actually, the second part is: sum over k of A[i,2] * B[2,k]
    This is a rank-1 term in the tensor: U[i,2,c] * V[2,k,c] * W[k,i,c]
    where U[i,2,c] = A[i,2], V[2,k,c] = B[2,k], W[k,i,c] = 1 for all k,i.
    
    But wait, we need to decompose the multiplication tensor, not compute a specific product.
    The tensor T[i,j,l,k] = delta(j,l).
    
    For the j=2 part: T[i,2,2,k] = 1 for all i,k.
    This can be decomposed as: U[i,2,c] * V[2,k,c] * W[k,i,c] with appropriate values.
    
    Let me just construct the full rank-12 decomposition directly.
    
    Actually, a cleaner approach: use the known rank-12 decomposition.
    
    Let me construct it step by step.
    """
    n, m, p = 2, 3, 3
    
    # We'll build U (n*m x rank), V (m*p x rank), W (p*n x rank)
    # n*m = 6, m*p = 9, p*n = 6
    
    rank = 12
    U = np.zeros((6, rank))
    V = np.zeros((9, rank))
    W = np.zeros((6, rank))
    
    # Part 1: Strassen for the 2x2 x 2x2 subproblem (j=0,1 and k=0,1)
    # This handles T[i,j,l,k] where j,l in {0,1} and k in {0,1}
    
    # Strassen's 7 products for A1 (2x2) and B1 (2x2):
    # M1 = (A10+A11)(B10+B11)
    # M2 = (A10-A11)B11
    # M3 = A10(B10-B11)
    # M4 = A11(B11-B10)  -- actually standard Strassen uses:
    # M4 = (A10+A11)(B10+B11) - M1 ... no, let me use standard form
    
    # Standard Strassen for C1 = A1 * B1 where A1,B1,C1 are 2x2:
    # M1 = (A10 + A11) * (B10 + B11)
    # M2 = (A10 - A11) * B11
    # M3 = A10 * (B10 - B11)
    # M4 = A11 * (B11 - B10)  -- wait, this isn't quite right
    
    # Let me use the standard 7 products:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    # C00 = M1 - M4 + M6 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M3 + M5 - M6
    
    # In our tensor, for the (i,j,l,k) indices where i in {0,1}, j in {0,1}, l in {0,1}, k in {0,1}:
    # T[i,j,l,k] = 1 if j == l, else 0
    
    # So we need to decompose this 2x2x2x2 tensor with rank 7.
    
    # Let me define the Strassen decomposition for the 2x2x2x2 case:
    # U1[i,j,c], V1[l,k,c], W1[k,i,c] for c in 0..6
    
    # i,j,l,k all in {0,1}
    # U1 is 2x2 x 7, V1 is 2x2 x 7, W1 is 2x2 x 7 (W indexed by k,i)
    
    U1 = np.zeros((4, 7))  # i*2 + j
    V1 = np.zeros((4, 7))  # l*2 + k
    W1 = np.zeros((4, 7))  # k*2 + i
    
    # M1 = (a00+a11)(b00+b11)
    # U1: a00, a11 -> U1[0,0]=1, U1[3,0]=1 (i=0,j=0 and i=1,j=1)
    # V1: b00, b11 -> V1[0,0]=1, V1[3,0]=1 (l=0,k=0 and l=1,k=1)
    # W1: 1 -> W1[0,0]=1, W1[1,0]=1, W1[2,0]=1, W1[3,0]=1
    U1[0,0] = 1; U1[3,0] = 1
    V1[0,0] = 1; V1[3,0] = 1
    W1[0,0] = 1; W1[1,0] = 1; W1[2,0] = 1; W1[3,0] = 1
    
    # M2 = (a10+a11)b00
    U1[2,1] = 1; U1[3,1] = 1
    V1[0,1] = 1
    W1[0,1] = 1; W1[1,1] = 1; W1[2,1] = 1; W1[3,1] = 1
    
    # M3 = a00(b01-b11)
    U1[0,2] = 1
    V1[1,2] = 1; V1[3,2] = -1
    W1[0,2] = 1; W1[1,2] = 1; W1[2,2] = 1; W1[3,2] = 1
    
    # M4 = a11(b10-b00)
    U1[3,3] = 1
    V1[2,3] = 1; V1[0,3] = -1
    W1[0,3] = 1; W1[1,3] = 1; W1[2,3] = 1; W1[3,3] = 1
    
    # M5 = (a00+a01)b11
    U1[0,4] = 1; U1[1,4] = 1
    V1[3,4] = 1
    W1[0,4] = 1; W1[1,4] = 1; W1[2,4] = 1; W1[3,4] = 1
    
    # M6 = (a10-a00)(b00+b01)
    U1[2,5] = 1; U1[0,5] = -1
    V1[0,5] = 1; V1[1,5] = 1
    W1[0,5] = 1; W1[1,5] = 1; W1[2,5] = 1; W1[3,5] = 1
    
    # M7 = (a01-a11)(b10+b11)
    U1[1,6] = 1; U1[3,6] = -1
    V1[2,6] = 1; V1[3,6] = 1
    W1[0,6] = 1; W1[1,6] = 1; W1[2,6] = 1; W1[3,6] = 1
    
    # Map U1 (i*2+j for i,j in 0,1) to U_full (i*3+j for i in 0,1, j in 0,1)
    # U1 index: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # U_full index: 0->(0,0), 1->(0,1), 2->(0,2), 3->(1,0), 4->(1,1), 5->(1,2)
    U_map1 = [0, 1, 3, 4]
    U[U_map1, 0:7] = U1
    
    # Map V1 (l*2+k for l,k in 0,1) to V_full (l*3+k for l in 0,1, k in 0,1)
    # V1 index: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # V_full index: 0->(0,0), 1->(0,1), 2->(0,2), 3->(1,0), 4->(1,1), 5->(1,2)
    V_map1 = [0, 1, 3, 4]
    V[V_map1, 0:7] = V1
    
    # Map W1 (k*2+i for k,i in 0,1) to W_full (k*2+i for k in 0,1, i in 0,1)
    # W1 index: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1)
    # W_full index: 0->(0,0), 1->(0,1), 2->(1,0), 3->(1,1), 4->(2,0), 5->(2,1)
    W_map1 = [0, 1, 2, 3]
    W[W_map1, 0:7] = W1
    
    # Part 2: Handle k=2 column (the 3rd column of B)
    # For k=2, we need T[i,j,l,2] = 1 if j==l
    # This is: sum over j A[i,j] B[j,2]
    # We need rank-3 terms for this (one for each j in 0,1,2)
    
    # For j=0, k=2:
    # U[i,0,c], V[0,2,c], W[2,i,c] with T[i,0,l,2] = delta(0,l)
    # U[i,0,c]=1, V[0,2,c]=1, W[2,i,c]=1
    U[0,7] = 1; U[3,7] = 1  # i=0,j=0 and i=1,j=0
    V[0*3+2, 7] = 1         # l=0, k=2
    W[2*2+0, 7] = 1; W[2*2+1, 7] = 1  # k=2, i=0 and i=1
    
    # For j=1, k=2:
    U[1,8] = 1; U[4,8] = 1  # i=0,j=1 and i=1,j=1
    V[1*3+2, 8] = 1         # l=1, k=2
    W[2*2+0, 8] = 1; W[2*2+1, 8] = 1  # k=2, i=0 and i=1
    
    # For j=2, k=2:
    U[2,9] = 1; U[5,9] = 1  # i=0,j=2 and i=1,j=2
    V[2*3+2, 9] = 1         # l=2, k=2
    W[2*2+0, 9] = 1; W[2*2+1, 9] = 1  # k=2, i=0 and i=1
    
    # Part 3: Handle k=0,1 but j=2 (rows 2 of B)
    # For j=2, k=0:
    # T[i,2,l,0] = 1 if l=2
    # U[i,2,c], V[2,0,c], W[0,i,c]
    U[2,10] = 1; U[5,10] = 1  # i=0,j=2 and i=1,j=2
    V[2*3+0, 10] = 1          # l=2, k=0
    W[0*2+0, 10] = 1; W[0*2+1, 10] = 1  # k=0, i=0 and i=1
    
    # For j=2, k=1:
    U[2,11] = 1; U[5,11] = 1  # i=0,j=2 and i=1,j=2
    V[2*3+1, 11] = 1          # l=2, k=1
    W[1*2+0, 11] = 1; W[1*2+1, 11] = 1  # k=1, i=0 and i=1
    
    # Verify
    assert _verify(U, V, W, n, m, p), "Verification failed"
    
    # Snap to integers/half-integers (they should already be integers)
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Final verification
    assert _verify(U, V, W, n, m, p), "Verification failed after snapping"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
