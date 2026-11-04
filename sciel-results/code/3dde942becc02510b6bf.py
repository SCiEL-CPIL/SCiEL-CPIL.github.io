"""Rank-minimal search for 4x5x6 matrix multiplication tensor with block-structured approach."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 5, 6]
N, M, P = 4, 5, 6

_T0 = time.time()
_DEADLINE = _T0 + 140  # 10s safety margin before 150s kill

def _schoolbook():
    """Return the schoolbook decomposition of rank n*m*p."""
    r = N * M * P
    U = np.zeros((N * M, r), dtype=np.int64)
    V = np.zeros((M * P, r), dtype=np.int64)
    W = np.zeros((P * N, r), dtype=np.int64)
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[i * M + j, col] = 1
                V[j * P + k, col] = 1
                W[k * N + i, col] = 1
                col += 1
    return U, V, W

def _verify(U, V, W):
    """Check that the decomposition exactly computes matrix multiplication."""
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    r = U.shape[1]
    T = np.zeros((N, M, P), dtype=np.float64)
    for r_idx in range(r):
        u_r = U[:, r_idx].reshape(N, M)
        v_r = V[:, r_idx].reshape(M, P)
        w_r = W[:, r_idx].reshape(P, N)
        for i in range(N):
            for j in range(M):
                uval = u_r[i, j]
                if uval == 0:
                    continue
                for k in range(P):
                    T[i, j, k] += uval * v_r[j, k] * w_r[k, i]
    return np.allclose(T, 1.0, atol=1e-9)

def _strassen_2x2_block(offset_i, offset_j, offset_k, U, V, W, base_col):
    """Apply a Strassen 2x2x2 block at the given offsets. Returns number of columns used."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n_strassen = 7
    for s in range(n_strassen):
        col = base_col + s
        # Determine which U, V, W entries are non-zero based on Strassen formula
        # This is complex; for now, just mark as schoolbook for 2x2x2 (4 products)
        # Actually, let's just use schoolbook for the 2x2x2 block (4 cols)
        pass
    # For simplicity, use schoolbook for 2x2x2: 4 columns
    return 4

def _block_decomposition():
    """
    Construct a block-structured decomposition:
    - Split 4x5x6 into blocks and use Strassen-like patterns where beneficial
    - For 4x4x4 portion, use 2x2x2 Strassen blocks (7 instead of 8 per 2x2x2)
    - For remaining parts, use schoolbook
    """
    # 4x5x6: We can think of this as:
    # - 4x4x4 block (top-left): use 2 Strassen 2x2x2 blocks = 14 cols (vs 16 schoolbook)
    # - 4x1x4 block: schoolbook = 16 cols
    # - 4x4x2 block: schoolbook = 32 cols  
    # - 4x1x2 block: schoolbook = 8 cols
    # Total: 14 + 16 + 32 + 8 = 70 vs schoolbook 120
    
    # But wait - the tensor structure means we can't simply block like this
    # because the middle dimension (j) is shared.
    
    # Let me think differently. The tensor T[i,j,k]=1 for all i in [0,4), j in [0,5), k in [0,6)
    # A decomposition is a set of rank-1 tensors whose sum equals T.
    
    # For the 4x4x4 sub-tensor (i<4, j<4, k<4), we can use Strassen.
    # For the rest, we need additional terms.
    
    # Actually, let's try a different structural approach:
    # Use the fact that T[i,j,k] = 1 can be written as sum of rank-1 terms.
    # A rank-1 term is u_i * v_j * w_k for some vectors u, v, w.
    
    # One approach: use Fourier-like or polynomial-based constructions.
    # For small dimensions, we can try:
    # T[i,j,k] = sum_{r} f_r(i) * g_r(j) * h_r(k)
    
    # Let's try a construction based on the fact that:
    # 1 = sum_{a=0}^{d-1} omega^{ai} * omega^{-aj} * omega^{ak} for appropriate omega
    # But this requires complex numbers and exactness.
    
    # Alternative: use the identity 1 = x^0 for any x != 0, but that's trivial.
    
    # Let's try a more systematic approach:
    # For the 4x4x4 part, use 2x2x2 Strassen blocks.
    # Split i into (i0, i1), j into (j0, j1), k into (k0, k1) each in {0,1}
    # For each 2x2x2 block, use Strassen (7 cols instead of 8)
    # 4x4x4 = 8 blocks of 2x2x2, each using 7 cols = 56 cols (vs 64 schoolbook)
    # Then 4x1x4, 4x4x2, 4x1x2 use schoolbook: 16+32+8=56
    # Total: 56+56=112. Still better than 120 but not dramatic.
    
    # Actually, I realize the issue: the tensor is T[i,j,k]=1 for ALL i,j,k, not just the 4x4x4 part.
    # The full tensor is 4x5x6 = 120 entries, all equal to 1.
    # So we need a decomposition of the all-ones tensor of shape 4x5x6.
    
    # The all-ones tensor of shape n x m x p has rank... well, it's rank 1!
    # Because T[i,j,k] = 1 = 1*1*1 for all i,j,k.
    # So U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1 for all i,j,k.
    # That gives rank 1!
    
    # Wait, that can't be right for matrix multiplication. Let me re-read the problem.
    
    # The problem says: "Find a rank-minimal exact tensor decomposition for multiplying a 4 by 5 matrix by a 5 by 6 matrix."
    # And: "The output factor follows the official trace convention W[k*n+i,r]."
    # And the verification: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For matrix multiplication C = AB, C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T[i,j,k] should be 1 if A[i,j]*B[j,k] contributes to C[i,k], else 0.
    # Since every A[i,j]*B[j,k] contributes to C[i,k], T[i,j,k] = 1 for all i,j,k.
    
    # So T is indeed the all-ones tensor! And it has rank 1.
    # But that seems too simple. Let me check the schoolbook:
    # Schoolbook: for each (i,j,k), U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    # Sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1 (only one term contributes)
    # So yes, T[i,j,k] = 1 for all i,j,k.
    
    # So the all-ones tensor of shape 4x5x6 can be decomposed with rank 1:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    
    # Let me verify: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = U[i*m+j,0]*V[j*p+k,0]*W[k*n+i,0] = 1*1*1 = 1 ✓
    
    # So the answer is rank 1!
    
    U = np.ones((N * M, 1), dtype=np.int64)
    V = np.ones((M * P, 1), dtype=np.int64)
    W = np.ones((P * N, 1), dtype=np.int64)
    
    return U, V, W

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x5x6 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' as lists of lists with integer/half-integer entries."""
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since every product A[i,j]*B[j,k] contributes to C[i,k])
    # This is the all-ones tensor, which has rank 1.
    # Decomposition: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    
    U = np.ones((N * M, 1), dtype=np.int64)
    V = np.ones((M * P, 1), dtype=np.int64)
    W = np.ones((P * N, 1), dtype=np.int64)
    
    # Verify
    if not _verify(U, V, W):
        # Fallback to schoolbook if something is wrong
        U, V, W = _schoolbook()
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
