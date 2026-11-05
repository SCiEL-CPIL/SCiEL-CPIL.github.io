import numpy as np
from datetime import datetime, timedelta

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = datetime.now()
    deadline = start_time + timedelta(seconds=120)
    
    def time_remaining():
        return (deadline - datetime.now()).total_seconds()
    
    # Strassen's rank-7 decomposition for 3x3x3
    # The 7 intermediate products for multiplying A(3x3) * B(3x3):
    # M1 = A00 * (B01 - B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A10 - A11) * (B11 + B21)
    # M4 = A11 * (B00 - B10)
    # M5 = (A00 + A01) * B00
    # M6 = (A01 - A11) * (B10 + B20)
    # M7 = (A00 + A10) * (B01 + B11)
    #
    # Result C[i,j] = sum of appropriate M's
    # C00 = M5 + M2 - M7 - M3
    # C01 = M1 + M7
    # C02 = M6 + M3
    # C10 = M4 + M3
    # C11 = M5 - M1 + M2 - M4
    # C12 = M1 - M2 + M4 + M6
    # C20 = M5 + M2 - M7 - M3
    # C21 = M1 + M7
    # C22 = M6 + M3
    
    # For 3x7 multiplication: A is 3x3, B is 3x7
    # Split B into B_left (3x3, cols 0-2) and B_right (3x4, cols 3-6)
    # C = A * B_left + A * B_right
    # 
    # For A * B_left: use Strassen rank-7
    # For A * B_right: use schoolbook rank-12 (3*4)
    # Total naive combined: rank 7 + 12 = 19
    # But we need to check for linear dependencies
    
    # Let me build the full tensor decomposition and then reduce rank
    
    # First, build the combined rank-19 decomposition
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (21, rank)
    # W: (p*n, rank) = (21, rank)
    
    # Strassen part for B_left (columns 0,1,2 of B)
    # M1 = A[0,0] * (B[0,1] - B[1,1])
    # M2 = (A[0,0] + A[1,0]) * B[1,1]
    # M3 = (A[1,0] - A[1,1]) * (B[1,1] + B[2,1])
    # M4 = A[1,1] * (B[0,0] - B[1,0])
    # M5 = (A[0,0] + A[0,1]) * B[0,0]
    # M6 = (A[0,1] - A[1,1]) * (B[1,0] + B[2,0])
    # M7 = (A[0,0] + A[1,0]) * (B[0,1] + B[1,1])
    
    # Result for left part:
    # C[0,0] = M5 + M2 - M7 - M3
    # C[0,1] = M1 + M7
    # C[0,2] = M6 + M3
    # C[1,0] = M4 + M3
    # C[1,1] = M5 - M1 + M2 - M4
    # C[1,2] = M1 - M2 + M4 + M6
    # C[2,0] = M5 + M2 - M7 - M3  (wait, this is wrong for 3x3)
    
    # Let me be more careful. For 3x3 * 3x3 = 3x3:
    # A = [[a00, a01, a02],
    #      [a10, a11, a12],
    #      [a20, a21, a22]]
    # B = [[b00, b01, b02],
    #      [b10, b11, b12],
    #      [b20, b21, b22]]
    # C[i,j] = sum_k A[i,k] * B[k,j]
    
    # Strassen for 3x3:
    # M1 = a00 * (b01 - b11)
    # M2 = (a00 + a10) * b11
    # M3 = (a10 - a11) * (b11 + b21)
    # M4 = a11 * (b00 - b10)
    # M5 = (a00 + a01) * b00
    # M6 = (a01 - a11) * (b10 + b20)
    # M7 = (a00 + a10) * (b01 + b11)
    
    # C[0,0] = M5 + M2 - M7 - M3 + a02*b00 + a12*b10 + a22*b20
    # Wait, this doesn't work cleanly for 3x3. Let me use a different approach.
    
    # Actually, for 3x3 matrix multiplication, the known minimal rank is 19 (Larsen 2010)
    # or 23 (Strassen-like). Let me use a simpler approach.
    
    # Let me just use schoolbook for everything first, then try to reduce.
    # Schoolbook rank = 3*3*7 = 63
    
    # Actually, let me try a smarter decomposition.
    # Split B into 3x3 and 3x4 blocks.
    # For the 3x3 block, use a rank-19 or rank-23 decomposition.
    # For the 3x4 block, use schoolbook rank-12.
    # Then look for linear dependencies.
    
    # Let me start with a known rank-23 decomposition for 3x3x3
    # and rank-12 for 3x3x4, giving total rank 35.
    # Then try to eliminate dependencies.
    
    # Actually, let me just build the full schoolbook decomposition and
    # then systematically try to find linear dependencies and remove them.
    
    # Build schoolbook rank-63 decomposition
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Now try to find linear dependencies among the rank-63 terms
    # The tensor is T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We want to find if some combination of terms is linearly dependent
    
    # A term r is linearly dependent if there exist coefficients c_s (s != r)
    # such that for all (i,j,k): U[i,j,r]*V[j,k,r]*W[k,i,r] = sum_s c_s * U[i,j,s]*V[j,k,s]*W[k,i,s]
    
    # This is a nonlinear problem. Let me try a different approach.
    
    # Use the fact that for matrix multiplication, certain symmetries exist.
    # For example, if we have terms that differ only in a permutation of indices,
    # they might be related.
    
    # Let me try to construct a lower-rank decomposition using known algorithms.
    
    # For 3x3 * 3x3, use Strassen's algorithm (rank 7 for the core, but needs more for 3x3)
    # Actually, Strassen's original algorithm is for 2x2. For 3x3, we can use
    # a recursive approach or a direct rank-19 decomposition.
    
    # Let me try to use a block decomposition.
    # Write A as 3x3 and B as 3x7.
    # Partition B into B1 (3x3) and B2 (3x4).
    # C = A*B1 + A*B2
    
    # For A*B1 (3x3 * 3x3 = 3x3), use a rank-19 decomposition.
    # For A*B2 (3x3 * 3x4 = 3x4), use schoolbook rank-12.
    # Total: rank 31.
    
    # Then look for dependencies between the two blocks.
    
    # Let me implement a rank-19 decomposition for 3x3x3.
    # This is based on Larsen's algorithm or similar.
    
    # Actually, let me just try to reduce the rank by finding linear dependencies
    # in the schoolbook decomposition using numerical linear algebra.
    
    # The condition for term r to be redundant is that the vector
    # v_r = vec(U[:,r] * V[:,r] * W[:,r]) is in the span of {v_s : s != r}
    # where vec flattens the 3D tensor.
    
    # Let me compute the tensor for each term and check rank.
    
    # Build the 3D tensor for each term
    tensors = np.zeros((rank, n, m, p))
    for r in range(rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensors[r, i, j, k] = U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Flatten each tensor
    flat_tensors = tensors.reshape(rank, -1)
    
    # Compute the rank of the matrix of flattened tensors
    # If rank < 63, then some terms are linearly dependent
    
    # Use SVD to find the effective rank
    try:
        U_svd, s, Vt_svd = np.linalg.svd(flat_tensors, full_matrices=False)
        # Effective rank: number of singular values > tolerance
        tol = 1e-10
        eff_rank = np.sum(s > tol)
    except:
        eff_rank = rank
    
    # If eff_rank < rank, we can reduce the rank
    # Let me find which terms are linearly dependent
    
    # For now, let me just return the schoolbook decomposition if no reduction found
    # and try to implement a proper rank-reduction algorithm
    
    # Let me try a different approach: use known fast matrix multiplication algorithms
    
    # For 3x3 * 3x7, we can think of it as 3x3 * (3x3 + 3x4)
    # Use Strassen-like for 3x3*3x3 and schoolbook for 3x3*3x4
    
    # Actually, let me just try to find a rank-21 decomposition by construction.
    
    # The idea: use 7 Strassen-like products for the first 3 columns of B,
    # and 12 schoolbook products for the last 4 columns.
    # Total: 19 products. But some may be linearly dependent.
    
    # Let me construct this explicitly.
    
    # Strassen-like for A(3x3) * B1(3x3):
    # M1 = A[0,0] * (B1[0,1] - B1[1,1])
    # M2 = (A[0,0] + A[1,0]) * B1[1,1]
    # M3 = (A[1,0] - A[1,1]) * (B1[1,1] + B1[2,1])
    # M4 = A[1,1] * (B1[0,0] - B1[1,0])
    # M5 = (A[0,0] + A[0,1]) * B1[0,0]
    # M6 = (A[0,1] - A[1,1]) * (B1[1,0] + B1[2,0])
    # M7 = (A[0,0] + A[1,0]) * (B1[0,1] + B1[1,1])
    
    # But this doesn't give the full 3x3 result. I need to add more terms.
    
    # Let me just use a simpler approach and try to optimize numerically.
    
    # For now, return the schoolbook decomposition
    def verify(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-10:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Naive decomposition failed verification")
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
