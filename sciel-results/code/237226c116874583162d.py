"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-style decomposition for 3x4 * 4x4.
    # Pad A (3x4) to 4x4 by adding a zero row.
    # Apply Strassen to the 4x4 * 4x4 product (rank 7 scalar mults on 2x2 blocks).
    # This gives 7 * 2^3 = 56 scalar multiplications total, but we can do better
    # by only keeping the parts needed for the 3x4 output.
    
    # Actually, let's use a known rank-minimal decomposition for 3x4 * 4x4.
    # The tensor rank for multiplying an m x n matrix by an n x p matrix is at most
    # ceil(m/2)*ceil(n/2)*ceil(p/2)*7 if using Strassen recursively, but for small
    # dimensions we can often do better.
    
    # For 3x4 * 4x4: pad to 4x4 * 4x4.
    # Strassen on 4x4: split each 4x4 into 2x2 blocks.
    # A = [[A11, A12], [A21, A22]] where each is 2x2.
    # B = [[B11, B12], [B21, B22]] where each is 2x2.
    # Strassen gives 7 products of 2x2 matrices, each of which needs 4 scalar mults.
    # Total: 28 scalar mults.
    # But we only need C to be 3x4, so we only need rows 0,1,2 of C.
    # Row 3 of C depends on A21, A22 which are zeros (since A is 3x4 padded to 4x4).
    # So C31 and C32 are zero, and we don't need them.
    # This means we might be able to eliminate some of the Strassen terms.
    
    # Let's just implement the full Strassen for 4x4 * 4x4 with A padded.
    # A is 4x4: A[0:2, :] and A[2:3, :] are the actual data, A[3, :] = 0.
    # B is 4x4.
    
    # 2x2 blocks of A (padded):
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4]
    # A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    # Note A21[1, :] = A[3, 0:2] = 0, A22[1, :] = A[3, 2:4] = 0.
    
    # 2x2 blocks of B:
    # B11 = B[0:2, 0:2], B12 = B[0:2, 2:4]
    # B21 = B[2:4, 0:2], B22 = B[2:4, 2:4]
    
    # Strassen formulas:
    # P1 = (A11 + A22)(B11 + B22)
    # P2 = (A21 + A22) B11
    # P3 = A11 (B12 - B22)
    # P4 = A22 (B21 - B11)
    # P5 = (A11 + A12) B22
    # P6 = (A21 - A11)(B11 + B12)
    # P7 = (A12 - A22)(B21 + B22)
    
    # C11 = P1 + P4 - P5 - P7
    # C12 = P3 + P5
    # C21 = P2 + P7
    # C22 = P1 - P2 + P3 + P6
    
    # Each Pi is a 2x2 matrix product, requiring 4 scalar multiplications.
    # Total: 28 scalar multiplications.
    
    # We need to express this as a tensor decomposition.
    # U has shape (n*m, rank) = (12, rank), V has shape (m*p, rank) = (16, rank), W has shape (p*n, rank) = (12, rank).
    # Wait, the interface says U: (n*m, rank), V: (m*p, rank), W: (p*n, rank).
    # n=3, m=4, p=4.
    # U: (12, rank), V: (16, rank), W: (12, rank).
    
    # The tensor T[i, j, k] = 1 if i in [0,3), j in [0,4), k in [0,4) and the term is A[i,j] B[j,k].
    # Actually T[i, j, k] for i in [0,3), j in [0,4), k in [0,4).
    
    # Let's build the decomposition by enumerating the 28 scalar multiplications.
    # Each scalar multiplication is of the form (sum of A terms) * (sum of B terms).
    # We need to track which A[i,j] and B[j,k] are involved.
    
    # Let me define the 2x2 sub-matrices in terms of the global indices.
    # A11: A[0,0], A[0,1], A[1,0], A[1,1]
    # A12: A[0,2], A[0,3], A[1,2], A[1,3]
    # A21: A[2,0], A[2,1], A[3,0]=0, A[3,1]=0
    # A22: A[2,2], A[2,3], A[3,2]=0, A[3,3]=0
    
    # B11: B[0,0], B[0,1], B[1,0], B[1,1]
    # B12: B[0,2], B[0,3], B[1,2], B[1,3]
    # B21: B[2,0], B[2,1], B[3,0], B[3,1]
    # B22: B[2,2], B[2,3], B[3,2], B[3,3]
    
    # For each Pi, I need to write it as a sum of scalar multiplications of the form:
    # (sum_{(i,j)} a_{ij} A_{ij}) * (sum_{(j',k)} b_{j'k} B_{j'k})
    # where the j in A must match the j' in B for the product to contribute to C.
    
    # Actually, the decomposition is:
    # C_{ik} = sum_r U_{ik,r} V_{j,r} W_{kj,r} ... no wait.
    # The standard form is:
    # sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ... no.
    
    # Let me re-read the interface.
    # "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "W[k*n+i, r]"
    
    # The tensor is T_{i,j,k} = delta (for the appropriate i,j,k).
    # The decomposition is: T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
    
    # So for each r, we have a triple (i_r, j_r, k_r) such that:
    # U_{i*m+j, r} is non-zero for some (i,j)
    # V_{j*p+k, r} is non-zero for some (j,k)
    # W_{k*n+i, r} is non-zero for some (k,i)
    
    # And the product U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} must equal T_{i,j,k} when summed over r.
    
    # This is a rank-1 tensor decomposition.
    
    # For the schoolbook: T_{i,j,k} = 1 if i in [0,3), j in [0,4), k in [0,4).
    # Each (i,j,k) gives one rank-1 term with U_{i*m+j, r} = 1, V_{j*p+k, r} = 1, W_{k*n+i, r} = 1.
    # Rank = 3*4*4 = 48.
    
    # For Strassen, we have 28 scalar multiplications. But each scalar multiplication
    # involves multiple A and B entries, so it's not a single rank-1 term.
    # We need to expand each Pi into its constituent scalar multiplications.
    
    # Let's just use the schoolbook for now to ensure correctness, and then try to optimize.
    # Actually, let me try a different approach: use the known optimal rank for 3x4*4x4.
    # The tensor rank of matmul(3,4,4) is at most 28 (from Strassen on 4x4*4x4 with padding).
    # But can we do better?
    
    # Let me just implement the 28-term Strassen decomposition.
    
    n, m, p = 3, 4, 4
    # We'll use rank 28.
    rank = 28
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    # Helper to add a term: U[i*m+j, r] = ua, V[j*p+k, r] = vb, W[k*n+i, r] = wc
    # This contributes ua*vb*wc to T[i,j,k]
    # But wait, the decomposition sums U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] over r.
    # For a fixed r, U[i*m+j, r] is non-zero for at most one (i,j)? No, it can be non-zero for multiple.
    # Actually, U is a matrix of shape (n*m, rank). For a fixed r, U[:, r] is a vector in R^{n*m}.
    # The tensor entry T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is not quite right. Let me re-examine.
    # The tensor T has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # T[i,j,k] = 1 for matmul.
    # The decomposition is T[i,j,k] = sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # But the interface flattens these: U has shape (n*m, rank) where the first index is i*m+j.
    # V has shape (m*p, rank) where the first index is j*p+k.
    # W has shape (p*n, rank) where the first index is k*n+i.
    
    # So T[i,j,k] = sum_r U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r]
    
    # For the schoolbook, each (i,j,k) corresponds to one r with U[i*m+j, r]=1, V[j*p+k, r]=1, W[k*n+i, r]=1.
    # And all other entries in those columns are 0.
    
    # For Strassen, we have 28 "scalar multiplications". But each scalar multiplication
    # is of the form (linear combo of A's) * (linear combo of B's), and the result is a
    # linear combo of C's.
    
    # I think the cleanest way is to just use the schoolbook decomposition. It's guaranteed to work.
    # Rank = 48.
    
    # Let me just do the schoolbook.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
