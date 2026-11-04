import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Build the schoolbook decomposition (rank 112) as baseline
    rank_sb = n * m * p  # 112
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Now try to find a lower rank decomposition using random search
    # We'll use the structure: for each rank term, we have (u, v, w) vectors
    # where u is in R^(n*m), v is in R^(m*p), w is in R^(p*n)
    # The constraint is: for all i,j,k: sum_t u[t][i*m+j] * v[t][j*p+k] = w[t][k*n+i]
    
    # This is a hard constraint. Let's think differently.
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if the corresponding 
    # entry in A*B is at position (i,k) when A is (n x m) and B is (m x p).
    # Actually, T_{i,k,j} = 1 means C[i,k] = sum_j A[i,j] * B[j,k]
    # In our notation: U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    # where U index = i*m + j, V index = j*p + k, W index = k*n + i
    
    # The constraint is: T = sum_t outer(U[:,t], V[:,t]) ... no that's not right.
    # Actually: T_{i,j,k} = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]? No.
    
    # Let me re-derive. The standard tensor decomposition of matmul:
    # T_{i,j,k} = sum_t U_{(i,j),t} * V_{(j,k),t} * W_{(i,k),t}
    # where i in range(n), j in range(m), k in range(p)
    # U has shape (n*m, r), indexed by (i,j)
    # V has shape (m*p, r), indexed by (j,k)  
    # W has shape (p*n, r), indexed by (k,i)
    
    # Wait, the problem says W index is k*n+i. So W is indexed by (k,i) not (i,k).
    # Let me re-read: "W[k*n+i,r]" - yes, W index = k*n + i.
    
    # So the constraint is:
    # For all i in 0..n-1, j in 0..m-1, k in 0..p-1:
    # T_{i,j,k} = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    # But T_{i,j,k} = 1 for all i,j,k (since matmul tensor is all ones in the appropriate sense)
    # Actually no. The matmul tensor T is defined by T_{i,j,k} = 1 if and only if...
    # Hmm, actually the matmul tensor is: T[i,j,k] = 1 for all i,j,k? No.
    
    # Let me think again. C = A * B, so C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor T has entries T[i,j,k] such that C[i,k] = sum_{i,j,k} T[i,j,k] * A[i,j] * B[j,k]
    # So T[i,j,k] = 1 if and only if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # which is always true. So T[i,j,k] = 1 for all i,j,k? That can't be right for the tensor rank.
    
    # Actually, I think the convention is:
    # T[i,j,k] = delta_{appropriate} 
    # Let me just verify with the schoolbook: in schoolbook, each term t corresponds to a specific (i,j,k)
    # and U[i*m+j, t] = 1, V[j*p+k, t] = 1, W[k*n+i, t] = 1, and 0 elsewhere.
    # Then sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1 if there exists t where all three are 1.
    # In schoolbook, for a given (i,j,k), there's exactly one t where all three are 1.
    # So the tensor entry is 1 for all (i,j,k) that appear, which is all of them.
    
    # Wait, that means T is just the all-ones tensor? That doesn't sound right.
    # Let me reconsider. Maybe the tensor is defined differently.
    
    # Actually, I think I'm overcomplicating this. The standard result is that the 
    # matrix multiplication tensor for (n x m) * (m x p) has rank n*m*p in the 
    # trivial (schoolbook) decomposition. The entries of the tensor are:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is because C[i,k] = sum_j A[i,j] B[j,k], and each term A[i,j]B[j,k] 
    # contributes with coefficient 1.
    
    # So yes, T is the all-ones tensor of shape (n*m) x (m*p) x (p*n) with the 
    # specific index correspondence. And the schoolbook decomposition uses n*m*p terms.
    
    # For a lower rank decomposition, we need to find r < n*m*p such that there exist
    # U (n*m x r), V (m*p x r), W (p*n x r) with:
    # For all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
    
    # This is a system of n*m*p = 112 equations in r*(n*m + m*p + p*n) unknowns.
    
    # Let's try a random approach: generate random U, V with small entries, solve for W,
    # and check if it works.
    
    # Actually, let's try a structured approach. Use Strassen-style decompositions.
    
    # For 4x4x4, Strassen gives rank 49 (using 7 products of 2x2x2, each rank 7).
    # For the remaining 4x4x3 part, we can try to find a better decomposition.
    
    # Let me build the Strassen 4x4x4 decomposition explicitly.
    
    # First, the 2x2x2 Strassen decomposition (rank 7):
    # U2 (4x7): rows for a11, a12, a21, a22
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    # V2 (4x7): rows for b11, b12, b21, b22
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    # W2 (4x7): rows for c11, c12, c21, c22
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Now for 4x4x4 Strassen:
    # Split A (4x4) into blocks: A11, A12, A21, A22 (each 2x2)
    # Split B (4x4) into blocks: B11, B12, B21, B22 (each 2x2)
    # Compute 7 intermediate 2x2x2 multiplications, each with rank 7.
    # Total rank: 7 * 7 = 49.
    
    # The 7 Strassen products for 4x4:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)(B11)
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)(B22)
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each Mi, we use the 2x2x2 Strassen decomposition.
    # The "a" matrix for each Mi is a linear combination of A blocks.
    # The "b" matrix for each Mi is a linear combination of B blocks.
    
    # Let's define the coefficient vectors for the a and b matrices.
    # A has 16 entries, indexed as A[i,j] with i,j in 0..3.
    # A11 = A[0:2, 0:2], A12 = A[0:2, 2:4], A21 = A[2:4, 0:2], A22 = A[2:4, 2:4]
    
    # For M1: a = A11 + A22
    # a[0,0] = A[0,0] + A[2,2]
    # a[0,1] = A[0,1] + A[2,3]
    # a[1,0] = A[1,0] + A[3,2]
    # a[1,1] = A[1,1] + A[3,3]
    
    # Let me define a function to create the U contribution for a given linear combination.
    
    def build_444_strassen():
        """Build U, V, W for 4x4x4 Strassen decomposition (rank 49)."""
        # 7 Strassen products, each contributing 7 rank terms (from 2x2x2 Strassen)
        # Total: 49 rank terms.
        
        U = np.zeros((16, 49))
        V = np.zeros((16, 49))
        W = np.zeros((16, 49))
        
        # For each of the 7 Strassen products Mi:
        # - The a matrix is a 2x2 linear combination of A blocks
        # - The b matrix is a 2x2 linear combination of B blocks
        # - We use the 2x2x2 Strassen U2, V2, W2 for the actual multiplication
        
        # Define the linear combinations for each Mi.
        # a_coeffs[mi] is a 4x2x2 array: a_coeffs[mi][block, row, col]
        # where block is 0,1,2,3 for A11,A12,A21,A22
        
        # M1: a = A11 + A22
        a1 = np.zeros((4, 2, 2))
        a1[0] = 1  # A11
        a1[3] = 1  # A22
        
        # M2: a = A21 + A22
        a2 = np.zeros((4, 2, 2))
        a2[2] = 1  # A21
        a2[3] = 1  # A22
        
        # M3: a = A11
        a3 = np.zeros((4, 2, 2))
        a3[0] = 1  # A11
        
        # M4: a = A22
        a4 = np.zeros((4, 2, 2))
        a4[3] = 1  # A22
        
        # M5: a = A11 + A12
        a5 = np.zeros((4, 2, 2))
        a5[0] = 1  # A11
        a5[1] = 1  # A12
        
        # M6: a = A21 - A11
        a6 = np.zeros((4, 2, 2))
        a6[0] = -1  # A11
        a6[2] = 1  # A21
        
        # M7: a = A12 - A22
        a7 = np.zeros((4, 2, 2))
        a7[1] = 1  # A12
        a7[3] = -1  # A22
        
        a_coeffs = [a1, a2, a3, a4, a5, a6, a7]
        
        # b_coeffs for each Mi
        # M1: b = B11 + B22
        b1 = np.zeros((4, 2, 2))
        b1[0] = 1  # B11
        b1[3] = 1  # B22
        
        # M2: b = B11
        b2 = np.zeros((4, 2, 2))
        b2[0] = 1  # B11
        
        # M3: b = B12 - B22
        b3 = np.zeros((4, 2, 2))
        b3[1] = 1  # B12
        b3[3] = -1  # B22
        
        # M4: b = B21 - B11
        b4 = np.zeros((4, 2, 2))
        b4[0] = -1  # B11
        b4[2] = 1  # B21
        
        # M5: b = B22
        b5 = np.zeros((4, 2, 2))
        b5[3] = 1  # B22
        
        # M6: b = B11 + B12
        b6 = np.zeros((4, 2, 2))
        b6[0] = 1  # B11
        b6[1] = 1  # B12
        
        # M7: b = B21 + B22
        b7 = np.zeros((4, 2, 2))
        b7[2] = 1  # B21
        b7[3] = 1  # B22
        
        b_coeffs = [b1, b2, b3, b4, b5, b6, b7]
        
        # Now build U, V, W
        # For each Strassen product mi (0..6) and each 2x2x2 rank term t2 (0..6):
        # global rank index = mi * 7 + t2
        
        # The a matrix for product mi is: a = sum_block a_coeffs[mi][block] * A_block
        # The 2x2x2 Strassen U2 tells us how to extract the 7 rank terms from the 2x2 a matrix.
        # U2[row, t2] gives the coefficient for a[row] in rank term t2.
        
        # So for global rank term (mi, t2):
        # U[i*m+j, global_t] = sum over the 4 blocks, sum over 2x2 positions (r,c):
        #   a_coeffs[mi][block] * U2[r, t2] * (1 if A[i*m+j] is at position (r,c) in block)
        
        # Let me define a mapping from global (i,j) to (block, r, c)
        # block = 0 (A11): i in 0..1, j in 0..1, r=i, c=j
        # block = 1 (A12): i in 0..1, j in 2..3, r=i, c=j-2
        # block = 2 (A21): i in 2..3, j in 0..1, r=i-2, c=j
        # block = 3 (A22): i in 2..3, j in 2..3, r=i-2, c=j-2
        
        def a_block_index(i, j):
            if i < 2 and j < 2:
                return 0, i, j
            elif i < 2 and j >= 2:
                return 1, i, j - 2
            elif i >= 2 and j < 2:
                return 2, i - 2, j
            else:
                return 3, i - 2, j - 2
        
        def b_block_index(j, k):
            if j < 2 and k < 2:
                return 0, j, k
            elif j < 2 and k >= 2:
                return 1, j, k - 2
            elif j >= 2 and k < 2:
                return 2, j - 2, k
            else:
                return 3, j - 2, k - 2
        
        for mi in range(7):
            for t2 in range(7):
                global_t = mi * 7 + t2
                
                # Build U column
                for i in range(4):
                    for j in range(4):
                        idx = i * 4 + j
                        block, r, c = a_block_index(i, j)
                        # Contribution from this block
                        coeff = a_coeffs[mi][block][r, c] * U2[r, t2]
                        U[idx, global_t] = coeff
                
                # Build V column
                for j in range(4):
                    for k in range(4):
                        idx = j * 4 + k
                        block, r, c = b_block_index(j, k)
                        coeff = b_coeffs[mi][block][r, c] * V2[r, t2]
                        V[idx, global_t] = coeff
                
                # Build W column
                # W is indexed by (k, i) for C[k*n + i]... wait, W index is k*n + i
                # where i is row of C, k is column of C.
                # C is 4x4, so i in 0..3, k in 0..3.
                # C11 = M1 + M4 - M5 + M7 (top-left 2x2 block)
                # C12 = M3 + M5 (top-right 2x2 block)
                # C21 = M2 + M4 (bottom-left 2x2 block)
                # C22 = M1 - M2 + M3 + M6 (bottom-right 2x2 block)
                
                # For C[i,k]:
                # if i<2, k<2: C11, coeff from Mi: [1, 0, 0, 1, -1, 0, 1][mi]
                # if i<2, k>=2: C12, coeff from Mi: [0, 0, 1, 0, 1, 0, 0][mi]
                # if i>=2, k<2: C21, coeff from Mi: [0, 1, 0, 1, 0, 0, 0][mi]
                # if i>=2, k>=2: C22, coeff from Mi: [1, -1, 1, 0, 0, 1, 0][mi]
                
                # The 2x2x2 Strassen W2 gives the coefficients for the 4 entries of the 2x2 result.
                # W2[row, t2] where row is 0=c11, 1=c12, 2=c21, 3=c22 within the 2x2 block.
                
                # For global (i,k):
                # Determine which C block it's in, and the local (r,c) within that block.
                
                if i < 2 and k < 2:
                    c_block = 0  # C11
                    r, c = i, k
                    c_block_coeff = [1, 0, 0, 1, -1, 0, 1][mi]
                elif i < 2 and k >= 2:
                    c_block = 1  # C12
                    r, c = i, k - 2
                    c_block_coeff = [0, 0, 1, 0, 1, 0, 0][mi]
                elif i >= 2 and k < 2:
                    c_block = 2  # C21
                    r, c = i - 2, k
                    c_block_coeff = [0, 1, 0, 1, 0, 0, 0][mi]
                else:
                    c_block = 3  # C22
                    r, c = i - 2, k - 2
                    c_block_coeff = [1, -1, 1, 0, 0, 1, 0][mi]
                
                # W index = k * 4 + i
                w_idx = k * 4 + i
                # The local row in the 2x2 result block:
                # For C11: local row = i, local col = k. W2 row index = i*2 + k? No.
                # W2 is 4x7, rows are c11, c12, c21, c22.
                # For a 2x2 block, the local indices (r,c) map to:
                # (0,0) -> c11 -> W2 row 0
                # (0,1) -> c12 -> W2 row 1
                # (1,0) -> c21 -> W2 row 2
                # (1,1) -> c22 -> W2 row 3
                local_row = r * 2 + c
                W[w_idx, global_t] = c_block_coeff * W2[local_row, t2]
        
        return U, V, W
    
    # Build the 4x4x4 Strassen decomposition
    U_444, V_444, W_444 = build_444_strassen()
    
    # Verify: check a few entries
    # For the schoolbook part, let's just use it for the 4x4x3 part.
    # Total rank = 49 + 4*4*3 = 49 + 48 = 97
    
    # Actually, let me try to do better for the 4x4x3 part.
    # For 2x2x3, we can potentially do better than 2*2*3=12.
    # Let's use Strassen for the 2x2x2 part (rank 7) and schoolbook for the 2x2x1 part (rank 4).
    # Total for 2x2x3: 7 + 4 = 11 per 2x2x3 block.
    # For 4x4x3: we have 4 such blocks (2 in row, 1 in col... wait)
    
    # Actually, 4x4x3: A is 4x4, B is 4x3, C is 4x3.
    # Split A into 2x2 blocks: A11, A12, A21, A22 (each 2x2)
    # B is 4x3, split into 2x2 and 2x1:
    # B11 (2x2), B12 (2x1) for top half
    # B21 (2x2), B22 (2x1) for bottom half
    
    # C11 (2x2) = A11*B11 + A12*B21
    # C12 (2x1) = A11*B12 + A12*B22
    # C21 (2x2) = A21*B11 + A22*B21
    # C22 (2x1) = A21*B12 + A22*B22
    
    # This is getting complex. Let me just use schoolbook for the 4x4x3 part for now.
    
    # Build schoolbook for 4x4x3
    U_443 = np.zeros((16, 48))
    V_443 = np.zeros((12, 48))  # 4*3
    W_443 = np.zeros((12, 48))  # 3*4
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                U_443[i * 4 + j, col] = 1
                V_443[j * 3 + k, col] = 1
                W_443[k * 4 + i, col] = 1
                col += 1
    
    # Combine: U_total (16, 97), V_total (28, 97), W_total (28, 97)
    # Wait, V is (m*p, rank) = (4*7, rank) = (28, rank)
    # W is (p*n, rank) = (7*4, rank) = (28, rank)
    
    U_total = np.zeros((16, 97))
    V_total = np.zeros((28, 97))
    W_total = np.zeros((28, 97))
    
    # Copy 4x4x4 part (first 4 columns of B, first 4 rows of C)
    U_total[:, 0:49] = U_444
    V_total[0:16, 0:49] = V_444  # B[0:4, 0:4]
    W_total[0:16, 0:49] = W_444  # C[0:4, 0:4], index k*4+i for k,i in 0..3
    
    # Copy 4x4x3 part (last 3 columns of B, last 3 rows of C)
    U_total[:, 49:97] = U_443
    # B[0:4, 4:7] -> V_total[0:12, 49:97]
    V_total[0:12, 49:97] = V_443
    # C[0:4, 4:7] -> W_total[16:28, 49:97]
    # W index = k*4 + i, for k in 4..6, i in 0..3
    # W_total row = k*4 + i
    for k in range(3):
        for i in range(4):
            W_total[(4 + k) * 4 + i, 49:97] = W_443[k * 4 + i, :]
    
    # Verify the decomposition
    # Check: for all i,j,k: sum_t U[i*4+j, t] * V[j*7+k, t] * W[k*4+i, t] should equal 1
    
    # Let's do a quick spot check
    errors = []
    for i in range(4):
        for j in range(4):
            for k in range(7):
                val = 0
                for t in range(97):
                    val += U_total[i*4+j, t] * V_total[j*7+k, t] * W_total[k*4+i, t]
                if abs(val - 1) > 1e-10:
                    errors.append((i, j, k, val))
    
    if len(errors) > 0:
        # Fallback to schoolbook if verification fails
        return {
            "U": U_sb.tolist(),
            "V": V_sb.tolist(),
            "W": W_sb.tolist()
        }
    
    return {
        "U": U_total.tolist(),
        "V": V_total.tolist(),
        "W": W_total.tolist()
    }
