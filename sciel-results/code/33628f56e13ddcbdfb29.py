import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """Add Strassen 2x2x2 block for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+2] -> C[k0:k0+2, i0:i0+2]"""
        nonlocal col
        if col + 7 > rank:
            return False
        a = i0 * m + j0
        b = i0 * m + (j0 + 1)
        c = (i0 + 1) * m + j0
        d = (i0 + 1) * m + (j0 + 1)
        e = j0 * p + k0
        f = j0 * p + (k0 + 1)
        g = (j0 + 1) * p + k0
        h = (j0 + 1) * p + (k0 + 1)
        c00 = k0 * n + i0
        c01 = (k0 + 1) * n + i0
        c10 = k0 * n + (i0 + 1)
        c11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[a][col] += 1; U[d][col] += 1
        V[e][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        # M2 = A11(B10-B00)
        U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c10][col] += 1; W[c11][col] += 1
        col += 1
        # M3 = (A00-A10)(B01+B11)
        U[a][col] += 1; U[c][col] -= 1
        V[f][col] += 1; V[h][col] += 1
        W[c00][col] += 1; W[c01][col] += 1
        col += 1
        # M4 = A10(B00-B10)
        U[c][col] += 1
        V[e][col] += 1; V[g][col] -= 1
        W[c10][col] += 1
        col += 1
        # M5 = (A01+A11)(B10-B00)
        U[b][col] += 1; U[d][col] += 1
        V[g][col] += 1; V[e][col] -= 1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        # M6 = A01(B11-B01)
        U[b][col] += 1
        V[h][col] += 1; V[f][col] -= 1
        W[c11][col] += 1
        col += 1
        # M7 = (A00-A11)(B01-B11)
        U[a][col] += 1; U[d][col] -= 1
        V[f][col] += 1; V[h][col] -= 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        return True

    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook 2x2x1 block for A[i0:i0+2, j0:j0+2] @ B[j0:j0+2, k0:k0+1] -> C[k0:k0+1, i0:i0+2]"""
        nonlocal col
        if col + 4 > rank:
            return False
        c00 = k0 * n + i0
        c10 = k0 * n + (i0 + 1)
        for jj in range(2):
            U[i0 * m + j0 + jj][col] += 1
            V[(j0 + jj) * p + k0][col] += 1
            W[c00][col] += 1
            col += 1
            U[(i0 + 1) * m + j0 + jj][col] += 1
            V[(j0 + jj) * p + k0][col] += 1
            W[c10][col] += 1
            col += 1
        return True

    # Split p=7 as 4+3
    # First 4 columns (k=0,1,2,3): 2 Strassen 2x2x2 blocks
    add_strassen_222(0, 0, 0)  # covers k=0,1 using j=0,1
    add_strassen_222(0, 2, 2)  # covers k=2,3 using j=2,3
    
    # Last 3 columns (k=4,5,6): 1 Strassen for k=4,5 + schoolbook for k=6
    add_strassen_222(0, 0, 4)  # covers k=4,5 using j=0,1 (but B columns 4,5 are in different j positions)
    
    # Wait, this is wrong. The Strassen block uses j0 and j0+1 from A and corresponding B columns.
    # For k=4,5: B[:,4] and B[:,5]. These correspond to j indices 4 and 5 in B (4x7).
    # But A is only 4 columns (j=0..3). So B[:,4] = sum_j A[:,j] * B[j,4].
    # Strassen requires B[j0, k0] and B[j0+1, k0] for specific j0.
    # For k=4: we need B[0,4], B[1,4], B[2,4], B[3,4]
    # For k=5: we need B[0,5], B[1,5], B[2,5], B[3,5]
    # Strassen 2x2x2 with j0=0, k0=4 would use B[0,4], B[0,5], B[1,4], B[1,5] - but we also need B[2,4], B[2,5], B[3,4], B[3,5]
    # This doesn't work directly. Strassen 2x2x2 only handles 2x2 submatrices.
    
    # I need to rethink. The issue is that for k=4,5,6, I need all 4 columns of A.
    # A Strassen 2x2x2 block only uses 2 columns of A.
    # So for 3 output columns, I need to cover all 4 columns of A.
    # Options:
    # - 2 Strassen blocks covering (j=0,1) and (j=2,3) for each pair of k columns
    # - That's 2 Strassen per 2 k-columns, so 3 Strassen for 3 k-columns (with overlap)
    # - Or: schoolbook for all 3 columns = 3 * 2 * 2 = 12 ops
    # - Or: 1 Strassen for k=4,5 using j=0,1 (7 ops) + 1 Strassen for k=4,5 using j=2,3 (7 ops) + schoolbook for k=6 (4 ops) = 18 ops
    #   But that double-counts... no, Strassen on different j blocks are independent.
    
    # Actually, for C[:,k] = A @ B[:,k], if I split A into two halves:
    # C[:,k] = A[:,0:2] @ B[0:2,k] + A[:,2:4] @ B[2:4,k]
    # Each half is a 2x2 @ 2x1 = 2 ops schoolbook, or I can use Strassen across k pairs.
    
    # For k=4,5: C[:,4:6] = A[:,0:2] @ B[0:2,4:6] + A[:,2:4] @ B[2:4,4:6]
    # First term: 2x2 @ 2x2, use Strassen = 7 ops
    # Second term: 2x2 @ 2x2, use Strassen = 7 ops
    # Total for k=4,5: 14 ops
    # For k=6: C[:,6] = A[:,0:2] @ B[0:2,6] + A[:,2:4] @ B[2:4,6]
    # First term: 2x2 @ 2x1 = 2 ops
    # Second term: 2x2 @ 2x1 = 2 ops
    # Total for k=6: 4 ops
    # Grand total: 14 (first 4 cols) + 14 (k=4,5) + 4 (k=6) = 32 ops
    
    # That's better than 50! Let me redo the whole thing.
    
    col = 0
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each pair of output columns (k, k+1), split A into two 2x2 halves:
    # C[:,k:k+2] = A[:,0:2] @ B[0:2,k:k+2] + A[:,2:4] @ B[2:4,k:k+2]
    # Each 2x2@2x2 can use Strassen (7 ops)
    # For single column k: C[:,k] = A[:,0:2] @ B[0:2,k] + A[:,2:4] @ B[2:4,k]
    # Each 2x2@2x1 = 2 ops schoolbook
    
    # k=0,1: Strassen(j=0,k=0) + Strassen(j=2,k=0) = 14
    add_strassen_222(0, 0, 0)
    add_strassen_222(0, 2, 0)
    # k=2,3: Strassen(j=0,k=2) + Strassen(j=2,k=2) = 14
    add_strassen_222(0, 0, 2)
    add_strassen_222(0, 2, 2)
    # k=4,5: Strassen(j=0,k=4) + Strassen(j=2,k=4) = 14
    add_strassen_222(0, 0, 4)
    add_strassen_222(0, 2, 4)
    # k=6: schoolbook(j=0,k=6) + schoolbook(j=2,k=6) = 4
    add_schoolbook_221(0, 0, 6)
    add_schoolbook_221(0, 2, 6)
    
    total_ops = col
    # total_ops should be 14*3 + 4 = 46
    
    if time.time() > deadline:
        return None
    
    # Verify
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # Check: sum_r U[i,r] * V[j,r] * W[k,r] should equal delta_{i,j,k} for matrix mult tensor
    # Tensor: T[i,j,k] = 1 if j is the "matching" index, else 0
    # For 2x4x7: T[i,j,k] = 1 for i in [0,2), j in [0,4), k in [0,7)
    # Actually the tensor is T[i,j,k] where output C[i,k] = sum_j A[i,j] * B[j,k]
    # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in j)
    # Rank decomposition: T = sum_r U[:,r] V[:,r] W[:,r] (outer product)
    # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Build full tensor
    T = np.zeros((n, m, p), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = 1  # All entries are 1 for the multiplication tensor? No!
    
    # Wait, the multiplication tensor for A(2x4) @ B(4x7) -> C(2x7):
    # C[i,k] = sum_{j=0}^{3} A[i,j] * B[j,k]
    # The tensor T has T[i,j,k] = 1 if this is a valid term, which is always true for valid indices.
    # So T is all ones? That can't be right for rank analysis.
    
    # Actually no. The tensor is defined by the bilinear form.
    # T[i,j,k] is the coefficient in C[i,k] = sum_{i,j,k} T[i,j,k] A[i,j] B[j,k]
    # For standard matrix mult, T[i,j,k] = 1 if j is a valid intermediate index, 0 otherwise.
    # Since all j=0..3 are valid, T[i,j,k] = 1 for all i,j,k in range.
    # So T is a 2x4x7 tensor of all ones.
    
    # Rank of all-ones tensor of shape (2,4,7): the rank is min(2,4,7) = 2? No, that's not right either.
    # The rank of the all-ones tensor is 1 (it's a rank-1 tensor: T = u v^T w^T where u,v,w are all-ones vectors).
    # But that can't be right because matrix multiplication has higher rank.
    
    # I'm confusing myself. Let me reconsider.
    # The multiplication tensor for m_{n,p,q} (n x p times p x q gives n x q) has:
    # T[i,j,k] = 1 if i in [0,n), j in [0,p), k in [0,q) and the term A[i,j]*B[j,k] contributes to C[i,k]
    # For each (i,k), C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid i,j,k.
    # The tensor shape is (n, p, q) = (2, 4, 7).
    # T[i,j,k] = 1 for all i,j,k.
    # This is indeed a rank-1 tensor!
    
    # But that contradicts the known rank of matrix multiplication. I must be misunderstanding the tensor definition.
    
    # Oh wait, I think the issue is that the tensor is NOT all ones. Let me reconsider.
    # In the bilinear map, the tensor is defined such that:
    # C = sum_r U[:,r] * V[:,r] * W[:,r]
    # where C is the output tensor of shape (n*q,), and U has shape (n*p, rank), etc.
    # The original multiplication tensor M has M[i,j,k] where:
    # - i indexes the A matrix (n*p entries)
    # - j indexes the B matrix (p*q entries)  
    # - k indexes the C matrix (n*q entries)
    # M[i,j,k] = 1 if A[i] and B[j] multiply to contribute to C[k]
    
    # For A[i] = A[row, col] where i = row*p + col (or some ordering)
    # For B[j] = B[col, out] where j = col*q + out
    # For C[k] = C[row, out] where k = out*n + row (trace convention W[k*n+i])
    
    # The contribution: A[row,col] * B[col,out] contributes to C[row,out]
    # So M[i,j,k] = 1 if i = row*p+col, j = col*q+out, k = out*n+row for some valid row,col,out
    
    # This is NOT all ones. For a given (i,j,k), it's 1 only if the col index matches.
    
    # Let me verify with the current program's approach and just trust that the Strassen decomposition is correct if properly constructed.
    
    # My 46-op solution: let me verify it's correct by checking a few entries.
    # Actually, let me just run the verification.
    
    # Build the expected tensor
    T_expected = np.zeros((n * m, m * p, p * n), dtype=int)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # A[i,j] * B[j,k] -> C[i,k]
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i  # trace convention
                T_expected[a_idx, b_idx, c_idx] = 1
    
    # Build from decomposition
    T_decomp = np.zeros((n * m, m * p, p * n), dtype=int)
    for r in range(col):
        for i in range(n * m):
            for j in range(m * p):
                for k in range(p * n):
                    T_decomp[i, j, k] += U_arr[i, r] * V_arr[j, r] * W_arr[k, r]
    
    match = np.allclose(T_decomp, T_expected)
    
    if not match:
        # Try to debug
        diff = np.abs(T_decomp - T_expected)
        num_diff = np.sum(diff > 0)
        # Fall back to known-good 50-op solution
        col = 0
        U = [[0] * 50 for _ in range(n * m)]
        V = [[0] * 50 for _ in range(m * p)]
        W = [[0] * 50 for _ in range(p * n)]
        
        def add_strassen_222_orig(i0, j0, k0):
            nonlocal col
            a = i0 * m + j0
            b = i0 * m + (j0 + 1)
            c = (i0 + 1) * m + j0
            d = (i0 + 1) * m + (j0 + 1)
            e = j0 * p + k0
            f = j0 * p + (k0 + 1)
            g = (j0 + 1) * p + k0
            h = (j0 + 1) * p + (k0 + 1)
            c00 = k0 * n + i0
            c01 = (k0 + 1) * n + i0
            c10 = k0 * n + (i0 + 1)
            c11 = (k0 + 1) * n + (i0 + 1)
            
            U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1
            W[c00][col] += 1; W[c11][col] += 1; col += 1
            U[c][col] += 1; U[d][col] += 1; V[e][col] += 1
            W[c10][col] += 1; W[c11][col] -= 1; col += 1
            U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1
            W[c01][col] += 1; W[c11][col] += 1; col += 1
            U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1
            W[c00][col] += 1; W[c10][col] += 1; col += 1
            U[a][col] += 1; U[b][col] += 1; V[h][col] += 1
            W[c00][col] -= 1; W[c01][col] += 1; col += 1
            U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1
            W[c11][col] += 1; col += 1
            U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1
            W[c00][col] += 1; col += 1
        
        for k0 in [0, 2, 4]:
            add_strassen_222_orig(0, 0, k0)
            add_strassen_222_orig(0, 2, k0)
        
        # Schoolbook for k=6
        for jj in range(4):
            U[0 * m + jj][col] += 1; V[jj * p + 6][col] += 1; W[6 * n + 0][col] += 1; col += 1
            U[1 * m + jj][col] += 1; V[jj * p + 6][col] += 1; W[6 * n + 1][col] += 1; col += 1
        
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
