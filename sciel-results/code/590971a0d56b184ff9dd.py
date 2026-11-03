"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS

    # Bini-Capovani-Lotti-Romani rank-23 decomposition for 3x3x3
    # U333: (9, 23), V333: (9, 23), W333: (9, 23)
    # Each factor row corresponds to (i,j) for U, (j,k) for V, (k,i) for W
    # using 0-indexed i,j,k in {0,1,2}
    
    # Known rank-23 decomposition of the 3x3x3 matrix multiplication tensor.
    # This is the Bini et al. (1978) construction.
    # Format: U[i*3+j, c] = U333[i,j,c]
    
    # We'll use a verified rank-23 decomposition.
    # The following is a known valid decomposition (Bini 1978, rank 23).
    
    # Let me construct it via a block approach: use Strassen for 2x2x2 sub-block
    # and fill in the rest. Actually, let me use a simpler but still better approach.
    
    # Strategy: embed a 2x2x2 Strassen (rank 7) and use trivial for the rest of 3x3x3
    # 3x3x3 trivial = 27. With one 2x2x2 Strassen: 7 + (27-8) = 7+19 = 26. Worse than 23.
    
    # Let me just use the trivial 27 for 3x3x3 and embed. That gives rank 27 for 3x4x7.
    # 27 < 84, so still a big improvement!
    
    # Actually, let me try to get rank 23 properly.
    # Here's a known rank-23 decomposition for 3x3x3 (Bini et al.):
    
    # I'll use a different approach: the "3x3x3 rank 23" is well-known but the
    # explicit entries are complex. Let me use a simpler construction.
    
    # Block Strassen for 3x3x3:
    # Split 3x3 into [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # This is asymmetric. Let me try a different split.
    
    # Actually, the simplest reliable improvement: use trivial 3x4x3 = 36, embed into 3x4x7.
    # Rank 36 < 84. Let me also try to do better.
    
    # Let me use a 2x2x2 Strassen block embedded in 3x4x7 directly:
    # This gives rank 7 + (84 - 8) = 83. Only saves 1.
    
    # Better: use 3x3x3 trivial (rank 27) embedded: rank 27.
    # Even better: if I can find a rank < 27 for 3x3x3.
    
    # Known: rank(M_{3,3,3}) is between 19 and 23. The best known upper bound is 23.
    # Let me just use 27 (trivial 3x3x3) for now, which already gives rank 27 << 84.
    
    # But wait - I should try to get the actual rank-23. Let me implement it.
    
    # Here is a known rank-23 decomposition for 3x3x3 (from Bini et al. 1978):
    # I'll represent it as lists of (i,j,k, coefficient) tuples.
    
    # Actually, let me use a cleaner approach. The following is a valid rank-23 
    # decomposition. Each term is (U_idx, V_idx, W_idx, coeff) where indices are 
    # linear: U_idx = i*3+j, V_idx = j*3+k, W_idx = k*3+i for i,j,k in {0,1,2}.
    
    # Bini rank-23 for 3x3x3:
    # I'll use the construction from the literature.
    
    # Let me just go with trivial 3x3x3 = 27 for reliability, then try to improve.
    
    rank_333 = 27
    U333 = [[0]*rank_333 for _ in range(9)]
    V333 = [[0]*rank_333 for _ in range(9)]
    W333 = [[0]*rank_333 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U333[i*3+j][col] = 1
                V333[j*3+k][col] = 1
                W333[k*3+i][col] = 1
                col += 1
    
    # Embed 3x3x3 into 3x4x7:
    # U: (12, 27) - rows i*4+j for i,j in {0,1,2} get U333[i*3+j], row i*4+3 = 0
    # V: (28, 27) - rows j*7+k for j,k in {0,1,2} get V333[j*3+k], others = 0
    # W: (21, 27) - rows k*3+i for k,i in {0,1,2} get W333[k*3+i], k>=3 rows = 0
    
    U = [[0]*rank_333 for _ in range(n*m)]  # (12, 27)
    V = [[0]*rank_333 for _ in range(m*p)]  # (28, 27)
    W = [[0]*rank_333 for _ in range(p*n)]  # (21, 27)
    
    for i in range(3):
        for j in range(3):
            U[i*m + j] = U333[i*3 + j][:]
    
    for j in range(3):
        for k in range(3):
            V[j*p + k] = V333[j*3 + k][:]
    
    for k in range(3):
        for i in range(3):
            W[k*n + i] = W333[k*3 + i][:]
    
    # Verify correctness
    if time.time() < deadline - 10:
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check: for random A (3x4), B (4x7), does sum_c U@A * V@B * W@B^T give AB?
        # Actually, the tensor decomposition means:
        # (AB)_{i,k} = sum_j A_{i,j} B_{j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c} ... 
        # Wait, the convention is W[k*n+i, r]. Let me verify.
        
        # The decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # where T[i,j,k] = delta (i.e., T is the matrix mult tensor: T[i,j,k] = 1 if this is a valid (i,j,k) triple, else 0)
        # Actually T[i,j,k] = 1 for all valid i,j,k since M_{i,j,k} = A_{i,j} * B_{j,k}
        
        # Let me verify with random matrices
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        
        # Compute AB
        AB = A_test @ B_test
        
        # Compute via decomposition: (AB)_{i,k} = sum_c (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k]) ... 
        # No, that's not right either. The tensor decomposition is:
        # sum_j A_{i,j} B_{j,k} = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] summed over j and c
        # Wait, I think the correct interpretation is:
        # The matrix multiplication tensor has entries T[i,j,k] = 1 (for the structure tensor)
        # and the decomposition is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # But T[i,j,k] should be 1 for all i,j,k in range, which can't be right.
        
        # Let me re-read: "exactly decompose the matrix multiplication tensor"
        # The matrix multiplication tensor M_{n,m,p} has M[i,j,k] = 1 for all valid indices.
        # No wait, that's the structure tensor. The decomposition means:
        # For any A (n x m) and B (m x p): (AB)[i,k] = sum_j A[i,j] B[j,k]
        # = sum_c (sum_j U[i*m+j,c] A[i,j]) ... no.
        
        # I think the correct check is:
        # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1 if we're looking at the structure tensor entry,
        # but actually the structure tensor for matrix mult is: T[i,j,k] = delta_{...}
        
        # Let me just check: does sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k?
        # No, that's not the right check. The right check is:
        # For the bilinear map: (A,B) -> AB, we need
        # (AB)[i,k] = sum_j A[i,j] B[j,k]
        # The rank-r decomposition gives:
        # (AB)[i,k] = sum_c (sum_j U[i*m+j,c] A[i,j]) * ... 
        
        # I think the correct formula is:
        # (AB)[i,k] = sum_c (U^T A)_c * (B V)_c * W_{k*n+i, c}
        # where (U^T A)_c = sum_{i,j} U[i*m+j,c] A[i,j] and (B V)_c = sum_{j,k} V[j*p+k,c] B[j,k]
        
        # Let me just verify numerically.
        
        # (U^T A)_c = sum_{i,j} U[i*m+j,c] * A[i,j]
        # (B V)_c = sum_{j,k} B[j,k] * V[j*p+k,c]
        # result[i,k] = sum_c (U^T A)_c * (B V)_c * W[k*n+i, c]
        
        UA = np.array([np.sum(U_arr[:, c] * A_test.ravel()) for c in range(rank_333)])
        BV = np.array([np.sum(V_arr[:, c] * B_test.ravel()) for c in range(rank_333)])
        
        result = np.zeros((n, p))
        for c in range(rank_333):
            for k in range(p):
                for i in range(n):
                    result[i, k] += UA[c] * BV[c] * W_arr[k*n + i, c]
        
        # This doesn't seem right either. Let me think about the tensor decomposition more carefully.
        
        # The matrix multiplication tensor T in R^{(n*m) x (m*p) x (p*n)} is defined by:
        # T[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' and j = j', i = i' ... 
        # No. T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k.
        
        # The rank-r decomposition: T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # So: for all i,j,k: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
        
        # Let me verify this directly.
        
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = sum(U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c] for c in range(rank_333))
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        
        if max_err > 1e-9:
            # Fallback to trivial
            rank = n * m * p
            U = [[0] * rank for _ in range(n * m)]
            V = [[0] * rank for _ in range(m * p)]
            W = [[0] * rank for _ in range(p * n)]
            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j][column] = 1
                        V[j * p + k][column] = 1
                        W[k * n + i][column] = 1
                        column += 1
        else:
            return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
