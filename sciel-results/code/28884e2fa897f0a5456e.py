import time
import numpy as np

DIMENSIONS = [2, 2, 14]

def _strassen_7():
    """Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (U_dict, V_dict, W_dict) with keys being local indices.
    U local: 0=A00, 1=A01, 2=A10, 3=A11
    V local: 0=B00, 1=B01, 2=B10, 3=B11
    W local: 0=C00, 1=C01, 2=C10, 3=C11
    """
    return [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),       # M1
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),             # M2
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),             # M3
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),             # M4
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),             # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),             # M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),             # M7
    ]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W form a valid tensor decomposition.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    W uses trace convention: W[k*n+i, r] where C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    rank = len(U[0])
    
    # Pick random integer values to test
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randint(-2, 3, size=(n, m))
        B = np.random.randint(-2, 3, size=(m, p))
        C_expected = A @ B

        C_computed = np.zeros((n, p))
        for r in range(rank):
            u_col = np.array([U[i][r] for i in range(n*m)])
            v_col = np.array([V[j][r] for j in range(m*p)])
            w_col = np.array([W[k][r] for k in range(p*n)])

            uA = np.zeros(n)
            for i in range(n):
                for j in range(m):
                    uA[i] += U[i*m + j][r] * A[i, j]
            
            vB = np.zeros(p)
            for j in range(m):
                for k in range(p):
                    vB[k] += V[j*p + k][r] * B[j, k]
            
            for k in range(p):
                for i in range(n):
                    C_computed[i, k] += W[k*n + i][r] * uA[i] * vB[k]
        
        if not np.allclose(C_computed, C_expected, atol=tol):
            return False
    return True

def _find_kronecker_decomposition(n, m, p, deadline):
    """Use Kronecker product to find minimal rank decomposition."""
    # The tensor T[i, j, k] = 1 if C[i,k] depends on A[i,j]*B[j,k]
    # Reshape into matrix: for each (i,k) output, the input is (j)
    # T as a 3-tensor of shape (n*m, m*p, p*n) with trace convention
    
    # Build the full tensor T[a, b, c] where:
    # a = i*m + j (input A index)
    # b = j*p + k (input B index)  
    # c = k*n + i (output C index, trace convention)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Reshape to matrix for SVD: (n*m, m*p * p*n)
    T_mat = T.reshape(n*m, m*p * p*n)
    
    # SVD to find rank
    try:
        U_svd, s, Vt_svd = np.linalg.svd(T_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    
    # Find numerical rank (singular values > tolerance)
    tol_svd = 1e-10
    num_rank = np.sum(s > tol_svd)
    
    if num_rank <= 0:
        return None
    
    # The SVD gives: T_mat = U_svd[:, :r] @ diag(s[:r]) @ Vt_svd[:r, :]
    # But we need a *tensor* decomposition, not just matrix rank
    # Matrix rank is an upper bound on tensor rank
    
    # For this specific structure, let's try to extract a tensor decomposition
    # from the SVD by reshaping back
    
    # Actually, for this block-diagonal structure, the tensor rank should equal
    # the number of non-zero singular values if the structure allows it.
    # Let's verify by trying to construct U, V, W from SVD factors.
    
    # The SVD approach gives matrix rank, which is an upper bound on tensor rank.
    # For this problem, we know the answer is at most 49 (block Strassen).
    # Let's check if the Kronecker structure reveals a lower rank.
    
    # Alternative: use the fact that T = sum over blocks of block_tensors
    # Each block tensor has rank 7 (Strassen)
    # The total tensor rank is at most 49
    
    # Try to find if there's shared structure by looking at the U-matrix rank
    # In the 49-term decomposition, U has only 7 distinct column patterns
    # This means the U-matrix (4 x 49) has rank at most 7
    
    # The key question: can we do better than 49 by merging across blocks?
    # Since V and W are block-local, we need at least 7 terms per "U pattern"
    # unless we can share V/W factors across blocks.
    
    # Let's try a greedy approach: for each pair of terms, check if they can be merged
    # (i.e., if their tensor product can be expressed as a single term)
    
    # For now, return the SVD-based rank as information
    return int(num_rank)

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 7 blocks
    
    # First, check the Kronecker product structure to see if we can do better
    deadline = start_time + 140  # 10s safety margin
    
    # Try to find a lower-rank decomposition using the Kronecker insight
    # The key observation: the 49-term Strassen decomposition has U-matrix rank 7
    # (only 7 distinct U patterns). This suggests potential for merging.
    
    # Let's build the 49-term decomposition and then try to reduce its rank
    # by identifying linear dependencies in the full tensor space.
    
    rank = 7 * num_blocks  # Start with 49
    U = [[0] * rank for _ in range(n * m)]  # 4 x 49
    V = [[0] * rank for _ in range(m * p)]  # 28 x 49
    W = [[0] * rank for _ in range(p * n)]  # 28 x 49

    terms = _strassen_7()

    col = 0
    for b in range(num_blocks):
        # V global offsets for block b: B00->2b, B01->2b+1, B10->14+2b, B11->14+2b+1
        v_offsets = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        # W global offsets for block b: C00->4b, C01->4b+2, C10->4b+1, C11->4b+3
        w_offsets = [4*b, 4*b+2, 4*b+1, 4*b+3]

        for u_dict, v_dict, w_dict in terms:
            for idx, val in u_dict.items():
                U[idx][col] += val
            for l_idx, val in v_dict.items():
                V[v_offsets[l_idx]][col] += val
            for l_idx, val in w_dict.items():
                W[w_offsets[l_idx]][col] += val
            col += 1

    # Now try to reduce rank by finding linear dependencies
    # Build the tensor as a matrix and check if we can remove columns
    # that are linear combinations of others in the tensor sense.
    
    # For a tensor decomposition T = sum_r u_r ⊗ v_r ⊗ w_r,
    # two terms can be merged if their contribution can be absorbed.
    # This is equivalent to checking if the matrix formed by "flattening"
    # has rank less than the number of terms.
    
    # Build the full tensor T from the 49-term decomposition
    T = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u_vec = np.array([U[i][r] for i in range(n*m)])
        v_vec = np.array([V[j][r] for j in range(m*p)])
        w_vec = np.array([W[k][r] for k in range(p*n)])
        T += np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
    
    # Check if the tensor has lower rank by trying to find a smaller decomposition
    # Use the fact that for this block-diagonal structure, we can analyze each "U pattern" group
    
    # Group terms by their U pattern
    u_patterns = {}
    for r in range(rank):
        u_key = tuple(U[i][r] for i in range(n*m))
        if u_key not in u_patterns:
            u_patterns[u_key] = []
        u_patterns[u_key].append(r)
    
    # For each U pattern group, the V and W vectors span a subspace
    # We need at least the rank of the combined V,W space for that pattern
    
    # For Strassen, each U pattern appears exactly 7 times (once per block)
    # The V vectors for a given pattern across blocks are orthogonal (support on different blocks)
    # Same for W vectors
    # So the minimum number of terms for each U pattern is 7 (one per block)
    # Total minimum: 7 * 7 = 49
    
    # This confirms that 49 is likely optimal for this structure.
    # Let's verify with a quick check that no further reduction is possible.
    
    # The Kronecker product approach: I_14 ⊗ A acting on B
    # The matrix representation is 28x28, and its rank is 28 (full rank for generic A)
    # But we need the *bilinear* rank, not the linear rank.
    
    # Given the analysis, 49 appears to be the minimal rank for this decomposition.
    # Return the 49-term Strassen decomposition.
    
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
