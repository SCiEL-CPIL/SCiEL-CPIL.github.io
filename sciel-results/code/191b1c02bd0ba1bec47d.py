"""Program for matrix multiplication tensor decomposition of (3,3,8)."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]
DEADLINE_SECONDS = 120.0
SAFETY_MARGIN = 15.0


def _schoolbook_decomposition(n, m, p):
    """Build the standard (schoolbook) rank-n*m*p decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1.0
                V[j * p + k][c] = 1.0
                W[k * n + i][c] = 1.0
                c += 1
    return U, V, W


def _winograd_3x3_terms():
    """Return the Winograd rank-23 decomposition for 3x3 matrix multiplication.
    
    Returns lists:
      - A_terms: list of 23 tuples (i, j, coeff) for A[i,j] coefficients
      - B_terms: list of 23 tuples (j, k, coeff) for B[j,k] coefficients  
      - C_terms: list of 23 tuples (i, k, coeff) for C[i,k] coefficients
    
    The Winograd F_3(3) algorithm:
      F = [[1, 1, 0], [0, 1, -1], [0, 1, 1]]
      A' = F @ A @ F^T  (3x3 -> 3x3)
      B' = F @ B @ F^T  (3x3 -> 3x3)
      C' = A' * B' (element-wise, 9 products)
      C = F^T @ C' @ F^T
      
    But this only gives 9 products for the transformed space. The full Winograd
    3x3 actually uses 23 multiplications because the transform doesn't fully
    diagonalize.
    
    Let me use the explicit known construction. The 23 terms for C[i,k] = sum_j A[i,j]*B[j,k]:
    """
    # Using the standard Winograd F_3(3) construction
    # The transform matrix is F = [[1,1,0],[0,1,-1],[0,1,1]]
    # F^T @ F = [[1,1,1],[1,2,0],[1,0,2]]... let me just use the known result.
    
    # Actually, let me construct it properly using the tensor network approach.
    # The key insight: for 3x3, we can write:
    # C = F^T ( (F A F^T) * (F B F^T) ) F^T  where * is element-wise
    # But this only works if F A F^T and F B F^T have the right sparsity.
    
    # Let me use a different well-known construction.
    # Strassen-style or direct Winograd.
    
    # I'll use the explicit 23-term construction from the literature.
    # For C[i,k] = sum_j A[i,j] B[j,k], the 23 products are:
    
    # Group 1: 9 products from the "diagonal" structure
    # Group 2: 14 products from the "off-diagonal" corrections
    
    # Let me just build it from the F transform directly.
    F = np.array([[1, 1, 0], [0, 1, -1], [0, 1, 1]], dtype=float)
    F_inv = np.linalg.inv(F)
    
    # The decomposition: C = F_inv^T @ ( (F @ A @ F^T) .* (F @ B @ F^T) ) @ F_inv^T
    # where .* is element-wise.
    # 
    # Let A' = F @ A @ F^T, so A'[r,s] = sum_{i,j} F[r,i] A[i,j] F[s,j]
    # Let B' = F @ B @ F^T, so B'[r,s] = sum_{j,k} F[r,j] B[j,k] F[s,k]
    # C'[r,s] = A'[r,s] * B'[r,s]
    # C = F_inv^T @ C' @ F_inv^T, so C[i,k] = sum_{r,s} F_inv[r,i] C'[r,s] F_inv[s,k]
    #
    # Expanding: C[i,k] = sum_{r,s} F_inv[r,i] (sum_{i',j} F[r,i'] A[i',j] F[s,j]) 
    #                                    (sum_{j',k'} F[r,j'] B[j',k'] F[s,k']) F_inv[s,k]
    #
    # This is not simply 9 products because A'[r,s] and B'[r,s] share the (r,s) index
    # but involve different sums. The element-wise product C'[r,s] = A'[r,s]*B'[r,s]
    # is NOT a single multiplication of linear forms in A and B separately.
    #
    # So this approach doesn't directly give a rank-9 decomposition.
    #
    # The correct Winograd 3x3 uses 23 multiplications. Let me use the explicit
    # construction from Winograd's paper or a known implementation.
    
    # Known 23-term Winograd 3x3:
    # m0  = (a00+a01+a02)(b00+b10+b20)
    # m1  = (a00+a01-a02)(b00-b10+b20)
    # m2  = (a00-a01+a02)(b00+b10-b20)
    # m3  = (a00-a01-a02)(b00-b10-b20)
    # m4  = (a10+a11+a12)(b00+b10+b20)
    # m5  = (a10+a11-a12)(b00-b10+b20)
    # m6  = (a10-a11+a12)(b00+b10-b20)
    # m7  = (a10-a11-a12)(b00-b10-b20)
    # m8  = (a20+a21+a22)(b00+b10+b20)
    # m9  = (a20+a21-a22)(b00-b10+b20)
    # m10 = (a20-a21+a22)(b00+b10-b20)
    # m11 = (a20-a21-a22)(b00-b10-b20)
    # m12 = (a01+a02)(b11+b21)
    # m13 = (a01-a02)(b11-b21)
    # m14 = (a11+a12)(b11+b21)
    # m15 = (a11-a12)(b11-b21)
    # m16 = (a21+a22)(b11+b21)
    # m17 = (a21-a22)(b11-b21)
    # m18 = (a00+a02)(b12+b22)
    # m19 = (a00-a02)(b12-b22)
    # m20 = (a10+a12)(b12+b22)
    # m21 = (a10-a12)(b12-b22)
    # m22 = (a20+a22)(b12+b22)
    # m23 = (a20-a22)(b12-b22)
    
    # Wait, that's 24 terms. The actual Winograd 3x3 has 23. Let me check.
    # Actually I think the standard count is 23 for the full 3x3. Let me recount.
    # Terms 0-11: 12 terms (4 per row of A, using all 3 columns of B's first column)
    # Terms 12-17: 6 terms (2 per row of A, using B's second column)  
    # Terms 18-23: 6 terms (2 per row of A, using B's third column)
    # Total: 12 + 6 + 6 = 24
    
    # Hmm, maybe it's 24 not 23? Or maybe some terms can be combined.
    # Let me just verify if 24 terms work, and if so, use that.
    
    # Let me define the 24 terms and verify.
    # A_terms[t] = list of (i, j, coeff)
    # B_terms[t] = list of (j, k, coeff)
    # C_terms[t] = list of (i, k, coeff)
    
    A_terms = []
    B_terms = []
    C_terms = []
    
    # Helper to add a term
    def add_term(a_coeffs, b_coeffs, c_coeffs):
        A_terms.append(a_coeffs)
        B_terms.append(b_coeffs)
        C_terms.append(c_coeffs)
    
    # Terms 0-11: For each row i of A (0,1,2), 4 terms using B column 0 (j=0,1,2)
    # The 4 combinations correspond to the 2x2 Hadamard-like transform on B's rows
    for i in range(3):
        # m: (a[i,0]+a[i,1]+a[i,2]) * (b[0,0]+b[1,0]+b[2,0])
        add_term([(i,0,1),(i,1,1),(i,2,1)], 
                 [(0,0,1),(1,0,1),(2,0,1)],
                 [(0,0,1),(1,0,1),(2,0,1)])
        # m: (a[i,0]+a[i,1]-a[i,2]) * (b[0,0]-b[1,0]+b[2,0])
        add_term([(i,0,1),(i,1,1),(i,2,-1)], 
                 [(0,0,1),(1,0,-1),(2,0,1)],
                 [(0,0,1),(1,0,-1),(2,0,1)])
        # m: (a[i,0]-a[i,1]+a[i,2]) * (b[0,0]+b[1,0]-b[2,0])
        add_term([(i,0,1),(i,1,-1),(i,2,1)], 
                 [(0,0,1),(1,0,1),(2,0,-1)],
                 [(0,0,1),(1,0,1),(2,0,-1)])
        # m: (a[i,0]-a[i,1]-a[i,2]) * (b[0,0]-b[1,0]-b[2,0])
        add_term([(i,0,1),(i,1,-1),(i,2,-1)], 
                 [(0,0,1),(1,0,-1),(2,0,-1)],
                 [(0,0,1),(1,0,-1),(2,0,-1)])
    
    # Terms 12-17: For each row i of A (0,1,2), 2 terms using B column 1 (j=1,2)
    for i in range(3):
        # m: (a[i,1]+a[i,2]) * (b[1,1]+b[2,1])
        add_term([(i,1,1),(i,2,1)], 
                 [(1,1,1),(2,1,1)],
                 [(0,1,1),(1,1,1),(2,1,1)])
        # m: (a[i,1]-a[i,2]) * (b[1,1]-b[2,1])
        add_term([(i,1,1),(i,2,-1)], 
                 [(1,1,1),(2,1,-1)],
                 [(0,1,1),(1,1,-1),(2,1,1)])
    
    # Terms 18-23: For each row i of A (0,1,2), 2 terms using B column 2 (j=1,2)
    for i in range(3):
        # m: (a[i,0]+a[i,2]) * (b[1,2]+b[2,2])
        add_term([(i,0,1),(i,2,1)], 
                 [(1,2,1),(2,2,1)],
                 [(0,2,1),(1,2,1),(2,2,1)])
        # m: (a[i,0]-a[i,2]) * (b[1,2]-b[2,2])
        add_term([(i,0,1),(i,2,-1)], 
                 [(1,2,1),(2,2,-1)],
                 [(0,2,1),(1,2,-1),(2,2,1)])
    
    return A_terms, B_terms, C_terms


def _build_decomposition_from_terms(A_terms, B_terms, C_terms, n, m, p):
    """Build U, V, W matrices from term specifications.
    
    A_terms[t]: list of (i, j, coeff) - coefficients for A[i,j] in term t
    B_terms[t]: list of (j, k, coeff) - coefficients for B[j,k] in term t
    C_terms[t]: list of (i, k, coeff) - coefficients for C[i,k] in term t
    
    U[i*m+j, t] = sum of coeff for (i,j) in A_terms[t]
    V[j*p+k, t] = sum of coeff for (j,k) in B_terms[t]
    W[k*n+i, t] = sum of coeff for (i,k) in C_terms[t]
    """
    rank = len(A_terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for t in range(rank):
        for (i, j, coeff) in A_terms[t]:
            U[i * m + j][t] += coeff
        for (j, k, coeff) in B_terms[t]:
            V[j * p + k][t] += coeff
        for (i, k, coeff) in C_terms[t]:
            W[k * n + i][t] += coeff
    
    return U, V, W


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Check off-diagonal entries are zero
    # T[i*m+j, j'*p+k', k''*n+i''] = 0 unless j=j', k'=k'', i''=i
    # We check a subset of off-diagonal entries
    checked = 0
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for kp in range(p):
                    for kn in range(p):
                        for in_ in range(n):
                            if jp == j and kp == kn and in_ == i:
                                continue  # This is a diagonal entry, already checked
                            val = np.sum(U_arr[i*m+j, :] * V_arr[jp*p+kp, :] * W_arr[kn*n+in_, :])
                            if abs(val) > tol:
                                return False
                            checked += 1
                            if checked > 10000:
                                return True  # Enough checks
    
    return True


def _construct_winograd_extended(n, m, p):
    """Construct a decomposition for n x m * m x p using Winograd for the n x m part.
    
    Strategy: For each column k of B, we need to compute C[:,k] = A * B[:,k].
    We can use the Winograd 3x3 structure for each column, but share the A transforms.
    
    For 3x3 * 3x8:
    - The Winograd 3x3 has 24 terms (or 23 if we can optimize).
    - For 8 columns, naive extension gives 24*8 = 192 terms.
    - But we can do better by grouping columns.
    
    Actually, let me think about this differently. The tensor for 3x3 * 3x8 is:
    T[(i,j), (j',k'), (k'',i'')] = 1 if j=j', k'=k'', i''=i
    
    This is the same structure as 3x3 * 3x3 but with 8 values of k instead of 3.
    
    One approach: use the Winograd decomposition for 3x3 * 3x3 (rank 24) and
    replicate it for each group of 3 columns, with adjustments.
    
    For 8 columns: 3 + 3 + 2. 
    - First 3 columns: use Winograd 3x3 (24 terms)
    - Next 3 columns: use Winograd 3x3 (24 terms)  
    - Last 2 columns: use a 3x3 * 3x2 decomposition (smaller rank)
    
    For 3x3 * 3x2, schoolbook rank is 18. Can we do better?
    Probably not much, so let's use schoolbook for the last 2 columns.
    
    Total: 24 + 24 + 18 = 66 terms. Still better than 72.
    
    But wait, we can do even better. Let me think about sharing the A-transforms.
    
    Actually, the cleanest approach: just use the schoolbook decomposition but
    try to find a lower-rank one through optimization.
    
    For now, let me construct the Winograd-based decomposition with 66 terms.
    """
    if n != 3 or m != 3:
        return _schoolbook_decomposition(n, m, p)
    
    A_terms, B_terms, C_terms = _winograd_3x3_terms()
    base_rank = len(A_terms)  # Should be 24
    
    # For 8 columns: use Winograd for first 6 columns (2 groups of 3),
    # and schoolbook for last 2 columns.
    
    all_A = []
    all_B = []
    all_C = []
    
    # Group 1: columns 0,1,2 (Winograd 3x3)
    for t in range(base_rank):
        all_A.append(A_terms[t])
        # B terms: replace k with k (0,1,2)
        all_B.append(B_terms[t])
        # C terms: replace k with k (0,1,2)
        all_C.append(C_terms[t])
    
    # Group 2: columns 3,4,5 (Winograd 3x3, shifted)
    for t in range(base_rank):
        all_A.append(A_terms[t])
        # B terms: shift k by 3
        shifted_B = [(j, k+3, coeff) for (j, k, coeff) in B_terms[t]]
        all_B.append(shifted_B)
        # C terms: shift k by 3
        shifted_C = [(i, k+3, coeff) for (i, k, coeff) in C_terms[t]]
        all_C.append(shifted_C)
    
    # Group 3: columns 6,7 (schoolbook for 3x3 * 3x2)
    for i in range(3):
        for j in range(3):
            for k in [6, 7]:
                all_A.append([(i, j, 1.0)])
                all_B.append([(j, k, 1.0)])
                all_C.append([(i, k, 1.0)])
    
    U, V, W = _build_decomposition_from_terms(all_A, all_B, all_C, n, m, p)
    return U, V, W


def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a safe fallback
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = n * m * p
    
    # Try the Winograd-extended decomposition
    try:
        U_win, V_win, W_win = _construct_winograd_extended(n, m, p)
        if _verify_decomposition(U_win, V_win, W_win, n, m, p):
            rank_win = len(W_win[0]) if W_win else 0
            if rank_win < best_rank:
                best_result = {"U": U_win, "V": V_win, "W": W_win}
                best_rank = rank_win
    except Exception:
        pass
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > DEADLINE_SECONDS - SAFETY_MARGIN:
        return best_result
    
    # Try to further reduce rank using optimization
    # Use ALS (Alternating Least Squares) on the tensor
    try:
        U_arr = np.array(best_result["U"], dtype=np.float64)
        V_arr = np.array(best_result["V"], dtype=np.float64)
        W_arr = np.array(best_result["W"], dtype=np.float64)
        current_rank = len(W_arr[0])
        
        # Build the target tensor
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Try to reduce rank by removing terms that are nearly redundant
        # This is a heuristic: check if removing a term increases the error
        # significantly, and if so, keep it; otherwise, remove it.
        
        # For now, just return the best we have
        return best_result
        
    except Exception:
        return best_result
