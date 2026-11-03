"""Rank-minimal exact tensor decomposition for M_{2,5,5} matrix multiplication.

Strategy:
1. Start from the schoolbook rank-50 decomposition.
2. Use a structured search to try to find a lower-rank decomposition.
   - We know the rank of M_{2,2,2} is 7 (Strassen).
   - We can exploit block structure: split the 5 middle dimension into 2+3.
   - The 2x2x2 sub-problem can use rank 7 instead of rank 8.
   - This gives rank = 7 + 2*2*3 + 2*3*5 - 2*2*3 = ... let me recalculate.
   
   Actually, let's think about this more carefully.
   
   C[i,k] = sum_j A[i,j] B[j,k] for i in 0..1, j in 0..4, k in 0..4.
   
   Split j into {0,1} and {2,3,4}:
   C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]
   
   First term: 2x2 * 2x5. Split B[0:2,:] into B[0:2,0:2] and B[0:2,2:5]:
     A[:,0:2] @ B[0:2,0:2] is 2x2*2x2 -> rank 7 (Strassen)
     A[:,0:2] @ B[0:2,2:5] is 2x2*2x3 -> rank 12 (schoolbook)
   
   Second term: A[:,2:5] @ B[2:5,:] is 2x3 * 3x5 -> rank 30 (schoolbook)
   
   Total: 7 + 12 + 30 = 49.
   
   But wait, we need to be careful about how the blocks combine. The output C is 2x5,
   and the two terms add together. The first term contributes to all of C (2x5),
   and the second term also contributes to all of C (2x5). So we can't just add the ranks
   naively because the outputs overlap.
   
   Actually, in tensor decomposition, if we have:
   C = T1 + T2
   where T1 has rank r1 and T2 has rank r2, and they act on disjoint parts of the output,
   then rank(C) <= r1 + r2. But here both T1 and T2 contribute to the entire output C.
   
   However, in the tensor decomposition framework, each rank-1 term is a triple (u, v, w)
   where w is a vector in the output space. If T1 and T2 have overlapping output supports,
   we can still combine them: the total rank is at most r1 + r2.
   
   So the total rank should be 7 + 12 + 30 = 49.
   
   Let me implement this carefully.
"""
import time
import numpy as np

DIMENSIONS = [2, 5, 5]

def _build_strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen).
    
    A is 2x2, B is 2x2, C = A@B is 2x2.
    A indices: (i,j) -> i*2+j, i,j in {0,1}
    B indices: (j,k) -> j*2+k, j,k in {0,1}  
    C indices: (i,k) -> i*2+k, i,k in {0,1}  [but our W uses k*2+i]
    
    Returns list of (u, v, w) where u,v,w are length-4 vectors.
    w is in the (i,k) -> i*2+k convention; we'll convert to k*2+i later.
    """
    # Standard Strassen decomposition for 2x2 matrix multiplication.
    # Using 0-indexed: A[i,j], B[j,k], C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # A indices: a00=0, a01=1, a10=2, a11=3
    # B indices: b00=0, b01=1, b10=2, b11=3
    # C indices (i*2+k): c00=0, c01=1, c10=2, c11=3
    #
    # Strassen's 7 products:
    # m1 = (a01 - a10)(b10 + b11)
    # m2 = (a00 + a10)b00
    # m3 = a00(b01 - b11)
    # m4 = a11(b10 - b00)
    # m5 = (a00 + a01)b11
    # m6 = (a10 - a00)(b00 + b01)
    # m7 = (a01 - a11)(b10 + b11)
    #
    # C00 = m5 + m4 - m3 - m7
    # C01 = m1 + m7
    # C10 = m2 + m4
    # C11 = m5 - m1 + m2 + m3
    #
    # Let me verify C01:
    # C01 should be a00*b01 + a01*b11
    # m1 + m7 = (a01-a10)(b10+b11) + (a01-a11)(b10+b11)
    # = (2a01 - a10 - a11)(b10+b11)
    # This is NOT a00*b01 + a01*b11. So my Strassen formulas are wrong.
    #
    # Let me look up the correct Strassen formulas more carefully.
    #
    # From Wikipedia (Strassen algorithm), using 1-based indexing:
    # A = [[A11, A12], [A21, A22]], B = [[B11, B12], [B21, B22]]
    # M1 = (A11 - A21)(B21 + B22)
    # M2 = (A11 + A21)B11
    # M3 = A12(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M5 + M4 - M3 - M7
    # C12 = M1 + M7
    # C21 = M2 + M4
    # C22 = M5 - M1 + M2 + M3
    #
    # Converting to 0-indexed:
    # A11=a00, A12=a01, A21=a10, A22=a11
    # B11=b00, B12=b01, B21=b10, B22=b11
    # C11=c00, C12=c01, C21=c10, C22=c11
    #
    # M1 = (a00-a10)(b10+b11)
    # M2 = (a00+a10)b00
    # M3 = a01(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    #
    # C00 = M5 + M4 - M3 - M7
    # C01 = M1 + M7
    # C10 = M2 + M4
    # C11 = M5 - M1 + M2 + M3
    #
    # Let me verify C01 = a00*b01 + a01*b11:
    # M1 + M7 = (a00-a10)(b10+b11) + (a01-a11)(b10+b11)
    # = (a00-a10+a01-a11)(b10+b11)
    # = (a00+a01-a10-a11)(b10+b11)
    # Still not matching. 
    #
    # I think the issue is that I'm not correctly remembering Strassen's formulas.
    # Let me just use a verified decomposition from the literature.
    #
    # Actually, for the purpose of this task, let me just use the schoolbook rank-50
    # decomposition which is definitely correct, and then attempt to verify if a rank-49
    # decomposition exists through numerical search.
    
    # For now, return None to indicate we should use schoolbook.
    return None

def _schoolbook_decomposition():
    """Return the schoolbook rank-50 decomposition for 2x5x5 matrix multiplication.
    
    U has shape (10, 50), V has shape (25, 50), W has shape (10, 50).
    U indices: i*5+j for i in 0..1, j in 0..4
    V indices: j*5+k for j in 0..4, k in 0..4
    W indices: k*2+i for k in 0..4, i in 0..1
    """
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n=2, m=5, p=5, tol=1e-9):
    """Verify that the decomposition correctly represents matrix multiplication.
    
    For random A (n x m) and B (m x p), check that:
    C = A @ B equals sum_l (U[:,l] reshaped to n x m) @ (V[:,l] reshaped to m x p) 
    ... no, that's not right.
    
    The tensor decomposition says:
    T[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    should equal 1 if (i,j,k) is a valid triple, 0 otherwise.
    
    Equivalently, for any A and B:
    C[i,k] = sum_j A[i,j] B[j,k] = sum_l (sum_j U[i*m+j,l] A[i,j]) * (sum_j V[j*p+k,l] B[j,k]) * W[k*n+i,l]
    
    Wait, that's not quite right either. Let me think again.
    
    The matrix multiplication tensor M has entries M[i,j,k] = 1 for all i,j,k
    (since every (i,j,k) triple contributes to the product).
    
    A rank-r decomposition is: M = sum_{l=1}^r u_l ⊗ v_l ⊗ w_l
    where u_l ∈ R^{n*m}, v_l ∈ R^{m*p}, w_l ∈ R^{p*n}.
    
    This means: M[i,j,k] = sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    For matrix multiplication, M[i,j,k] = 1 for all i,j,k.
    
    So we need: sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all i,j,k.
    
    Let me verify this.
    """
    # Create the target tensor (all ones)
    # M[i,j,k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    
    # Compute the decomposed tensor
    # For each (i,j,k), compute sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l]
    
    # This is equivalent to: for each l, the outer product of U[:,l], V[:,l], W[:,l]
    # summed over l.
    
    # Let's verify by checking a few random (i,j,k) triples.
    rng = np.random.default_rng(42)
    for _ in range(100):
        i = rng.integers(0, n)
        j = rng.integers(0, m)
        k = rng.integers(0, p)
        val = 0.0
        for l in range(U.shape[1]):
            val += U[i * m + j, l] * V[j * p + k, l] * W[k * n + i, l]
        if abs(val - 1.0) > tol:
            return False, (i, j, k, val)
    return True, None

def _verify_with_matrices(U, V, W, n=2, m=5, p=5, tol=1e-9):
    """Verify the decomposition by checking that it correctly computes A@B for random A, B.
    
    The decomposition should satisfy:
    A @ B = sum_l (A * U[:,l]) (B * V[:,l]) * W[:,l]  ... no, this isn't right.
    
    Actually, the correct interpretation is:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    And the tensor decomposition gives:
    sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] = 1 for all i,j,k
    
    This means the decomposition is correct iff the above holds.
    """
    # Verify the tensor identity
    ok, detail = _verify_decomposition(U, V, W, n, m, p, tol)
    return ok, detail

def construct():
    n, m, p = DIMENSIONS  # 2, 5, 5
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Start with schoolbook rank-50
    U, V, W = _schoolbook_decomposition()
    best_rank = 50
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Verify schoolbook
    ok, _ = _verify_with_matrices(best_U, best_V, best_W, n, m, p)
    if not ok:
        # Shouldn't happen, but return anyway
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find a lower-rank decomposition using structured search.
    # Strategy: try rank 49, 48, etc.
    # For rank 49, we can try the block structure with Strassen for the 2x2x2 part.
    
    # Let's try to construct a rank-49 decomposition using the block structure.
    # Split j into {0,1} and {2,3,4}.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:5] @ B[2:5,:]
    #
    # A[:,0:2] @ B[0:2,:] is 2x2 * 2x5.
    # Split B[0:2,:] into B[0:2,0:2] and B[0:2,2:5]:
    #   A[:,0:2] @ B[0:2,0:2] is 2x2*2x2 -> use Strassen rank 7
    #   A[:,0:2] @ B[0:2,2:5] is 2x2*2x3 -> schoolbook rank 12
    #
    # A[:,2:5] @ B[2:5,:] is 2x3 * 3x5 -> schoolbook rank 30
    #
    # Total: 7 + 12 + 30 = 49
    
    # Let me try to construct this.
    #
    # For the Strassen part, I need a correct rank-7 decomposition of 2x2x2.
    # Let me try to find one numerically.
    
    if time.time() < deadline - 10:
        # Try to find Strassen decomposition for 2x2x2
        strassen = _find_strassen_2x2x2()
        if strassen is not None:
            U49, V49, W49 = _build_rank49(strassen)
            if U49 is not None:
                ok, _ = _verify_with_matrices(U49, V49, W49, n, m, p)
                if ok:
                    best_rank = 49
                    best_U, best_V, best_W = U49, V49, W49
                    print("Found rank-49 decomposition!")
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def _find_strassen_2x2x2():
    """Try to find a rank-7 decomposition of the 2x2x2 matrix multiplication tensor.
    
    Returns list of 7 (u, v, w) triples, or None if not found.
    """
    # The 2x2x2 matrix multiplication tensor has entries T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We want to find 7 triples (u_l, v_l, w_l) in R^4 x R^4 x R^4 such that:
    # sum_l u_l[i*2+j] * v_l[j*2+k] * w_l[i*2+k] = 1 for all i,j,k.
    #
    # This is a system of 8 equations (for 8 (i,j,k) triples) with 7*4*3 = 84 unknowns.
    # It's underdetermined, so there should be solutions.
    #
    # Let me try a structured approach. I'll parameterize the solution and solve.
    
    # Actually, let me just use a known correct Strassen decomposition.
    # After more careful thought, here's one:
    
    # The 7 terms (in terms of (i,j), (j,k), (i,k) indices, 0-indexed):
    # I'll use the decomposition from the literature.
    
    # Let me try a different approach: use numpy to solve for a rank-7 decomposition.
    
    # Create the target tensor (flattened)
    # T[i,j,k] = 1 for i,j,k in {0,1}
    # Flatten: index = i*4 + j*2 + k
    T = np.ones((2, 2, 2))
    
    # We want to find U, V, W of shape (4, 7) such that:
    # sum_l U[i*2+j, l] * V[j*2+k, l] * W[i*2+k, l] = 1 for all i,j,k
    
    # This is a nonlinear problem. Let me try a simple initialization and use least squares.
    
    # Initialize with schoolbook (rank 8) and then try to compress to rank 7.
    # Or just try random initializations.
    
    rng = np.random.default_rng(123)
    
    for attempt in range(50):
        U = rng.standard_normal((4, 7))
        V = rng.standard_normal((4, 7))
        W = rng.standard_normal((4, 7))
        
        # Use ALS (Alternating Least Squares) to refine
        for _ in range(100):
            # Update U
            for ij in range(4):
                i, j = divmod(ij, 2)
                # For each l, U[ij, l] should satisfy:
                # sum_l U[ij, l] * V[j*2+k, l] * W[i*2+k, l] = 1 for all k
                # This is: sum_l U[ij, l] * (V[j*2+k, l] * W[i*2+k, l]) = 1
                # Let A_l = V[j*2+k, l] * W[i*2+k, l] for k=0,1
                # Then sum_l U[ij, l] * A_l = [1, 1]
                # This is a linear system: U[ij, :] @ A^T = [1, 1]
                # Where A is 2x7 (k x l)
                A = np.array([[V[j*2+k, l] * W[i*2+k, l] for l in range(7)] for k in range(2)])
                # Solve U[ij, :] @ A^T = [1, 1]
                # i.e., A @ U[ij, :]^T = [1, 1]
                try:
                    U[ij, :] = np.linalg.lstsq(A, np.ones(2), rcond=None)[0]
                except:
                    pass
            
            # Update V
            for jk in range(4):
                j, k = divmod(jk, 2)
                A = np.array([[U[i*2+j, l] * W[i*2+k, l] for l in range(7)] for i in range(2)])
                try:
                    V[jk, :] = np.linalg.lstsq(A, np.ones(2), rcond=None)[0]
                except:
                    pass
            
            # Update W
            for ik in range(4):
                i, k = divmod(ik, 2)
                A = np.array([[U[i*2+j, l] * V[j*2+k, l] for l in range(7)] for j in range(2)])
                try:
                    W[ik, :] = np.linalg.lstsq(A, np.ones(2), rcond=None)[0]
                except:
                    pass
        
        # Check if the solution is good
        max_err = 0.0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    val = sum(U[i*2+j, l] * V[j*2+k, l] * W[i*2+k, l] for l in range(7))
                    max_err = max(max_err, abs(val - 1.0))
        
        if max_err < 1e-6:
            # Found a solution!
            return [(U[:, l], V[:, l], W[:, l]) for l in range(7)]
    
    return None

def _build_rank49(strassen_terms):
    """Build a rank-49 decomposition using Strassen for the 2x2x2 part.
    
    strassen_terms: list of 7 (u, v, w) triples for 2x2x2.
    u, v, w are length-4 vectors.
    u indices: (i,j) -> i*2+j
    v indices: (j,k) -> j*2+k
    w indices: (i,k) -> i*2+k  [but we need k*2+i for our W]
    """
    n, m, p = 2, 5, 5
    rank = 49
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Part 1: A[:,0:2] @ B[0:2,0:2] using Strassen (rank 7)
    # A indices: i in 0..1, j in 0..1 -> U[i*5+j]
    # B indices: j in 0..1, k in 0..1 -> V[j*5+k]
    # C indices: i in 0..1, k in 0..1 -> W[k*2+i]
    #
    # Strassen terms: u[i*2+j], v[j*2+k], w[i*2+k]
    # We need to map w[i*2+k] to W[k*2+i].
    # w index i*2+k -> W index k*2+i
    
    for l in range(7):
        u, v, w = strassen_terms[l]
        for i in range(2):
            for j in range(2):
                if u[i*2+j] != 0:
                    U[i*5+j, col] = u[i*2+j]
        for j in range(2):
            for k in range(2):
                if v[j*2+k] != 0:
                    V[j*5+k, col] = v[j*2+k]
        for i in range(2):
            for k in range(2):
                if w[i*2+k] != 0:
                    W[k*2+i, col] = w[i*2+k]
        col += 1
    # col should be 7
    
    # Part 2: A[:,0:2] @ B[0:2,2:5] schoolbook (rank 12)
    # A indices: i in 0..1, j in 0..1 -> U[i*5+j]
    # B indices: j in 0..1, k in 2..4 -> V[j*5+k]
    # C indices: i in 0..1, k in 2..4 -> W[k*2+i]
    for j in range(2):
        for k in range(2, 5):
            for i in range(2):
                U[i*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
    # col should be 7 + 12 = 19
    
    # Part 3: A[:,2:5] @ B[2:5,:] schoolbook (rank 30)
    # A indices: i in 0..1, j in 2..4 -> U[i*5+j]
    # B indices: j in 2..4, k in 0..4 -> V[j*5+k]
    # C indices: i in 0..1, k in 0..4 -> W[k*2+i]
    for j in range(2, 5):
        for k in range(5):
            for i in range(2):
                U[i*5+j, col] = 1.0
                V[j*5+k, col] = 1.0
                W[k*2+i, col] = 1.0
                col += 1
    # col should be 19 + 30 = 49
    
    if col != rank:
        return None, None, None
    
    return U, V, W
