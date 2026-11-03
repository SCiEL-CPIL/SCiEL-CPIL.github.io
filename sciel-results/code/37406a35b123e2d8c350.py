"""Strassen-based rank-84 decomposition for 3x5x6 matrix multiplication."""
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START

def _verify(U, V, W, n, m, p):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == delta(i,j,k)."""
    rank = len(U[0])
    # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    import numpy as np
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    # T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # Reshape U to (n,m,rank), V to (m,p,rank), W to (p,n,rank)
    U3 = Ua.reshape(n, m, rank)
    V3 = Va.reshape(m, p, rank)
    W3 = Wa.reshape(p, n, rank)
    # T[i,j,k] = sum_c U3[i,j,c] * V3[j,k,c] * W3[k,i,c]
    T = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k] = np.dot(U3[i, j, :], V3[j, k, :]) * 0 + np.einsum('c,c,c->', U3[i, j, :], V3[j, k, :], W3[k, i, :])
    # Expected: T[i,j,k] = 1 if i==j==k else 0? No.
    # The tensor for matrix multiplication (A in R^{n x m}, B in R^{m x p}, C = A B in R^{n x p})
    # C[i,k] = sum_j A[i,j] B[j,k]
    # So the tensor T[i,j,k] should be 1 for all i,j,k (it's the identity in the middle index).
    # Actually the matrix multiplication tensor has T[i,j,k] = 1 if and only if the (i,j) of A and (j,k) of B
    # contribute to (i,k) of C. The standard convention: T[i,j,k] = 1 for all valid i,j,k.
    # Wait, let me re-read the task. The tensor decomposition represents the bilinear map.
    # C[i,k] = sum_{j,c} A[i,j] B[j,k] * (coefficient)
    # The coefficient tensor is T[i,j,k] = 1 for all i, j, k.
    expected = np.ones((n, m, p))
    return np.allclose(T, expected, atol=1e-9)

def _strassen_2x2_block():
    """
    Return the 7 Strassen rank-1 terms for 2x2x2 matrix multiplication.
    Each term is (u, v, w) where u, v, w are 4-dim vectors (flattened 2x2 matrices).
    The tensor T[i,j,k] = 1 for i,j,k in {0,1}.
    Flattening: index = i*2 + j for U/V, index = i*2 + j for W (but W uses k*n+i convention).
    
    For 2x2: n=m=p=2.
    U has shape (4, 7): rows indexed by i*2+j, cols by c.
    V has shape (4, 7): rows indexed by j*2+k, cols by c.
    W has shape (4, 7): rows indexed by k*2+i, cols by c.
    
    Strassen:
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    # Each term c has U_c (4-dim), V_c (4-dim), W_c (4-dim)
    # U_c[i*2+j] = coefficient of A[i,j] in M_c
    # V_c[j*2+k] = coefficient of B[j,k] in M_c
    # W_c[k*2+i] = coefficient for C[i,k] (i.e., C_{i,k} gets W_c[k*2+i] * M_c)
    
    terms = []
    
    # M1 = (A11+A22)(B11+B22)
    # A part: A[0,0]=1, A[1,1]=1 -> U[0]=1, U[3]=1
    # B part: B[0,0]=1, B[1,1]=1 -> V[0]=1, V[3]=1
    # C part: C[0,0]+=1, C[1,1]+=1 -> W[0]=1 (k=0,i=0), W[3]=1 (k=1,i=1)
    terms.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))
    
    # M2 = (A21+A22)B11
    # A part: A[1,0]=1, A[1,1]=1 -> U[2]=1, U[3]=1
    # B part: B[0,0]=1 -> V[0]=1
    # C part: C[1,0]+=1, C[1,1]-=1 -> W[0]=-1 (k=0,i=1), W[2]=1 (k=1,i=1)? 
    # Wait: C11 = M1 + M4 - M5 + M7, C12 = M3 + M5, C21 = M2 + M4, C22 = M1 - M2 + M3 + M6
    # So M2 contributes +1 to C21 and -1 to C22.
    # C21: i=1, k=0 -> W index = k*2+i = 0*2+1 = 1
    # C22: i=1, k=1 -> W index = k*2+i = 1*2+1 = 3
    # So W[1]=1, W[3]=-1
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,-1]))
    
    # M3 = A11(B21-B22)
    # A part: A[0,0]=1 -> U[0]=1
    # B part: B[1,0]=1, B[1,1]=-1 -> V[2]=1, V[3]=-1
    # C part: M3 contributes +1 to C12 and +1 to C22.
    # C12: i=0, k=1 -> W index = 1*2+0 = 2
    # C22: i=1, k=1 -> W index = 1*2+1 = 3
    # So W[2]=1, W[3]=1
    terms.append(([1,0,0,0], [0,0,1,-1], [0,0,1,1]))
    
    # M4 = A22(B12-B11)
    # A part: A[1,1]=1 -> U[3]=1
    # B part: B[0,1]=1, B[0,0]=-1 -> V[1]=1, V[0]=-1
    # C part: M4 contributes +1 to C11 and +1 to C21.
    # C11: i=0, k=0 -> W index = 0*2+0 = 0
    # C21: i=1, k=0 -> W index = 0*2+1 = 1
    # So W[0]=1, W[1]=1
    terms.append(([0,0,0,1], [-1,1,0,0], [1,1,0,0]))
    
    # M5 = (A11+A12)B22
    # A part: A[0,0]=1, A[0,1]=1 -> U[0]=1, U[1]=1
    # B part: B[1,1]=1 -> V[3]=1
    # C part: M5 contributes -1 to C11 and +1 to C12.
    # C11: i=0, k=0 -> W[0]=-1
    # C12: i=0, k=1 -> W[2]=1
    # So W[0]=-1, W[2]=1
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    
    # M6 = (A21-A11)(B11+B12)
    # A part: A[1,0]=1, A[0,0]=-1 -> U[2]=1, U[0]=-1
    # B part: B[0,0]=1, B[0,1]=1 -> V[0]=1, V[1]=1
    # C part: M6 contributes +1 to C22.
    # C22: i=1, k=1 -> W[3]=1
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    
    # M7 = (A12-A22)(B21+B22)
    # A part: A[0,1]=1, A[1,1]=-1 -> U[1]=1, U[3]=-1
    # B part: B[1,0]=1, B[1,1]=1 -> V[2]=1, V[3]=1
    # C part: M7 contributes +1 to C11.
    # C11: i=0, k=0 -> W[0]=1
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,0]))
    
    return terms

def _build_strassen_3x5x6():
    """
    Build a rank-84 decomposition for 3x5x6 matrix multiplication.
    
    n=3, m=5, p=6.
    
    Strategy: Partition the 5 columns of A into blocks and the 6 columns of B into blocks.
    Use 2x2 Strassen blocks where possible.
    
    Let's partition:
    A is 3x5, B is 5x6, C is 3x6.
    
    We can think of this as computing C = A * B where A has 5 columns and B has 5 rows.
    
    One approach: Pad to 4x8 or use a different decomposition.
    
    Actually, let me use a simpler approach: decompose into rank-1 terms directly.
    The schoolbook is rank 90. We want rank 84.
    
    Let me try: use Strassen on a 2x2x2 subproblem embedded in the 3x5x6.
    
    Alternative: Use the fact that 3x5x6 can be decomposed by considering
    the 5 shared dimension. Split the 5 into 2+2+1.
    
    For the 2+2 part (columns 0,1,2,3 of A and rows 0,1,2,3 of B):
    This is a 3x4x4 subproblem. We can use Strassen-like methods.
    
    Actually, let me just construct a valid rank-84 decomposition by modifying the schoolbook.
    
    Schoolbook: rank 90, each term (i,j,k) has U[i*5+j]=1, V[j*6+k]=1, W[k*3+i]=1.
    
    To reduce rank, we need to combine terms. Strassen combines 8 terms into 7.
    
    Let me identify a 2x2x2 substructure in the 3x5x6 tensor.
    Take i in {0,1}, j in {0,1}, k in {0,1}. This gives 8 schoolbook terms.
    We can replace them with 7 Strassen terms.
    
    The 8 schoolbook terms for (i,j,k) with i,j,k in {0,1}:
    (0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)
    
    For each, U[i*5+j], V[j*6+k], W[k*3+i].
    
    i=0,j=0: U[0], V[0], W[0]
    i=0,j=0: U[0], V[1], W[3]
    i=0,j=1: U[1], V[6], W[0]
    i=0,j=1: U[1], V[7], W[3]
    i=1,j=0: U[5], V[0], W[1]
    i=1,j=0: U[5], V[1], W[4]
    i=1,j=1: U[6], V[6], W[1]
    i=1,j=1: U[6], V[7], W[4]
    
    Now I need to express these 8 rank-1 tensors as a sum of 7 rank-1 tensors.
    This is the Strassen decomposition applied to this subcube.
    
    Let me map the 2x2x2 Strassen terms to the 3x5x6 indices.
    In the 2x2x2 case, the indices are:
    U2[i*2+j], V2[j*2+k], W2[k*2+i]
    
    Mapping to 3x5x6:
    i in {0,1} -> same
    j in {0,1} -> j in {0,1} (using first 2 of the 5)
    k in {0,1} -> k in {0,1} (using first 2 of the 6)
    
    U2[i*2+j] -> U[i*5+j]
    V2[j*2+k] -> V[j*6+k]
    W2[k*2+i] -> W[k*3+i]
    
    So for each Strassen term (u2, v2, w2) where u2 is 4-dim (indexed i*2+j),
    u is 15-dim with u[i*5+j] = u2[i*2+j] for i,j in {0,1}, else 0.
    Similarly for v (30-dim) and w (18-dim).
    
    This gives 7 terms that exactly decompose the 2x2x2 subcube.
    The remaining 82 schoolbook terms (90-8=82) stay as-is.
    Total rank = 7 + 82 = 89. Not 84.
    
    To get rank 84, I need to find more 2x2x2 subcubes to apply Strassen to.
    Each application saves 1 rank. I need 6 more savings.
    
    How many disjoint 2x2x2 subcubes can I fit in 3x5x6?
    i: 3 values, j: 5 values, k: 6 values.
    
    I can partition j into {0,1},{2,3},{4} and k into {0,1},{2,3},{4,5}.
    For each (i-pair, j-pair, k-pair) I can apply Strassen.
    i-pairs: {0,1} (only one since n=3)
    j-pairs: {0,1}, {2,3} (two pairs, j=4 left over)
    k-pairs: {0,1}, {2,3}, {4,5} (three pairs)
    
    So I can apply Strassen to 1 * 2 * 3 = 6 subcubes, saving 6 ranks.
    Total rank = 90 - 6 = 84.
    
    The remaining terms involve j=4 or i=2.
    Terms with i=2: 5*6=30 terms (j in 0..4, k in 0..5)
    Terms with i in {0,1}, j=4: 2*6=12 terms
    Total remaining: 30 + 12 = 42 terms.
    Strassen terms: 6 * 7 = 42 terms.
    Total: 42 + 42 = 84. 
    """
    n, m, p = 3, 5, 6
    rank = 84
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    strassen_terms = _strassen_2x2_block()
    
    col = 0
    # Apply Strassen to each (i_pair, j_pair, k_pair)
    i_pairs = [(0, 1)]
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for (i0, i1) in i_pairs:
        for (j0, j1) in j_pairs:
            for (k0, k1) in k_pairs:
                for (u2, v2, w2) in strassen_terms:
                    # u2 is 4-dim: u2[i*2+j] for i in {0,1}, j in {0,1}
                    # Map to U[(i0+i)*m + (j0+j)]
                    for di in range(2):
                        for dj in range(2):
                            idx = di * 2 + dj
                            if u2[idx] != 0:
                                U[(i0 + di) * m + (j0 + dj)][col] += u2[idx]
                    # v2 is 4-dim: v2[j*2+k] for j in {0,1}, k in {0,1}
                    # Map to V[(j0+j)*p + (k0+k)]
                    for dj in range(2):
                        for dk in range(2):
                            idx = dj * 2 + dk
                            if v2[idx] != 0:
                                V[(j0 + dj) * p + (k0 + dk)][col] += v2[idx]
                    # w2 is 4-dim: w2[k*2+i] for k in {0,1}, i in {0,1}
                    # Map to W[(k0+k)*n + (i0+i)]
                    for dk in range(2):
                        for di in range(2):
                            idx = dk * 2 + di
                            if w2[idx] != 0:
                                W[(k0 + dk) * n + (i0 + di)][col] += w2[idx]
                    col += 1
    
    # Remaining terms: schoolbook terms not covered by Strassen
    # Covered: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    # Remaining: i=2 (any j,k) OR i in {0,1} and j=4 (any k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if this term is covered by Strassen
                if i in (0, 1) and j in (0, 1, 2, 3):
                    continue
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    # Convert to integer/half-integer (all entries are already integers)
    U = [[int(x) if x == int(x) else x for x in row] for row in U]
    V = [[int(x) if x == int(x) else x for x in row] for row in V]
    W = [[int(x) if x == int(x) else x for x in row] for row in W]
    
    return {"U": U, "V": V, "W": W}

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try Strassen-based rank-84 decomposition
    try:
        result = _build_strassen_3x5x6()
        # Verify
        if _verify(result["U"], result["V"], result["W"], 3, 5, 6):
            if time.time() < deadline:
                return result
    except Exception:
        pass
    
    # Fallback to schoolbook rank-90
    return _schoolbook()

# EVOLVE-BLOCK-END
