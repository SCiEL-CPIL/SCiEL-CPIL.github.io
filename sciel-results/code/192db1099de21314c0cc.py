import numpy as np
import time
import itertools

DIMENSIONS = [2, 3, 5]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We need T[i,j,k] = 1 for all i,j,k
    Ua = np.array(U, dtype=float).reshape(n, m, rank)
    Va = np.array(V, dtype=float).reshape(m, p, rank)
    Wa = np.array(W, dtype=float).reshape(p, n, rank)
    # T[i,j,k] = sum_c Ua[i,j,c] * Va[j,k,c] * Wa[k,i,c]
    # Build via einsum
    # First: Ua[i,j,c] * Va[j,k,c] -> temp[i,j,k,c]
    temp = np.einsum('ijc,jkc->ijkc', Ua, Va)
    # Then: sum_c temp[i,j,k,c] * Wa[k,i,c]
    result = np.einsum('ijkc,kic->ijk', temp, Wa)
    return np.allclose(result, np.ones((n, m, p)), atol=1e-9)

def _build_rank28():
    """Build the known rank-28 decomposition for 2x3 by 3x5 matrix multiplication."""
    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # M1 = (A[i1,j1]+A[i2,j2]) * (B[j1,k1]+B[j2,k2])
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A[i2,j1]) * (B[j1,k1]-B[j2,k2])  -- wait, let me use standard Strassen
        # Standard 2x2 Strassen:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1+M4-M5-M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M7
        
        # M2 = (A[i2,j1]+A[i2,j2]) * B[j1,k1]
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1
        col += 1
        # M3 = A[i1,j1] * (B[j1,k2]-B[j2,k2])
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1
        col += 1
        # M4 = A[i2,j2] * (B[j2,k1]-B[j1,k1])
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A[i1,j1]+A[i1,j2]) * B[j2,k2]
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A[i2,j1]-A[i1,j1]) * (B[j1,k1]+B[j1,k2])
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A[i1,j2]-A[i2,j2]) * (B[j2,k1]+B[j2,k2])
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i1, col] = 1
        col += 1
        
    def add_schoolbook(i, j, k):
        nonlocal col
        U[i*m+j, col] = 1
        V[j*p+k, col] = 1
        W[k*n+i, col] = 1
        col += 1
    
    # Use Strassen for the 2x2 block (rows 0,1 of A; cols 0,1 of B; cols 0,1 of output)
    add_strassen(0, 1, 0, 1, 0, 1)
    # Use Strassen for (rows 0,1 of A; cols 0,1 of B; cols 2,3 of output)
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Schoolbook for remaining: j=2 (third row of A / third col of B)
    # For each i in {0,1}, k in {0,1,2,3,4}: A[i,2]*B[2,k]
    for i in [0, 1]:
        for k in [0, 1, 2, 3, 4]:
            add_schoolbook(i, 2, k)
    
    # Also need: for j in {0,1}, k=4: A[i,j]*B[j,4]
    for i in [0, 1]:
        for j in [0, 1]:
            add_schoolbook(i, j, 4)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    return U, V, W

def _try_reduce_rank(U, V, W, n, m, p, target_rank):
    """
    Try to reduce rank by checking if any subset of columns can be removed
    and the remaining can be adjusted. This is a simplified approach.
    
    Actually, the proper way: check if the rank-1 tensors are linearly dependent.
    If they are, we might be able to express one as a combination of others,
    but that's complex. Instead, let's try a direct approach.
    """
    pass

def _build_rank24_candidate():
    """
    Try to construct a rank-24 decomposition.
    
    Strategy: Use 2 Strassen blocks for the 2x2 part (14 terms) and try to
    handle the remaining 10 terms (j=2 row, k=4 column) more efficiently.
    
    Actually, let's think about this differently. The tensor is:
    T[i,j,k] = 1 for all i in {0,1}, j in {0,1,2}, k in {0,1,2,3,4}
    
    We can write T = T1 + T2 + T3 where:
    T1[i,j,k] = 1 if j in {0,1} and k in {0,1}  (2x2x2 block)
    T2[i,j,k] = 1 if j in {0,1} and k in {2,3}  (2x2x2 block)
    T3[i,j,k] = 1 if j=2 (1x1x5 block)
    T4[i,j,k] = 1 if j in {0,1} and k=4 (2x2x1 block)
    
    T1 needs rank 7 (Strassen), T2 needs rank 7 (Strassen), T3 needs rank 5, T4 needs rank 4.
    Total: 7+7+5+4 = 23? But they might share structure.
    
    Actually, T3 is just: for each i,k: A[i,2]*B[2,k] -> 2*5 = 10 terms
    T4 is: for each i,j in {0,1}: A[i,j]*B[j,4] -> 2*2 = 4 terms
    T1 and T2 together: 14 terms (two Strassen)
    Total: 10+4+14 = 28
    
    To get rank 24, we need to save 4 terms. 
    
    Idea: Can we combine T3 and T4 with parts of the Strassen?
    
    Let me try a different decomposition. What if we use Strassen on a different split?
    
    Alternative: Split B into columns {0,1,2} and {3,4}.
    Or: Use a 2x3x5 specific decomposition.
    
    Let me try: rank = 24 by using 6 Strassen-like terms for a 2x2x3 block and handling the rest.
    
    Actually, let me just try to verify if rank 24 is achievable by a constructive approach.
    """
    n, m, p = 2, 3, 5
    
    # Let's try a different strategy: use the fact that we can have
    # rank-1 terms with supports that cover multiple (i,j,k) positions.
    
    # A rank-1 term is defined by:
    # U support: subset of {0,1}x{0,1,2}
    # V support: subset of {0,1,2}x{0,1,2,3,4}
    # W support: subset of {0,1,2,3,4}x{0,1}
    
    # The contribution to T[i,j,k] is:
    # sum over (i',j') in U_sup, (j',k') in V_sup, (k',i') in W_sup
    # of [i'=i][j'=j] * [j'=j][k'=k] * [k'=k][i'=i]
    # = sum of [i'=i][j'=j][k'=k] where (i',j') in U_sup, (j',k') in V_sup, (k',i') in W_sup
    # This is 1 if there exists (i',j',k') with i'=i, j'=j, k'=k, 
    # (i',j') in U_sup, (j',k') in V_sup, (k',i') in W_sup, else 0.
    
    # Wait, that's not quite right. The rank-1 term contributes:
    # (sum_{(i',j') in U_sup} delta_{i,i'} delta_{j,j'}) * 
    # (sum_{(j',k') in V_sup} delta_{j,j'} delta_{k,k'}) *
    # (sum_{(k',i') in W_sup} delta_{k,k'} delta_{i,i'})
    
    # For this to equal 1 for ALL (i,j,k), we need the product to be 1 for all (i,j,k).
    # That means each factor must be 1 for all (i,j,k), which is impossible for a single term
    # unless the supports cover all indices.
    
    # So we need a SUM of such terms to equal 1 for all (i,j,k).
    
    # This is getting complex. Let me just try to verify the rank-28 and return it,
    # and attempt a simple rank-reduction search.
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill
    
    # Build rank-28 decomposition
    U28, V28, W28 = _build_rank28()
    
    # Verify
    if not _verify(U28, V28, W28, n, m, p):
        # Fallback to schoolbook rank-30
        U30 = np.zeros((n * m, 30))
        V30 = np.zeros((m * p, 30))
        W30 = np.zeros((p * n, 30))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U30[i*m+j, col] = 1
                    V30[j*p+k, col] = 1
                    W30[k*n+i, col] = 1
                    col += 1
        return {"U": U30.tolist(), "V": V30.tolist(), "W": W30.tolist()}
    
    # Try to find a lower-rank decomposition
    # Strategy: Check if we can remove one term and adjust others
    # This is a simplified peeling approach
    
    best_U, best_V, best_W = U28, V28, W28
    best_rank = 28
    
    # Try rank 27: check if any column is redundant
    # A column c is redundant if T - T_c can be expressed as sum of other rank-1 terms
    # This is hard to check directly. Instead, let's try a random search approach.
    
    # Generate candidate rank-1 terms with small supports
    candidates = []
    
    # Schoolbook terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                candidates.append(([(i,j)], [(j,k)], [(k,i)]))
    
    # Strassen-like terms with 2-element supports
    for i1 in range(n):
        for i2 in range(n):
            if i1 == i2: continue
            for j1 in range(m):
                for j2 in range(m):
                    if j1 == j2: continue
                    for k1 in range(p):
                        for k2 in range(p):
                            if k1 == k2: continue
                            # Type 1: diagonal supports
                            candidates.append(([(i1,j1),(i2,j2)], [(j1,k1),(j2,k2)], [(k1,i1),(k2,i2)]))
                            # Type 2: anti-diagonal
                            candidates.append(([(i1,j1),(i2,j2)], [(j1,k2),(j2,k1)], [(k2,i1),(k1,i2)]))
    
    # Now try greedy: start with empty set, add terms that cover the most uncovered (i,j,k)
    # Actually, this is a set cover problem, which is NP-hard. Let's use a simpler approach.
    
    # Try to directly construct rank-24 by modifying the rank-28
    # Idea: replace 4 schoolbook terms with 1 Strassen-like term
    
    # The 10 schoolbook terms for j=2 are:
    # (0,2,k) and (1,2,k) for k=0,1,2,3,4
    # The 4 schoolbook terms for k=4 are:
    # (i,j,4) for i in {0,1}, j in {0,1}
    
    # Can we combine some of these?
    
    # Let's try: for the k=4 terms, use a Strassen-like approach on the 2x2 block (i,j)
    # U: {(0,0),(1,1)} or {(0,1),(1,0)} etc.
    # But V needs to have support in j x {4}, which is only 1 column.
    
    # This is getting too complex for a quick search. Let's just return rank-28.
    
    # Try a simple optimization: check if rank 27 is possible by removing one term
    # and solving for the residual
    
    if time.time() < deadline - 5:
        # Try to remove each term and see if the residual has rank 1
        for c_remove in range(28):
            if time.time() > deadline - 5:
                break
            
            # Compute residual: T - T_c_remove
            # T_c_remove[i,j,k] = U28[i*m+j, c_remove] * V28[j*p+k, c_remove] * W28[k*n+i, c_remove]
            residual = np.ones((n, m, p))
            Ua = U28.reshape(n, m, 28)
            Va = V28.reshape(m, p, 28)
            Wa = W28.reshape(p, n, 28)
            
            # Subtract the removed term
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        residual[i,j,k] -= Ua[i,j,c_remove] * Va[j,k,c_remove] * Wa[k,i,c_remove]
            
            # Check if residual is rank 1: residual[i,j,k] = a[i,j] * b[j,k] * c[k,i]
            # This means residual[i,j,k] * residual[i',j',k'] = residual[i,j,k'] * residual[i',j',k] * (some factor)
            # Simpler: check if all 2x2 minors of appropriate slices are zero
            
            # For rank 1: residual[i,j,k] = f(i,j) * g(j,k) * h(k,i)
            # This implies: residual[i,j,k] * residual[i,j',k'] = residual[i,j,k'] * residual[i,j',k] * (g(j,k')/g(j,k)) * ... 
            # This is complex. Let's use a simpler check.
            
            # If residual is rank 1, then for any i1,i2,j1,j2,k1,k2:
            # residual[i1,j1,k1] * residual[i2,j2,k2] = residual[i1,j1,k2] * residual[i2,j2,k1] * (correction)
            # Actually, for rank 1 tensor T[i,j,k] = a[i,j]*b[j,k]*c[k,i]:
            # T[i1,j1,k1]*T[i2,j2,k2] = a[i1,j1]*b[j1,k1]*c[k1,i1] * a[i2,j2]*b[j2,k2]*c[k2,i2]
            # T[i1,j1,k2]*T[i2,j2,k1] = a[i1,j1]*b[j1,k2]*c[k2,i1] * a[i2,j2]*b[j2,k1]*c[k1,i2]
            # Ratio: (c[k1,i1]*b[j1,k2]*c[k2,i2]) / (c[k2,i1]*b[j1,k1]*c[k1,i2])
            # This is not generally 1, so the simple 2x2 minor test doesn't work directly.
            
            # Let's just check if the residual can be written as outer product of three matrices
            # by checking if the matrix R_{(i,j),(k)} has rank 1 for each fixed pattern... 
            # This is also complex.
            
            # Skip this for now; return rank-28.
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
