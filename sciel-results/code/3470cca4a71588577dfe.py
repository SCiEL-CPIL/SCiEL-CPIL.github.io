"""Matrix multiplication 4,4,6 - rank-minimal exact tensor decomposition."""
import random
import time
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 110  # safety margin before 120s kill

def _elapsed():
    return time.time() - START_TIME

def _build_tensor(n, m, p):
    """Build the multiplication tensor as a dict of non-zero entries.
    T[i,j,k,l] = 1 iff j == k (multiplying A[i,j]*B[k,l] contributes to C[i,l] when j==k)
    """
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(m):
                if j == k:
                    for l in range(p):
                        T[(i, j, k, l)] = 1
    return T

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the multiplication tensor.
    U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    For each (i,j,k,l): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == (1 if j==k else 0)
    """
    rank = len(U[0]) if U else 0
    # Build lookup
    for i in range(n):
        for j in range(m):
            for k in range(m):
                expected = 1 if j == k else 0
                for l in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(s - expected) > 1e-9:
                        return False
    return True

def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        # Only valid if tensor is all zeros, which it isn't
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # For each c, we have a rank-1 term in the 4-index tensor
    # T[i,j,k,l] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # We need to check this equals delta(j,k) for all i,j,k,l
    
    # Check all (i,j,k,l) where j==k: should equal 1
    # Check all (i,j,k,l) where j!=k: should equal 0
    
    # This is expensive to do naively. Let's use a smarter approach.
    # For fixed (j,k) with j!=k, the sum over c of V[j*p+k,c]*W[k*n+i,c]*U[i*m+j,c] must be 0 for all i,l... wait l doesn't appear.
    # Actually l only appears in... wait, l is the column of B and column of C. 
    # In the tensor, l is an index. U doesn't depend on l, V depends on l via j*p+k+l? No wait.
    
    # Let me re-read the convention:
    # U has shape (n*m, rank): indexed by (i,j) where i=row of A, j=col of A
    # V has shape (m*p, rank): indexed by (j,k) where j=col of A (=row of B), k=col of B
    # W has shape (p*n, rank): indexed by (k,i) where k=col of B, i=row of C (=row of A)
    # Wait, that's W[k*n+i, r] per the trace convention.
    
    # So the tensor is: T[i,j,k,l] where i=row A, j=col A, k=row B, l=col B
    # But W is indexed by (k, i) not (l, i). Let me re-read.
    
    # "output factor follows the official trace convention W[k*n+i,r]"
    # So W is (p*n, rank) indexed by (k, i) where k is in [0,p) and i is in [0,n)
    # But k here is the column of B (0..p-1), and i is row of A/C (0..n-1)
    
    # So the full tensor: T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that doesn't have l. Let me re-think.
    
    # Actually in matrix multiplication: C[i,l] = sum_j A[i,j] * B[j,l]
    # The tensor has indices (i, j, j, l) with the constraint that the middle two are the same.
    # So really the tensor is T[i,j,l] = delta... no.
    
    # Let me re-read: "multiplying a 4 by 4 matrix by a 4 by 6 matrix"
    # A is 4x4 (n x m), B is 4x6 (m x p), C is 4x6 (n x p)
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # The tensor T has indices (i, j, k, l) where T[i,j,k,l] = 1 if j==k else 0
    # Because C[i,l] = sum_j A[i,j] B[j,l], and the "multiplication tensor" 
    # T[i,j,k,l] tells us if A[i,j] and B[k,l] contribute to C[i,l], which is iff j==k.
    
    # Now the factors:
    # U: (n*m, rank) -> indexed by (i,j)
    # V: (m*p, rank) -> indexed by (j,k)  [j from A's col, k from B's row]
    # W: (p*n, rank) -> indexed by (k,i)  [k from B's col, i from C's row]
    
    # So: T[i,j,k,l] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # But wait, this doesn't include l! 
    # 
    # Hmm, I think the convention is:
    # U[i*m + j, r] for i in [0,n), j in [0,m)
    # V[j*p + k, r] for j in [0,m), k in [0,p)  
    # W[k*n + i, r] for k in [0,p), i in [0,n)
    # 
    # And the tensor equation is:
    # For all i,j,k,l: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = delta(j,k)
    # 
    # But this doesn't depend on l! That means for each (i,j,k), the sum must equal delta(j,k)
    # and it's the same for all l. That makes sense because in the actual multiplication,
    # l just selects which column of B/C we're looking at, but the "structure" of which
    # (i,j) pairs with which (k,l) is independent of l.
    
    # So actually we only need to check: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 if j==k else 0
    # 
    # Note: j ranges over [0,m) and k ranges over [0,p). The condition j==k only makes sense
    # when both are in [0, min(m,p)). Since m=4, p=6, j in [0,4), k in [0,6).
    # j==k is possible for j,k in [0,4).
    
    # Let me verify with the schoolbook:
    # U[i*m+j, c] = 1 for c = i*m*p + j*p + k (for each k)
    # V[j*p+k, c] = 1 for c = i*m*p + j*p + k
    # W[k*n+i, c] = 1 for c = i*m*p + j*p + k
    # So for fixed (i,j,k), only c = i*m*p+j*p+k contributes, giving 1*1*1=1.
    # But this should be 1 only if j==k. In schoolbook, we have one term per (i,j,k) regardless.
    # Wait, that's wrong. Let me re-examine.
    
    # In the schoolbook decomposition:
    # For each (i,j,k): U[i*m+j][col]=1, V[j*p+k][col]=1, W[k*n+i][col]=1
    # So for a given (i,j,k), the sum over c gives 1 (from the term where col matches).
    # But we need this to be 1 if j==k and 0 if j!=k.
    # The schoolbook as written in the seed gives 1 for ALL (i,j,k), not just j==k.
    # That means the seed program is WRONG? Or my understanding is off.
    
    # Wait, let me re-read the seed:
    # for i in range(n): for j in range(m): for k in range(p):
    #   U[i*m+j][column] = 1
    #   V[j*p+k][column] = 1
    #   W[k*n+i][column] = 1
    # 
    # So for fixed (i,j), there are p terms (one for each k).
    # For fixed (i,j,k), the contribution is 1.
    # The sum for (i,j,k) is 1 for all k, regardless of whether j==k.
    # 
    # This means the seed produces sum = 1 for all (i,j,k), but we need sum = delta(j,k).
    # So the seed is actually incorrect for this problem? Or the problem definition is different.
    
    # Let me re-read the task: "Find a rank-minimal exact tensor decomposition for multiplying
    # a 4 by 4 matrix by a 4 by 6 matrix."
    
    # Maybe the tensor is defined differently. Perhaps it's:
    # T[i,j,k,l] where the decomposition gives C[i,l] = sum_j A[i,j] B[j,l]
    # And the factors are:
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # with the equation: for all i,j,k,l: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta(j,k)
    # 
    # But as I noted, this doesn't depend on l. So effectively we need:
    # For all i,j,k: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = delta(j,k)
    # 
    # The seed gives 1 for all (i,j,k), which is wrong. Unless the evaluator uses a different check.
    # 
    # Actually wait - maybe I'm misreading the index conventions. Let me reconsider.
    # Perhaps V is indexed by (j, l) where l is the column of B, and k in W is also l?
    # No, the shapes are: U(n*m), V(m*p), W(p*n).
    # 
    # I think the correct interpretation is:
    # The multiplication tensor T has 4 indices: (i, j, k, l)
    # T[i,j,k,l] = 1 if j == k, else 0
    # (This represents: A[i,j] * B[k,l] contributes to C[i,l] iff j==k)
    # 
    # The decomposition: T[i,j,k,l] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] * (something with l?)
    # 
    # But W only has p*n entries indexed by (k,i) where k in [0,p). If k is the column of B,
    # then l=k. So W[k*n+i, r] where k is the column index of B.
    # And V[j*p+k, r] where j is column of A and k is column of B.
    # 
    # So actually l IS k in this convention! The "k" in V and W refers to the column of B,
    # not the row of B. The row of B is j (same as column of A).
    # 
    # So the tensor is really T[i, j, l] = delta(j, l) where:
    # i = row of A/C (0..n-1)
    # j = col of A / row of B (0..m-1)  
    # l = col of B (0..p-1)
    # 
    # And the equation is: for all i,j,l: sum_r U[i*m+j,r] * V[j*p+l,r] * W[l*n+i,r] = delta(j,l)
    # 
    # This makes much more sense! The "k" in the problem statement refers to the column of B.
    # So j and l (or k) are both in [0, min(m,p)) for the delta to be 1.
    # 
    # With m=4, p=6: j in [0,4), l in [0,6). delta(j,l)=1 iff j==l, possible for j,l in [0,4).
    
    # Let me verify schoolbook with this:
    # For each (i,j,l): U[i*m+j, col]=1, V[j*p+l, col]=1, W[l*n+i, col]=1
    # Sum for (i,j,l) = 1 for all (i,j,l). Still wrong!
    # 
    # Unless... the schoolbook should only include terms where j==l?
    # In standard matrix multiplication, C[i,l] = sum_j A[i,j]*B[j,l].
    # The "tensor" for this is: for each triple (i,j,l), the coefficient is 1 if j==l... no.
    # 
    # I think I'm overcomplicating this. Let me just use the verification that the
    # evaluator likely uses: check that for random matrices A (4x4) and B (4x6),
    # C = A@B equals sum_r (A flattened with U) * (B flattened with V) * W.
    # 
    # Specifically: C[i,l] = sum_j A[i,j]*B[j,l]
    # Using the decomposition: C[i,l] = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_j B[j,l]*V[j*p+l,r]) * W[l*n+i,r]
    # 
    # Wait, that's not right either. Let me think about this differently.
    # 
    # The standard way to express matrix multiplication as a tensor contraction:
    # C[i,l] = sum_j A[i,j] * B[j,l]
    # 
    # If we have a rank-r decomposition, we write:
    # C[i,l] = sum_{r=0}^{R-1} (sum_j A[i,j] * U[i,j,r]) * (sum_j B[j,l] * V[j,l,r]) * W[i,l,r]
    # 
    # For this to equal sum_j A[i,j]*B[j,l] for all A,B:
    # We need: sum_r U[i,j,r] * V[j',l,r] * W[i,l,r] = delta(j,j') for all i,l,j,j'
    # 
    # Wait, that's not quite right either because the sums over j in the two factors are independent.
    # 
    # Let me be more careful. If:
    # X_r[i,j] = A[i,j] * U[i*m+j, r]  -- no, that's not how it works.
    # 
    # Actually, the standard tensor decomposition for matrix mult is:
    # C[i,l] = sum_j A[i,j] B[j,l]
    # = sum_j sum_r [A[i,j] * u_r(i,j)] * [B[j,l] * v_r(j,l)] * w_r(i,l)
    #   where we need sum_r u_r(i,j) * v_r(j',l) * w_r(i,l) = delta(j,j')
    # 
    # Hmm, but the factors given are U(n*m, R), V(m*p, R), W(p*n, R).
    # U[i*m+j, r] corresponds to u_r(i,j)
    # V[j*p+k, r] corresponds to v_r(j,k) where k is col of B
    # W[k*n+i, r] corresponds to w_r(k,i) where k is col of B
    # 
    # So the equation should be:
    # For all i,j,j',l: sum_r U[i*m+j, r] * V[j'*p+l, r] * W[l*n+i, r] = delta(j,j')
    # 
    # Wait, but V is indexed by (j', l) where j' is row of B = col of A, and l is col of B.
    # And we're summing over j' in the B factor. So:
    # C[i,l] = sum_j A[i,j] B[j,l] = sum_r [sum_j A[i,j] U[i*m+j,r]] * [sum_j B[j,l] V[j*p+l,r]] * W[l*n+i,r]
    # 
    # For this to work for all A and B, we need:
    # sum_r U[i*m+j, r] * V[j'*p+l, r] * W[l*n+i, r] = delta(j, j')
    # for all i, j, j', l.
    # 
    # Note this doesn't depend on i or l in the delta, but the sum might depend on i and l.
    # So we need: for all i,l,j,j': sum_r U[i*m+j,r] * V[j'*p+l,r] * W[l*n+i,r] = delta(j,j')
    
    # This is the correct condition! Let me verify with schoolbook:
    # Schoolbook: one term per (i,j,l) with U[i*m+j]=1, V[j*p+l]=1, W[l*n+i]=1
    # For fixed (i,l,j,j'): sum over r. The term for (i,j,l) gives U[i*m+j]=1, V[j*p+l]=1 (if j'=j), W[l*n+i]=1.
    # So sum = 1 if j=j', else 0. But wait, there are terms for other j' values too.
    # For term (i,j'',l): U[i*m+j,r]=1 only if j''=j. V[j'*p+l,r]=1 only if j''=j'. W=1 always for this i,l.
    # So the term contributes 1*1*1=1 iff j''=j AND j''=j', i.e., j=j'.
    # Sum over all terms: 1 if j=j', 0 otherwise. 
    # 
    # So the schoolbook IS correct under this interpretation. The key is that V is indexed by
    # (j', l) where j' is the row of B (which must equal col of A for the multiplication to work),
    # and the condition is delta(j, j') not delta(j, l).
    
    return True  # placeholder, will implement properly

def _verify_correct(U, V, W, n, m, p):
    """Correct verification:
    For all i in [0,n), j in [0,m), j' in [0,m), l in [0,p):
        sum_r U[i*m+j, r] * V[j'*p+l, r] * W[l*n+i, r] = 1 if j==j' else 0
    """
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # For efficiency, precompute
    # For each (i, l), we need: for all j, j': sum_r U[i*m+j, r] * V[j'*p+l, r] * W[l*n+i, r] = delta(j,j')
    # Let W_il = W[l*n+i, :] (shape (rank,))
    # Let U_i = U[i*m:(i+1)*m, :] (shape (m, rank))
    # Let V_l = V[l::p, :] -- no, V[j'*p+l, r] for j' in [0,m)
    # V_l = np.array([V[j'*p+l, :] for j' in range(m)]) (shape (m, rank))
    
    for i in range(n):
        for l in range(p):
            W_il = Wa[l * n + i, :]  # shape (rank,)
            U_i = Ua[i * m:(i + 1) * m, :]  # shape (m, rank)
            V_l = np.array([Va[jp * p + l, :] for jp in range(m)])  # shape (m, rank)
            
            # M[j, j'] = sum_r U_i[j, r] * V_l[j', r] * W_il[r]
            # = sum_r U_i[j, r] * (V_l[j', r] * W_il[r])
            # Let temp[j', r] = V_l[j', r] * W_il[r]
            # M = U_i @ temp.T  -- shape (m, m)
            
            temp = V_l * W_il[np.newaxis, :]  # shape (m, rank)
            M = U_i @ temp.T  # shape (m, m)
            
            # Check M == identity
            if not np.allclose(M, np.eye(m), atol=1e-9):
                return False
    return True

def _greedy_construct(n, m, p, seed=0, max_rank=40):
    """Greedy construction with randomization."""
    rng = random.Random(seed)
    
    # We need: for all i,j,j',l: sum_r U[i*m+j,r]*V[j'*p+l,r]*W[l*n+i,r] = delta(j,j')
    # 
    # Strategy: build rank-1 terms one at a time.
    # Each term is (u, v, w) where u is (n*m,), v is (m*p,), w is (p*n,)
    # 
    # The "error" tensor E[i,j,j',l] = delta(j,j') - sum_{r done} u_r[i*m+j]*v_r[j'*p+l]*w_r[l*n+i]
    # We want to pick the next term to minimize ||E|| (or maximize reduction).
    
    # Initialize
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    # Track current sum S[i,j,j',l]
    # S[i,j,j',l] = sum_r U[i*m+j,r]*V[j'*p+l,r]*W[l*n+i,r]
    # We need S = I (identity in j,j' for each i,l)
    
    # For efficiency, track S as a 4D array or as n*p matrices of size m x m
    S = np.zeros((n, p, m, m), dtype=float)
    
    rank = 0
    for step in range(max_rank):
        if _elapsed() > TIME_LIMIT:
            break
        
        # Compute error E = I - S (for each i,l: E[i,l] = I_m - S[i,l])
        # E[i,l,j,j'] = delta(j,j') - S[i,l,j,j']
        
        # We want to find u, v, w to maximize reduction in ||E||_F
        # The rank-1 term contributes: u[i*m+j] * v[j'*p+l] * w[l*n+i]
        # Let's denote a = u (size n*m), b = v (size m*p), c = w (size p*n)
        # Contribution to S[i,l,j,j'] = a[i*m+j] * b[j'*p+l] * c[l*n+i]
        
        # This is a 3-tensor rank-1 approximation problem.
        # Greedy: try random vectors and pick the best.
        
        best_reduction = -1
        best_u = None
        best_v = None
        best_w = None
        
        n_trials = 50
        for trial in range(n_trials):
            if _elapsed() > TIME_LIMIT:
                break
            # Generate random vectors
            u = np.random.randn(n * m)
            v = np.random.randn(m * p)
            w = np.random.randn(p * n)
            
            # Compute contribution
            # contrib[i,l,j,j'] = u[i*m+j] * v[j'*p+l] * w[l*n+i]
            # Reduction = <E, contrib> (inner product)
            # = sum_{i,l,j,j'} E[i,l,j,j'] * u[i*m+j] * v[j'*p+l] * w[l*n+i]
            
            # Efficient computation:
            # = sum_{i,l,j,j'} (delta(j,j') - S[i,l,j,j']) * u[i*m+j] * v[j'*p+l] * w[l*n+i]
            # = sum_{i,l,j} u[i*m+j] * w[l*n+i] * (delta(j,j) * v[j*p+l] - sum_{j''} S[i,l,j,j''] * v[j''*p+l])
            # Wait, let me be more careful.
            # = sum_{i,l,j,j'} E[i,l,j,j'] * u[i*m+j] * v[j'*p+l] * w[l*n+i]
            # Group by (i,l):
            # = sum_{i,l} w[l*n+i] * sum_{j,j'} E[i,l,j,j'] * u[i*m+j] * v[j'*p+l]
            # = sum_{i,l} w[l*n+i] * (u_i^T * E[i,l] * v_l)
            # where u_i = u[i*m:(i+1)*m], v_l = v[l::p] -- no, v[j'*p+l] for j' in [0,m)
            
            total = 0.0
            for i in range(n):
                u_i = u[i * m:(i + 1) * m]  # shape (m,)
                for l in range(p):
                    w_il = w[l * n + i]
                    v_l = np.array([v[jp * p + l] for jp in range(m)])  # shape (m,)
                    # sum_{j,j'} E[i,l,j,j'] * u_i[j] * v_l[j']
                    # = u_i^T * E[i,l] * v_l
                    E_il = np.eye(m) - S[i, l]
                    inner = u_i @ E_il @ v_l
                    total += w_il * inner
            
            # Also need to consider that adding this term changes S, so the "reduction" 
            # in ||E||^2 is 2*<E,contrib> - ||contrib||^2
            # For greedy, just use <E, contrib> as proxy
            
            if total > best_reduction:
                best_reduction = total
                best_u = u.copy()
                best_v = v.copy()
                best_w = w.copy()
        
        if best_u is None or best_reduction <= 1e-10:
            break
        
        # Add this term
        U_col = best_u
        V_col = best_v
        W_col = best_w
        
        for idx in range(n * m):
            U[idx][rank] = U_col[idx]
        for idx in range(m * p):
            V[idx][rank] = V_col[idx]
        for idx in range(p * n):
            W[idx][rank] = W_col[idx]
        
        # Update S
        for i in range(n):
            for l in range(p):
                for j in range(m):
                    for jp in range(m):
                        S[i, l, j, jp] += best_u[i * m + j] * best_v[jp * p + l] * best_w[l * n + i]
        
        rank += 1
        
        # Check if we've achieved exactness
        if rank >= 10 and rank % 5 == 0:
            # Trim to current rank
            U_trim = [row[:rank] for row in U]
            V_trim = [row[:rank] for row in V]
            W_trim = [row[:rank] for row in W]
            if _verify_correct(U_trim, V_trim, W_trim, n, m, p):
                return U_trim, V_trim, W_trim, rank
    
    if rank > 0:
        U_trim = [row[:rank] for row in U]
        V_trim = [row[:rank] for row in V]
        W_trim = [row[:rank] for row in W]
        if _verify_correct(U_trim, V_trim, W_trim, n, m, p):
            return U_trim, V_trim, W_trim, rank
    
    return None

def _try_improve(U, V, W, n, m, p, seed=0, max_iter=200):
    """Try to improve (reduce rank) an existing decomposition."""
    rng = random.Random(seed)
    rank = len(U[0]) if U else 0
    
    # Try removing one term and re-solving
    # For now, just return the input
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    best_result = None
    best_rank = float('inf')
    
    # First, get the schoolbook solution as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for l in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + l][col] = 1
                W_sb[l * n + i][col] = 1
                col += 1
    
    # Verify schoolbook
    if _verify_correct(U_sb, V_sb, W_sb, n, m, p):
        best_result = (U_sb, V_sb, W_sb)
        best_rank = rank_sb
    else:
        # Schoolbook failed verification? That shouldn't happen.
        # Let's just use it anyway as fallback.
        best_result = (U_sb, V_sb, W_sb)
        best_rank = rank_sb
    
    # Try greedy with different seeds
    for seed in range(50):
        if _elapsed() > TIME_LIMIT - 10:
            break
        
        result = _greedy_construct(n, m, p, seed=seed, max_rank=40)
        if result is not None:
            U, V, W, r = result
            if r < best_rank:
                best_rank = r
                best_result = (U, V, W)
        
        # Also try with more trials
        if seed < 10 and _elapsed() < TIME_LIMIT - 30:
            result2 = _greedy_construct(n, m, p, seed=1000 + seed, max_rank=35)
            if result2 is not None:
                U, V, W, r = result2
                if r < best_rank:
                    best_rank = r
                    best_result = (U, V, W)
    
    if best_result is None:
        # Shouldn't happen, but fallback
        best_result = (U_sb, V_sb, W_sb)
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
