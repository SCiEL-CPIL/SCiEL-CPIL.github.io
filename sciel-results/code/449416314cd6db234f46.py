"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _strassen_2x2x2():
    """Return 7 Strassen 2x2x2 rank-1 terms as sparse dicts."""
    return [
        ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
        ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
        ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
        ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
        ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
        ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
        ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
    ]

def _build_2x2x3_rank10():
    """
    Attempt to build a rank-10 decomposition for 2x2 x 2x3 -> 2x3.
    Strategy: Use Strassen for cols 0-1 (7 terms) and try to handle col 2 with only 3 terms
    by sharing structure with the Strassen terms.
    
    Actually, let's try a different approach: use 10 terms by modifying the trivial part.
    For col 2: C[0,2] = A[0,0]B[0,2] + A[0,1]B[1,2], C[1,2] = A[1,0]B[0,2] + A[1,1]B[1,2]
    We need 4 terms for col 2 in the trivial case. Can we do 3?
    
    Let's try: 
    T7: (A[0,0]+A[1,0]) * B[0,2] -> contributes to C[0,2] and C[1,2]
    T8: (A[0,1]+A[1,1]) * B[1,2] -> contributes to C[0,2] and C[1,2]
    T9: (A[0,0]-A[1,0]) * (B[0,2]) ... this doesn't work cleanly.
    
    Let me just try to find any rank-10 decomposition by search.
    For now, fall back to rank 11 if rank 10 doesn't work.
    """
    # Start with rank 11 (Strassen + trivial)
    terms = []
    for s in range(7):
        u, v, w = _strassen_2x2x2()[s]
        # Map V indices: (j,k) where k in {0,1} -> (j,k) in 2x2x3
        v_new = {}
        for (j,k), c in v.items():
            if k < 2:
                v_new[(j,k)] = c
        # Map W indices: (k,i) where k in {0,1} -> (k,i) in 2x2x3
        w_new = {}
        for (k,i), c in w.items():
            if k < 2:
                w_new[(k,i)] = c
        terms.append((u, v_new, w_new))
    
    # Trivial for column 2: 4 terms
    terms.append(({(0,0):1}, {(0,2):1}, {(2,0):1}))
    terms.append(({(0,1):1}, {(1,2):1}, {(2,0):1}))
    terms.append(({(1,0):1}, {(0,2):1}, {(2,1):1}))
    terms.append(({(1,1):1}, {(1,2):1}, {(2,1):1}))
    
    return terms  # rank 11

def _tensor_product_term(u1, v1, w1, u2, v2, w2):
    """Compute tensor product of two rank-1 terms for 2x2x2 x 2x2x3 -> 4x4x6."""
    U_g, V_g, W_g = {}, {}, {}
    for (i1,j1),c1 in u1.items():
        for (i2,j2),c2 in u2.items():
            k=(2*i1+i2, 2*j1+j2)
            U_g[k]=U_g.get(k,0)+c1*c2
    for (j1,k1),c1 in v1.items():
        for (j2,k2),c2 in v2.items():
            k=(2*j1+j2, 3*k1+k2)
            V_g[k]=V_g.get(k,0)+c1*c2
    for (k1,i1),c1 in w1.items():
        for (k2,i2),c2 in w2.items():
            k=(3*k1+k2, 2*i1+i2)
            W_g[k]=W_g.get(k,0)+c1*c2
    return U_g, V_g, W_g

def _build_decomposition(n, m, p, rank_223=11):
    """Build the Kronecker product decomposition."""
    s22 = _strassen_2x2x2()
    t223 = _build_2x2x3_rank10() if rank_223 == 10 else _build_2x2x3_rank10()  # always 11 for now
    actual_rank_223 = len(t223)
    terms = []
    for s in range(7):
        for t in range(actual_rank_223):
            terms.append(_tensor_product_term(s22[s][0],s22[s][1],s22[s][2],
                                              t223[t][0],t223[t][1],t223[t][2]))
    rank = len(terms)
    U=np.zeros((n*m,rank)); V=np.zeros((m*p,rank)); W=np.zeros((p*n,rank))
    for r,(ud,vd,wd) in enumerate(terms):
        for (i,j),c in ud.items(): U[i*m+j,r]=c
        for (j,k),c in vd.items(): V[j*p+k,r]=c
        for (k,i),c in wd.items(): W[k*n+i,r]=c
    return U,V,W

def _verify(U,V,W,n,m,p,atol=1e-10):
    np.random.seed(42)
    A=np.random.randn(n,m); B=np.random.randn(m,p)
    C_exp=A@B
    C=np.zeros((n,p))
    for r in range(U.shape[1]):
        a=np.sum(A*U[:,r].reshape(n,m))
        b=np.sum(B*V[:,r].reshape(m,p))
        C+=a*b*W[:,r].reshape(p,n).T
    return np.allclose(C,C_exp,atol=atol)

def _try_reduce_rank_linear(U,V,W,n,m,p,max_time=100.0):
    """
    Try to reduce rank by finding linear dependencies among the rank-1 terms.
    A set of rank-1 terms {u_r, v_r, w_r} has a linear dependency if there exist
    coefficients c_r (not all zero) such that sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0.
    
    We can detect this by checking if the columns of the "coefficient tensor" are
    linearly dependent. Specifically, form the matrix M where M[r, (i,j,k)] = 
    U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]. If columns of M are linearly dependent,
    we can reduce rank.
    
    More practically: check if any column r can be expressed as a linear combination
    of other columns by solving a least-squares problem.
    """
    start = time.time()
    rank = U.shape[1]
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Build the tensor coefficient matrix
    # M has shape (rank, n*m*p) where M[r, idx] = U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    # This is expensive for rank=77, n*m*p=96, so 77*96 = 7392 entries
    
    # More efficient: use the structure. For each r, the term is u_r ⊗ v_r ⊗ w_r.
    # We want to find if sum_r c_r * u_r ⊗ v_r ⊗ w_r = 0 for some non-trivial c.
    
    # Approach: randomly project to check for dependencies
    # Or: use the fact that if we fix two of the three tensors, the third is determined.
    
    # Practical approach: try to eliminate one term at a time by checking if it's
    # in the span of the others.
    
    # For term r, we need: u_r ⊗ v_r ⊗ w_r = sum_{s != r} c_s * u_s ⊗ v_s ⊗ w_s
    # This is a system of n*m*p equations in (rank-1) unknowns.
    
    # Let's try a simpler heuristic: check for duplicate or proportional terms first
    for attempt in range(rank):
        if time.time() - start > max_time:
            break
        
        # Find the term with smallest "energy" to try eliminating
        energies = []
        for r in range(rank):
            e = np.sum(U[:,r]**2) * np.sum(V[:,r]**2) * np.sum(W[:,r]**2)
            energies.append(e)
        
        # Try eliminating the term with smallest energy
        r_elim = int(np.argmin(energies))
        
        # Check if term r_elim is in the span of others
        # Build the tensor for term r_elim
        u_r = U[:, r_elim].reshape(n, m)
        v_r = V[:, r_elim].reshape(m, p)
        w_r = W[:, r_elim].reshape(p, n)
        tensor_r = np.einsum('ij,jk,ki->ijk', u_r, v_r, w_r)
        
        # Build the coefficient matrix for other terms
        # M[s, (i,j,k)] = U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s]
        other_indices = [s for s in range(rank) if s != r_elim]
        n_other = len(other_indices)
        
        # Flatten tensor_r
        target = tensor_r.flatten()
        
        # Build matrix A where A[:, s] = flatten(u_s ⊗ v_s ⊗ w_s)
        # This is expensive, so let's be smart
        A_mat = np.zeros((n*m*p, n_other))
        for idx, s in enumerate(other_indices):
            u_s = U[:, s].reshape(n, m)
            v_s = V[:, s].reshape(m, p)
            w_s = W[:, s].reshape(p, n)
            tensor_s = np.einsum('ij,jk,ki->ijk', u_s, v_s, w_s)
            A_mat[:, idx] = tensor_s.flatten()
        
        # Solve A_mat @ c = target
        try:
            c, residuals, rank_a, sv = np.linalg.lstsq(A_mat, target, rcond=1e-10)
            residual = np.linalg.norm(A_mat @ c - target)
            if residual < 1e-8 * (np.linalg.norm(target) + 1e-10):
                # Term r_elim is in the span of others! Remove it.
                # But we need to be careful: the coefficients c give us the representation,
                # but we don't need to adjust the other terms because the decomposition
                # is already valid - we just remove the redundant term.
                # Actually, no: if tensor_r = sum_s c_s * tensor_s, then the original
                # decomposition has a redundancy. We can simply remove term r_elim.
                U = np.delete(U, r_elim, axis=1)
                V = np.delete(V, r_elim, axis=1)
                W = np.delete(W, r_elim, axis=1)
                rank -= 1
                best_rank = rank
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        except:
            pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    deadline = start_time + 110.0
    
    # Build the 77-term decomposition
    U, V, W = _build_decomposition(n, m, p)
    
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        rank_school = n * m * p
        U = np.zeros((n * m, rank_school), dtype=int)
        V = np.zeros((m * p, rank_school), dtype=int)
        W = np.zeros((p * n, rank_school), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank
    if time.time() < deadline - 30:
        U2, V2, W2, rank2 = _try_reduce_rank_linear(U, V, W, n, m, p, 
                                                     max_time=min(80, deadline - time.time() - 15))
        if rank2 < U.shape[1] and _verify(U2, V2, W2, n, m, p):
            U, V, W = U2, V2, W2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
