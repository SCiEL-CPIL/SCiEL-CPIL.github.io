"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that U,V,W exactly decompose the matrix multiplication tensor.
    
    The tensor T[i,j,k] = delta_{i,k} in the sense that (AB)[i,k] = sum_j A[i,j] B[j,k].
    We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if (i,j,k) is a "valid" triple
    Actually the decomposition is: T_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    where T_{i,j,k} = 1 for all i,j,k (the multiplication tensor has all entries 1 in the
    standard basis? No...)
    
    Actually, the matrix multiplication tensor M_{i,j,k} satisfies:
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    The tensor decomposition means: M_{i,j,k} = sum_r U_{i,j}^{(r)} V_{j,k}^{(r)} W_{k,i}^{(r)}
    where M_{i,j,k} = 1 for all valid (i,j,k) in the range.
    
    Wait, that's not right either. Let me think more carefully.
    
    The matrix multiplication tensor is a 3-tensor T of shape (n, m, p) where
    T[i,j,k] = 1 for all i,j,k? No.
    
    Actually, the standard way: the multiplication map is a trilinear form.
    (AB)_{i,k} = sum_j A_{i,j} B_{j,k}
    
    This is a bilinear map from A (n x m) and B (m x p) to C (n x p).
    As a tensor, it's T in R^{n*m} x R^{m*p} x R^{n*p} (or with the trace convention W[k*n+i,r]).
    
    The rank-r decomposition: for each r, we have vectors u_r in R^{n*m}, v_r in R^{m*p}, w_r in R^{n*p}
    such that T = sum_r u_r ⊗ v_r ⊗ w_r (outer product / Khatri-Rao).
    
    Concretely: T[i*m+j, j'*p+k, k'*n+i'] = sum_r U[i*m+j, r] * V[j'*p+k, r] * W[k'*n+i', r]
    
    But T is the matrix multiplication tensor. The non-zero entries of T are:
    T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k (where the indices match up).
    
    Actually, let me reconsider. The matrix multiplication tensor M has shape (n*m, m*p, n*p) in the
    flattened sense, but really it's a 3-way tensor. The entry M[a,b,c] where a indexes (i,j), b indexes (j',k), c indexes (i',k') is:
    M[i,j, j',k, i',k'] = delta_{j,j'} * delta_{i,i'} * delta_{k,k'}
    
    So M is non-zero only when j=j', i=i', k=k', and then it equals 1.
    
    In the flattened indices: M[i*m+j, j*p+k, k'*n+i'] = 1 if j=j' and i=i' and k=k', else 0.
    Wait, that's not quite right. Let me be more careful.
    
    A is n x m, B is m x p, C = AB is n x p.
    C[i,k] = sum_j A[i,j] B[j,k]
    
    The tensor T has indices (i, j, k) -> coefficient 1, meaning:
    T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    And the decomposition is:
    T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    
    where U has shape (n*m, rank) with U[i*m+j, r], V has shape (m*p, rank) with V[j*p+k, r],
    W has shape (p*n, rank) with W[k*n+i, r].
    
    So we need: for all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    And for all other combinations (i,j,k) where the indices don't form a valid triple... wait, 
    actually every (i,j,k) with i<n, j<m, k<p IS a valid triple. So T[i,j,k] = 1 for ALL i,j,k.
    
    Hmm, but that can't be right because then the tensor would be all ones and its rank would be 1.
    
    Let me reconsider. The matrix multiplication tensor is NOT all ones. 
    
    The correct statement: the multiplication map is defined by the tensor M where
    M[i, j, k] (with i indexing rows of A, j indexing columns of A / rows of B, k indexing columns of B)
    equals 1. But the way it's used is:
    
    C[i,k] = sum_j A[i,j] B[j,k]
    
    So the "tensor" that encodes this is actually a linear map from (A,B) to C. The tensor M 
    in the tensor product space has entries M[i,j,k] = 1 for all i,j,k, BUT the way it's 
    contracted is specific.
    
    I think the correct interpretation for the decomposition is:
    
    We want U, V, W such that for ALL matrices A (n x m) and B (m x p):
    C = AB where C[i,k] = sum_j A[i,j] B[j,k]
    
    And the decomposition says:
    C[i,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k' B[j',k'] V[j'*p+k', r]) * W[k*n+i, r]
    
    No wait, that's not right either. Let me look at the schoolbook example:
    
    U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1 for col corresponding to (i,j,k).
    
    So for a given (i,j,k), there's exactly one column r where U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
    
    The product sum_r U[i*m+j,r] * V[j'*p+k',r] * W[k''*n+i'',r] = 1 only if there exists a column
    where all three are 1 simultaneously, which requires (i,j) = (i,j), (j',k') = (j,k), (k'',i'') = (k,i).
    So j'=j, k'=k, k''=k, i''=i. Thus the product is 1 iff i''=i, j'=j, k'=k, k''=k.
    
    So sum_r U[i*m+j,r] V[j'*p+k',r] W[k''*n+i'',r] = delta_{i,i''} * delta_{j,j'} * delta_{k,k'} * delta_{k,k''}
    
    Hmm, that gives a tensor with entries delta_{i,i''} delta_{j,j'} delta_{k,k'} delta_{k,k''}, which 
    is 1 when i=i'', j=j', k=k', k=k''. Since k=k' and k=k'' implies k'=k'', this is consistent.
    
    So the tensor T[i*m+j, j'*p+k', k''*n+i''] = delta_{i,i''} delta_{j,j'} delta_{k,k'}
    
    And this is the matrix multiplication tensor! Because:
    C[i,k] = sum_j A[i,j] B[j,k]
    
    If we think of A and B as vectors in R^{n*m} and R^{m*p}, and C as a vector in R^{n*p}, then:
    C[k*n+i] = sum_{j,j',k'} T[i*m+j, j'*p+k', k*n+i] * A[i*m+j] * B[j'*p+k']
    
    Wait, I'm overcomplicating this. Let me just verify the schoolbook works.
    
    For the schoolbook: U[i*m+j, (i,j,k)] = 1, V[j*p+k, (i,j,k)] = 1, W[k*n+i, (i,j,k)] = 1.
    
    The bilinear form: for fixed i, k, C[i,k] should equal sum_j A[i,j] B[j,k].
    
    Using the decomposition: C[k*n+i] = sum_r W[k*n+i, r] * (sum_j A[i*m+j, ...]) 
    
    Actually, I think the correct check is:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    Let's verify: for a fixed (i,j,k), which columns r have U[i*m+j,r] != 0? Only the column 
    corresponding to (i,j,k) itself, where U=1. For that column, V[j*p+k, r] = 1 and W[k*n+i, r] = 1.
    So the sum is 1. Good.
    
    But we also need it to be 0 for "invalid" combinations? No, there are no invalid combinations.
    Every (i,j,k) with i<n, j<m, k<p is valid.
    
    Wait, but the tensor has dimensions (n*m) x (m*p) x (p*n). The first index ranges over n*m values,
    the second over m*p values, the third over p*n values. So there are n*m * m*p * p*n = (nmp)^2 entries total.
    
    The schoolbook decomposition gives: sum_r U[a,r] V[b,r] W[c,r] where a=i*m+j, b=j'*p+k', c=k''*n+i''.
    
    This equals 1 iff there's a column (i,j,k) such that i*m+j = a, j*p+k = b, k*n+i = c.
    That means i = a//m, j = a%m, j' = b//p, k' = b%p, k'' = c//n, i'' = c%n.
    And we need j = j', k = k', i = i''. So the sum is 1 iff j=j', k=k', i=i'', else 0.
    
    So T[a,b,c] = 1 if (a%m == b//p) and (b%p == c//n) and (a//m == c%n), else 0.
    
    This IS the matrix multiplication tensor. 
    
    So to verify: for all a in [0,n*m), b in [0,m*p), c in [0,p*n):
    sum_r U[a,r] V[b,r] W[c,r] should equal 1 if (a%m == b//p and b%p == c//n and a//m == c%n) else 0.
    """
    nm = n * m
    mp = m * p
    pn = p * n
    # Build expected tensor
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Compute actual tensor from decomposition
    # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
    # This is like a 3-way product. We can compute it as:
    # For each r, outer product of U[:,r], V[:,r], W[:,r]
    # Sum over r.
    # Efficient: reshape and use einsum
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    rank = U_arr.shape[1]
    
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Using einsum: 'ar,br,cr->abc'
    T = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    max_err = np.max(np.abs(T - expected))
    return max_err < tol


def _try_random_rank(n, m, p, rank, num_iters=100, seed=42):
    """Try to find a rank-r decomposition using random initialization and gradient descent."""
    rng = np.random.RandomState(seed)
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build expected tensor
    expected = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i*m + j, j*p + k, k*n + i] = 1.0
    
    best_err = float('inf')
    best_U = best_V = best_W = None
    
    for it in range(num_iters):
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        # Gradient descent on ||T - sum_r u_r v_r w_r||_F^2
        lr = 0.01
        for step in range(200):
            # T = einsum('ar,br,cr->abc', U, V, W)
            T = np.einsum('ar,br,cr->abc', U, V, W)
            err = T - expected
            loss = np.sum(err**2)
            
            if loss < best_err:
                best_err = loss
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
            
            if loss < 1e-14:
                break
            
            # Gradients
            # dL/dU = 2 * einsum('abc,br,cr->ar', err, V, W)
            # dL/dV = 2 * einsum('abc,ar,cr->br', err, U, W)
            # dL/dW = 2 * einsum('abc,ar,br->cr', err, U, V)
            dU = 2 * np.einsum('abc,br,cr->ar', err, V, W)
            dV = 2 * np.einsum('abc,ar,cr->br', err, U, W)
            dW = 2 * np.einsum('abc,ar,br->cr', err, U, V)
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
    
    if best_err < 1e-9:
        return best_U, best_V, best_W, best_err
    return None, None, None, best_err


def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS
    nm = n * m  # 20
    mp = m * p  # 25
    pn = p * n  # 20
    
    # Start with schoolbook
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    best_rank = n * m * p  # 100
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try to find lower rank decompositions
    # Theoretical lower bound: max(n*m, m*p, p*n) = max(20, 25, 20) = 25
    # But actually for matrix multiplication, the rank is at least max(n*m, m*p, n*p) = 25
    # Known bounds: rank >= max(n*m, m*p, n*p), and for n=4,m=5,p=5, 
    # the rank is known to be at most 100 (schoolbook) and at least 25.
    # Actually, the rank of matrix multiplication M_{n,m,p} is at least max(n*m, m*p, n*p)
    # and at most n*m*p.
    # For 4x5x5: lower bound 25, upper bound 100.
    
    # Try various ranks from low to high
    for rank in range(25, 100):
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        found = False
        for attempt in range(3):
            if time.time() > deadline - 30:
                break
            
            U, V, W, err = _try_random_rank(n, m, p, rank, num_iters=10, seed=rank*100+attempt)
            
            if U is not None and err < 1e-9:
                # Verify
                if _verify(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
                    found = True
                    break
        
        if found:
            # Try to go even lower
            break
    
    # Convert to lists for the return interface
    U_list = [[float(x) for x in row] for row in np.array(best_U).tolist()]
    V_list = [[float(x) for x in row] for row in np.array(best_V).tolist()]
    W_list = [[float(x) for x in row] for row in np.array(best_W).tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
