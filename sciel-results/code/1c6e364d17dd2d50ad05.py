import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

def _construct_tensor(n, m, p):
    """
    Constructs the matrix multiplication tensor T[i, j, k] for A(n,m) * B(m,p) -> C(n,p).
    The tensor T is defined such that C[i, k] = sum_j T[i, j, k] * A[i, j] * B[j, k].
    Actually, the standard decomposition is:
    C[i, k] = sum_{a,b,c} U[i, a] * V[b, a] * W[k, a] * A[i, b] * B[c, k]
    
    Wait, the interface is:
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    
    The decomposition is:
    C[i, k] = sum_{r} W[k*n+i, r] * ( sum_{j} U[i*m+j, r] * A[i, j] ) * ( sum_{j'} V[j'*p+k, r] * B[j', k] )
    
    Let's check the indices.
    U has rows indexed by (i, j) where i in [0,n), j in [0,m).
    V has rows indexed by (j, k) where j in [0,m), k in [0,p).
    W has rows indexed by (k, i) where k in [0,p), i in [0,n). Note the task says W[k*n+i, r].
    
    The product is:
    C[i, k] = sum_{r} W[k*n+i, r] * ( sum_{j} U[i*m+j, r] * A[i, j] ) * ( sum_{j} V[j*p+k, r] * B[j, k] )
    
    This matches the structure if we view the tensor T as having modes:
    Mode 1: (i, j) size n*m
    Mode 2: (j, k) size m*p
    Mode 3: (k, i) size p*n
    
    And the equation is C[i, k] = sum_{r} W_{k,i}^r * (U^r A)_i * (V^r B)_k ?
    No, let's expand:
    Term = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    
    Let's verify with schoolbook.
    For a specific (i,j,k), we want a term that contributes to C[i,k] proportional to A[i,j]B[j,k].
    In schoolbook, rank = n*m*p.
    For each (i,j,k), we have a rank-1 component.
    U[i*m+j, r_{ijk}] = 1
    V[j*p+k, r_{ijk}] = 1
    W[k*n+i, r_{ijk}] = 1
    All other entries in that column are 0.
    
    Then sum_r W[k*n+i, r] * (sum_j' U[i*m+j', r] A[i,j']) * (sum_j'' V[j''*p+k, r] B[j'',k])
    Only r = r_{ijk} is non-zero for W[k*n+i, r].
    For that r, U[i*m+j', r] is 1 only if j'=j. So the first sum becomes A[i,j].
    For that r, V[j''*p+k, r] is 1 only if j''=j. So the second sum becomes B[j,k].
    Result: W[k*n+i, r_{ijk}] * A[i,j] * B[j,k] = 1 * A[i,j] * B[j,k].
    This is correct.
    
    So the tensor T is a 3-way tensor of size (n*m, m*p, p*n).
    T[i*m+j, j'*p+k, k'*n+i'] = 1 if i==i' and j==j' and k==k', else 0.
    Wait, let's re-read the contraction.
    C[i, k] = sum_{r} W[k*n+i, r] * (sum_{j} U[i*m+j, r] A[i,j]) * (sum_{j} V[j*p+k, r] B[j,k])
    
    Let's define the tensor T such that:
    T[a, b, c] where a is index for U (i,j), b is index for V (j',k'), c is index for W (k'', i'').
    The contraction is sum_{r} T_{a,b,c}^{(r)} ... no, T is the tensor we are decomposing.
    We want T to be such that the decomposition recovers the multiplication.
    
    Actually, it's easier to think of T as the tensor of the multiplication map.
    The multiplication map is linear in A and B.
    C = A B.
    We can view this as a bilinear form.
    
    Let's construct the tensor T explicitly.
    T is of shape (n*m, m*p, p*n).
    T[i*m+j, j*p+k, k*n+i] = 1.
    All other entries are 0.
    
    Let's check the decomposition of this T.
    We want T = sum_r U[:, r] outer V[:, r] outer W[:, r].
    So T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    
    If we use schoolbook:
    For each (i,j,k), let r correspond to (i,j,k).
    U[i*m+j, r_{ijk}] = 1
    V[j*p+k, r_{ijk}] = 1
    W[k*n+i, r_{ijk}] = 1
    Then T[i*m+j, j*p+k, k*n+i] = sum_r ... = 1.
    And for any other combination of indices (a,b,c) not matching (i*m+j, j*p+k, k*n+i) with consistent i,j,k, the sum is 0.
    Is it possible for T[a,b,c] to be non-zero for inconsistent indices?
    a = i1*m + j1
    b = j2*p + k1
    c = k2*n + i2
    T[a,b,c] = 1 iff i1==i2 and j1==j2 and k1==k2.
    In the schoolbook decomposition, T[a,b,c] = sum_r U[a,r]V[b,r]W[c,r].
    U[a,r] is non-zero only if r corresponds to (i1,j1).
    V[b,r] is non-zero only if r corresponds to (j2,k1).
    W[c,r] is non-zero only if r corresponds to (k2,i2).
    For the product to be non-zero, r must be the same index, so (i1,j1) = (j2,k1) = (k2,i2) in terms of the triple (i,j,k).
    This implies i1=j2? No, the index r is just a label.
    r_{ijk} is unique for each triple (i,j,k).
    U[a, r_{ijk}] is 1 if a = i*m+j.
    V[b, r_{ijk}] is 1 if b = j*p+k.
    W[c, r_{ijk}] is 1 if c = k*n+i.
    So T[a,b,c] = 1 if there exists (i,j,k) such that a=i*m+j, b=j*p+k, c=k*n+i.
    This matches the definition T[i*m+j, j*p+k, k*n+i] = 1.
    
    So the task is to find the CP rank of the tensor T of size (n*m, m*p, p*n) defined by T[i*m+j, j*p+k, k*n+i] = 1.
    """
    n, m, p = n, m, p
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    return T

def _cp_decompose(T, max_rank, n_restarts=5, tol=1e-10, max_iter=100):
    """
    Attempts to find a CP decomposition of T with rank <= max_rank.
    Returns (U, V, W, rank) if successful, else None.
    """
    I, J, K = T.shape
    
    for restart in range(n_restarts):
        # Random initialization
        U = np.random.randn(I, max_rank)
        V = np.random.randn(J, max_rank)
        W = np.random.randn(K, max_rank)
        
        for it in range(max_iter):
            # Update U
            # U[:, r] = (T x2 V[:, r] x1 W[:, r]) / (V[:, r] * U[:, r] * W[:, r])? No.
            # T = sum_r U_r V_r W_r
            # T x2 V_r is a matrix of size I x K.
            # (T x2 V_r) x1 W_r is a vector of size I.
            # Denominator: sum_k W_r[k]^2 * sum_i U_r[i]^2 ? No.
            # Let's use the standard ALS update.
            # U[:, r] = (T x2 V[:, r] x1 W[:, r]) / (sum_k W[:, r][k]^2 * sum_i U[:, r][i]^2) -- No, this is for normalized.
            # Standard:
            # T_{ijk} = sum_r U_{ir} V_{jr} W_{kr}
            # Fix r.
            # U_{ir} = (sum_{j,k} T_{ijk} V_{jr} W_{kr}) / (sum_{j,k} V_{jr}^2 W_{kr}^2)
            
            # Efficient update using einsum or tensordot
            # For each r:
            # Numerator: np.einsum('ijk,jr,kr->ir', T, V, W)
            # Denominator: np.einsum('jr,kr->r', V, W) ** 2 ? No.
            # Denom_r = sum_j V[j,r]^2 * sum_k W[k,r]^2
            
            num_U = np.einsum('ijk,jr,kr->ir', T, V, W)
            den_U = np.einsum('jr,jr->r', V, V) * np.einsum('kr,kr->r', W, W)
            den_U = np.where(den_U == 0, 1e-10, den_U)
            U_new = num_U / den_U[:, np.newaxis]
            
            # Check convergence
            if np.allclose(U, U_new, rtol=tol, atol=tol):
                U = U_new
                break
            U = U_new
            
            # Update V
            num_V = np.einsum('ijk,ir,kr->jr', T, U, W)
            den_V = np.einsum('ir,ir->r', U, U) * np.einsum('kr,kr->r', W, W)
            den_V = np.where(den_V == 0, 1e-10, den_V)
            V_new = num_V / den_V[:, np.newaxis]
            
            if np.allclose(V, V_new, rtol=tol, atol=tol):
                V = V_new
                break
            V = V_new
            
            # Update W
            num_W = np.einsum('ijk,ir,jr->kr', T, U, V)
            den_W = np.einsum('ir,ir->r', U, U) * np.einsum('jr,jr->r', V, V)
            den_W = np.where(den_W == 0, 1e-10, den_W)
            W_new = num_W / den_W[:, np.newaxis]
            
            if np.allclose(W, W_new, rtol=tol, atol=tol):
                W = W_new
                break
            W = W_new
        
        # Verify the decomposition
        # T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        # if np.allclose(T, T_recon, rtol=1e-5, atol=1e-5):
        #     # Try to reduce rank by zeroing out small components
        #     pass
        # 
        # For now, just check if the reconstruction error is small.
        T_recon = np.einsum('ir,jr,kr->ijk', U, V, W)
        if np.allclose(T, T_recon, rtol=1e-6, atol=1e-6):
            # Try to prune rank
            # Sort by magnitude of components
            # This is a heuristic.
            # For now, return max_rank
            return U, V, W, max_rank
            
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _construct_tensor(n, m, p)
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    # Start with a reasonable upper bound
    # Known bounds for matmul:
    # Strassen for 2x2x2 is 7.
    # For 4x5x5, we can try ranks lower than 100.
    # Let's try a range of ranks.
    # The minimum rank is at least max(n*m, m*p, p*n) / something?
    # Lower bound: max(n*m, m*p, p*n) is not a lower bound for rank.
    # Lower bound is max(n*m, m*p, p*n) / (min dimension)? No.
    # A simple lower bound is max(n*m, m*p, p*n) / (n+m+p-2)? No.
    # Let's just search from a low rank up to schoolbook.
    
    # We can use the fact that the tensor is sparse.
    # Number of non-zeros is n*m*p = 100.
    # Rank is at least the number of non-zeros in any slice?
    # Slice i=0: size m*p x p*n.
    
    # Let's try ranks: 50, 60, 70, 80, 90, 100
    # Or better, start from a lower estimate.
    # For square matrices N x N x N, rank is ~ N^2.
    # Here 4x5x5.
    # Let's try 80, 90, 100.
    
    # Actually, let's try to find the minimal rank.
    # We'll iterate rank from 50 to 100.
    # If we find a solution at rank R, we can try R-1, etc.
    
    # To save time, we'll just try a few specific ranks.
    # If we fail to find a lower rank, we return the schoolbook one.
    
    ranks_to_try = [80, 90, 100]
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        # Try to find decomposition of rank r
        res = _cp_decompose(T, r, n_restarts=3, tol=1e-10, max_iter=50)
        if res is not None:
            U, V, W, rank = res
            if rank < best_rank:
                best_U, best_V, best_W, best_rank = U, V, W, rank
                # If we found a solution, we can try lower ranks
                # But for now, let's just return the first one found
                # Or continue to find lower?
                # Given time constraints, let's try to find the lowest.
                # We'll break if we find rank 80? No, maybe 70 exists.
                # Let's just collect the best.
    
    # If no lower rank found, use schoolbook
    if best_U is None:
        # Construct schoolbook
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists of lists for the interface
    # The interface expects integer/half-integer entries.
    # Our CP decomposition produces floats.
    # We need to round them?
    # "exactly decompose" means the product must be exact.
    # If we use floats, it might not be exact.
    # The schoolbook uses integers.
    # If we find a float decomposition, we should check if it can be represented exactly.
    # Or, we should only accept decompositions that are "exact" in the sense of the evaluator.
    # The evaluator likely checks:
    # C = (U A) * (V B) * W? No.
    # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] A[i,j]) * (sum_j V[j*p+k, r] B[j,k])
    # This must hold for all A, B.
    # This is equivalent to T = sum_r U_r V_r W_r.
    # If we use floats, there will be rounding errors.
    # So we must find a decomposition with rational (ideally integer) entries.
    # CP decomposition via SVD/ALS does not guarantee rational entries.
    # 
    # Strategy adjustment:
    # The CP decomposition found by ALS is a *real* decomposition.
    # If the tensor has a low-rank *integer* decomposition, ALS might find it (up to permutation and scaling).
    # However, ALS converges to local minima.
    # And the entries might be irrational.
    # 
    # Given the risk of inexactness, and the time limit, 
    # and the fact that no known sub-schoolbook integer decomposition is easily found by simple ALS,
    # it is safer to return the schoolbook decomposition if we cannot verify exactness of the float one.
    # 
    # But wait, the task says "integer/half-integer entries".
    # This implies that such a decomposition is expected or at least allowed.
    # If ALS finds a solution with entries like 0.5, 1.0, -1.0, it's valid.
    # If it finds 0.33333, it's invalid.
    # 
    # Let's add a check: after ALS, check if entries are close to half-integers.
    # If yes, round them and re-verify.
    
    def _is_half_integer(x, tol=1e-6):
        return np.allclose(x, np.round(2*x)/2, atol=tol)
    
    def _verify_exact(U, V, W, T):
        # Check if U, V, W are half-integers
        if not (_is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W)):
            return False
        # Round them
        U_r = np.round(2*U)/2
        V_r = np.round(2*V)/2
        W_r = np.round(2*W)/2
        # Verify T
        T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
        return np.allclose(T, T_recon, atol=1e-9)
    
    # Re-evaluate the results from ALS
    # We didn't store the ALS results in a way that allows re-checking easily in the loop above.
    # Let's modify the loop to check exactness.
    
    # Reset best
    best_U = None
    best_V = None
    best_W = None
    best_rank = schoolbook_rank
    
    for r in ranks_to_try:
        if time.time() > deadline:
            break
        
        res = _cp_decompose(T, r, n_restarts=5, tol=1e-12, max_iter=100)
        if res is not None:
            U, V, W, rank = res
            if _verify_exact(U, V, W, T):
                if rank < best_rank:
                    best_U, best_V, best_W, best_rank = U, V, W, rank
                    # If we found a valid exact decomposition, we can stop or try lower.
                    # Let's try lower ranks if we have time.
    
    if best_U is None:
        # Fallback to schoolbook
        U = np.zeros((n*m, schoolbook_rank))
        V = np.zeros((m*p, schoolbook_rank))
        W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        best_U, best_V, best_W = U, V, W
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

if __name__ == "__main__":
    # Quick test
    res = construct()
    print("Rank:", len(res["U"][0]))
