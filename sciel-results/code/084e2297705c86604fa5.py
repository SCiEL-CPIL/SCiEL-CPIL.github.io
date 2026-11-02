import numpy as np
import time
import itertools
import random

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W decompose the matrix multiplication tensor.
    
    For matrix C = A * B where A is n x m, B is m x p, C is n x p:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T has T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i (with matching indices)
    Otherwise T[a,b,c] = 0.
    
    A rank-R decomposition satisfies: sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c] for all a,b,c.
    """
    R = U.shape[1]
    
    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                s = 0.0
                for r in range(R):
                    s += U[a, r] * V[b, r] * W[c, r]
                if abs(s - 1.0) > tol:
                    return False
    
    # Check zero entries
    # A zero entry occurs when indices don't align: e.g., different j or i or k
    # We need to check all combinations where the indices don't form a valid triple
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                
                for j2 in range(m):
                    for k2 in range(p):
                        b = j2 * p + k2
                        
                        for i2 in range(n):
                            for k3 in range(p):
                                c = k3 * n + i2
                                
                                # Skip if this is actually a non-zero entry
                                if j2 == j and i2 == i and k3 == k:
                                    continue
                                    
                                s = 0.0
                                for r in range(R):
                                    s += U[a, r] * V[b, r] * W[c, r]
                                if abs(s) > tol:
                                    return False
    
    return True

def _build_schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    return U, V, W

def _build_block_decomposition(U1, V1, W1, U2, V2, W2, n, m, p1, p2):
    """Build a 3x3 x 3x(p1+p2) decomposition from two 3x3 x 3x3 decompositions."""
    # U1, V1, W1 are for 3x3 x 3x3 (p1=3)
    # U2, V2, W2 are for 3x3 x 3x3 (p2=3)
    # Combined: 3x3 x 3x6
    # a_dim = n*m = 9, b_dim = m*p = 3*6 = 18, c_dim = p*n = 6*3 = 18
    
    a_dim = n * m
    b_dim = m * (p1 + p2)
    c_dim = (p1 + p2) * n
    
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((a_dim, R))
    V = np.zeros((b_dim, R))
    W = np.zeros((c_dim, R))
    
    # First block: k in [0, p1)
    # a = i*m + j (same for both)
    # b = j*p + k where p = p1+p2 = 6
    #   For first block: b = j*6 + k where k in [0,3)
    #   U1 uses b1 = j*p1 + k = j*3 + k
    #   So V[b, r] = V1[b1, r] for r in [0, R1)
    # c = k*n + i where k in [0, p1)
    #   W[c, r] = W1[c1, r] where c1 = k*n + i
    
    for i in range(n):
        for j in range(m):
            a = i * m + j
            U[a, :R1] = U1[a, :]
            
            for k in range(p1):
                b = j * (p1 + p2) + k
                b1 = j * p1 + k
                V[b, :R1] = V1[b1, :]
                
                c = k * n + i
                W[c, :R1] = W1[c, :]
    
    # Second block: k in [p1, p1+p2)
    # a = i*m + j (same)
    # b = j*6 + k where k in [3,6)
    #   U2 uses b2 = j*p2 + (k-p1) = j*3 + (k-3)
    #   So V[b, R1+r] = V2[b2, r] for r in [0, R2)
    # c = k*n + i where k in [3,6)
    #   W[c, R1+r] = W2[c2, r] where c2 = (k-p1)*n + i
    
    for i in range(n):
        for j in range(m):
            a = i * m + j
            U[a, R1:] = U2[a, :]
            
            for k in range(p1, p1 + p2):
                b = j * (p1 + p2) + k
                k2 = k - p1
                b2 = j * p2 + k2
                V[b, R1:] = V2[b2, :]
                
                c = k * n + i
                c2 = k2 * n + i
                W[c, R1:] = W2[c2, :]
    
    return U, V, W

def _als_333(target_rank, seed=0, max_iter=500, tol=1e-10):
    """Try to find a rank-target_rank decomposition for 3x3 x 3x3 matrix multiplication
    using Alternating Least Squares."""
    n = m = p = 3
    a_dim = n * m  # 9
    b_dim = m * p  # 9
    c_dim = p * n  # 9
    
    rng = np.random.RandomState(seed)
    R = target_rank
    
    # Initialize
    U = rng.randn(a_dim, R) * 0.1
    V = rng.randn(b_dim, R) * 0.1
    W = rng.randn(c_dim, R) * 0.1
    
    # Precompute the non-zero indices
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    def compute_error():
        err = 0.0
        for a, b, c in nonzero:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err += (s - 1.0) ** 2
        # Sample some zero entries
        for _ in range(200):
            a = rng.randint(0, a_dim)
            b = rng.randint(0, b_dim)
            c = rng.randint(0, c_dim)
            # Check if this is a non-zero entry
            is_nz = False
            for (a2, b2, c2) in nonzero:
                if a2 == a and b2 == b and c2 == c:
                    is_nz = True
                    break
            if is_nz:
                continue
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err += s ** 2
        return err
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        # Update W: for each c, solve linear system
        for c in range(c_dim):
            # Equations: sum_r (U[a,r] * V[b,r]) * W[c,r] = T[a,b,c]
            # for all (a,b) where T[a,b,c] != 0
            # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
            # So for fixed c = k*n+i:
            #   j = a - i*m, and b = j*p + k
            #   So for each j in [0,m): a = i*m+j, b = j*p+k
            #   T[a,b,c] = 1 if these are valid
            
            A_mat = np.zeros((m, R))
            rhs = np.zeros(m)
            for j in range(m):
                a = i * m + j
                b = j * p + k
                A_mat[j, :] = U[a, :] * V[b, :]
                rhs[j] = 1.0
            
            # Solve A_mat @ w = rhs for w = W[c, :]
            try:
                W[c, :], residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Update U: for each a, solve linear system
        for a in range(a_dim):
            # a = i*m + j
            # Equations: sum_r U[a,r] * (V[b,r] * W[c,r]) = T[a,b,c]
            # for all (b,c) where T[a,b,c] != 0
            # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
            # For fixed a = i*m+j:
            #   b = j*p+k, c = k*n+i
            #   So for each k in [0,p): b = j*p+k, c = k*n+i
            
            A_mat = np.zeros((p, R))
            rhs = np.zeros(p)
            for k in range(p):
                b = j * p + k
                c = k * n + i
                A_mat[k, :] = V[b, :] * W[c, :]
                rhs[k] = 1.0
            
            try:
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Update V: for each b, solve linear system
        for b in range(b_dim):
            # b = j*p + k
            # Equations: sum_r V[b,r] * (U[a,r] * W[c,r]) = T[a,b,c]
            # for all (a,c) where T[a,b,c] != 0
            # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
            # For fixed b = j*p+k:
            #   a = i*m+j, c = k*n+i
            #   So for each i in [0,n): a = i*m+j, c = k*n+i
            
            A_mat = np.zeros((n, R))
            rhs = np.zeros(n)
            for i in range(n):
                a = i * m + j
                c = k * n + i
                A_mat[i, :] = U[a, :] * W[c, :]
                rhs[i] = 1.0
            
            try:
                V[b, :], residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Compute error
        error = compute_error()
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    # Final verification
    if error < 1e-6:
        # Check all entries
        valid = True
        for a in range(a_dim):
            for b in range(b_dim):
                for c in range(c_dim):
                    is_nz = False
                    for (a2, b2, c2) in nonzero:
                        if a2 == a and b2 == b and c2 == c:
                            is_nz = True
                            break
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if is_nz:
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    else:
                        if abs(s) > 1e-6:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None

def _try_direct_336(target_rank, seed=0, max_iter=300, tol=1e-10):
    """Try to find a rank-target_rank decomposition for 3x3 x 3x6 directly using ALS."""
    n = m = 3
    p = 6
    a_dim = n * m  # 9
    b_dim = m * p  # 18
    c_dim = p * n  # 18
    
    rng = np.random.RandomState(seed)
    R = target_rank
    
    # Initialize
    U = rng.randn(a_dim, R) * 0.1
    V = rng.randn(b_dim, R) * 0.1
    W = rng.randn(c_dim, R) * 0.1
    
    # Precompute the non-zero indices
    nonzero = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                nonzero.append((a, b, c))
    
    def compute_error():
        err = 0.0
        for a, b, c in nonzero:
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err += (s - 1.0) ** 2
        # Sample some zero entries
        for _ in range(300):
            a = rng.randint(0, a_dim)
            b = rng.randint(0, b_dim)
            c = rng.randint(0, c_dim)
            is_nz = False
            for (a2, b2, c2) in nonzero:
                if a2 == a and b2 == b and c2 == c:
                    is_nz = True
                    break
            if is_nz:
                continue
            s = np.sum(U[a, :] * V[b, :] * W[c, :])
            err += s ** 2
        return err
    
    prev_error = float('inf')
    
    for it in range(max_iter):
        # Update W: for each c, solve linear system
        for c in range(c_dim):
            # c = k*n + i
            # T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
            # For fixed c = k*n+i:
            #   j = a - i*m, b = j*p + k
            #   So for each j in [0,m): a = i*m+j, b = j*p+k
            #   T[a,b,c] = 1 if these are valid
            
            A_mat = np.zeros((m, R))
            rhs = np.zeros(m)
            for j in range(m):
                a = i * m + j
                b = j * p + k
                A_mat[j, :] = U[a, :] * V[b, :]
                rhs[j] = 1.0
            
            try:
                W[c, :], residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Update U: for each a, solve linear system
        for a in range(a_dim):
            # a = i*m + j
            # For fixed a = i*m+j:
            #   b = j*p+k, c = k*n+i
            #   So for each k in [0,p): b = j*p+k, c = k*n+i
            
            A_mat = np.zeros((p, R))
            rhs = np.zeros(p)
            for k in range(p):
                b = j * p + k
                c = k * n + i
                A_mat[k, :] = V[b, :] * W[c, :]
                rhs[k] = 1.0
            
            try:
                U[a, :], residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Update V: for each b, solve linear system
        for b in range(b_dim):
            # b = j*p + k
            # For fixed b = j*p+k:
            #   a = i*m+j, c = k*n+i
            #   So for each i in [0,n): a = i*m+j, c = k*n+i
            
            A_mat = np.zeros((n, R))
            rhs = np.zeros(n)
            for i in range(n):
                a = i * m + j
                c = k * n + i
                A_mat[i, :] = U[a, :] * W[c, :]
                rhs[i] = 1.0
            
            try:
                V[b, :], residuals, rank, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
            except np.linalg.LinAlgError:
                pass
        
        # Compute error
        error = compute_error()
        
        if abs(prev_error - error) < tol:
            break
        prev_error = error
    
    # Final verification
    if error < 1e-6:
        valid = True
        for a in range(a_dim):
            for b in range(b_dim):
                for c in range(c_dim):
                    is_nz = False
                    for (a2, b2, c2) in nonzero:
                        if a2 == a and b2 == b and c2 == c:
                            is_nz = True
                            break
                    s = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if is_nz:
                        if abs(s - 1.0) > 1e-6:
                            valid = False
                            break
                    else:
                        if abs(s) > 1e-6:
                            valid = False
                            break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard: return best found before wall-clock limit
    deadline = time.time() + 120 - 10  # 110s safety margin
    
    # Schoolbook baseline: rank 54
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = n * m * p
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy 1: Try to find a 3x3 x 3x3 decomposition of rank < 27,
    # then use block construction to get 3x3 x 3x6 of rank 2*R333
    # If R333 = 26, combined rank = 52 < 54
    
    # Try various target ranks for 3x3 x 3x3
    for target_333 in [26, 25, 24, 23, 22, 21, 20]:
        if time.time() > deadline - 30:
            break
        
        result = _als_333(target_333, seed=target_333, max_iter=200)
        if result is not None:
            U1, V1, W1 = result
            # Use the same decomposition for both blocks
            U2, V2, W2 = U1.copy(), V1.copy(), W1.copy()
            
            U_comb, V_comb, W_comb = _build_block_decomposition(
                U1, V1, W1, U2, V2, W2, n, m, 3, 3)
            
            if _verify(U_comb, V_comb, W_comb, n, m, p, tol=1e-6):
                combined_rank = U_comb.shape[1]
                if combined_rank < best_rank:
                    best_rank = combined_rank
                    best_U, best_V, best_W = U_comb, V_comb, W_comb
                break
    
    # Strategy 2: Try direct ALS on 3x3 x 3x6
    for target_336 in [53, 52, 50, 48, 46, 45, 44, 42, 40]:
        if time.time() > deadline - 15:
            break
        
        result = _try_direct_336(target_336, seed=target_336, max_iter=150)
        if result is not None:
            U_try, V_try, W_try = result
            if _verify(U_try, V_try, W_try, n, m, p, tol=1e-6):
                if target_336 < best_rank:
                    best_rank = target_336
                    best_U, best_V, best_W = U_try, V_try, W_try
                break
    
    # Return the best result found
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
