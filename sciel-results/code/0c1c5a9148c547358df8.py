import time
import numpy as np
from itertools import product

_START_TIME = time.time()
_DEADLINE = 145  # 5s safety margin before 150s kill

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Check if U,V,W form a valid decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # Construct target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
                
    # Compute reconstructed tensor
    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check difference
    if np.max(np.abs(T_recon - T_target)) > tol:
        return False
    return True

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

def _try_rank_23(n, m, p):
    """
    Attempt to find a rank 23 decomposition using structured perturbation
    of the schoolbook solution. The idea is that for 2x3x4 matrix multiplication,
    there may exist a rank 23 decomposition by cleverly combining terms.
    
    We use a parameterized family of decompositions and search for parameters
    that satisfy the tensor equality constraint.
    """
    if time.time() > _DEADLINE - 15:
        return None
    
    # For 2x3x4, the schoolbook has 24 terms. We try to find a rank 23 decomposition
    # by introducing a structured perturbation.
    
    # The matrix multiplication tensor T[i,j,k] = 1 if the indices are compatible
    # For 2x3x4: i in {0,1}, j in {0,1,2}, k in {0,1,2,3}
    # T[a,b,c] where a=i*m+j, b=j*p+k, c=k*n+i
    
    # Strategy: Start from schoolbook and try to merge two terms by introducing
    # a parameter that cancels out in the tensor reconstruction.
    
    # We'll try a family of decompositions with 23 terms where one term is
    # a linear combination that effectively replaces two schoolbook terms.
    
    # Let's try: take the schoolbook and modify it so that two of the basis vectors
    # in U, V, W are adjusted by a parameter t, and we see if for some t the
    # extra terms cancel.
    
    # Actually, let's try a different approach: use the fact that the border rank
    # might be lower. We'll set up a system of equations for a rank-23 decomposition.
    
    # For rank 23, we need U (6x23), V (12x23), W (8x23) such that
    # sum_r U[a,r]*V[b,r]*W[c,r] = T[a,b,c]
    
    # This is a nonlinear system. Let's try a structured approach.
    
    # One known technique: if we can find vectors u, v, w such that
    # u ⊗ v ⊗ w is in the span of the tensor, we can potentially reduce rank.
    
    # Let's try a simpler heuristic: randomly perturb the schoolbook and use
    # least squares to find if a rank-23 approximation exists, then check exactness.
    
    n_terms = n * m * p  # 24 for schoolbook
    target_rank = 23
    
    # Build target tensor
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0
    
    # Try multiple random initializations for rank-23 decomposition
    rng = np.random.default_rng(42)
    
    max_attempts = 50
    for attempt in range(max_attempts):
        if time.time() > _DEADLINE - 5:
            return None
        
        # Initialize U, V, W with random values
        U = rng.standard_normal((n*m, target_rank))
        V = rng.standard_normal((m*p, target_rank))
        W = rng.standard_normal((p*n, target_rank))
        
        # Use ALS (Alternating Least Squares) to minimize ||T_target - sum_r u_r ⊗ v_r ⊗ w_r||
        # This is a rank-constrained least squares problem.
        
        # ALS: fix V and W, solve for U; fix U and W, solve for V; fix U and V, solve for W
        for iteration in range(50):
            if time.time() > _DEADLINE - 5:
                return None
            
            # Solve for U: for each a, U[a,:] should satisfy
            # sum_r U[a,r] * V[b,r] * W[c,r] ≈ T[a,b,c]
            # Reshape: for fixed a, we have a linear system in U[a,:]
            # Let M_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # This is U[a,:] * (V^T * W) but with element-wise product...
            
            # Actually, M_a = U[a,:] @ (V * W.T) where * is element-wise
            # No, let's think more carefully.
            # M_a[b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
            # = U[a,:] dot (V[b,:] * W[c,:]) element-wise
            
            # For fixed a, define Z_a[b,c] = V[b,:] * W[c,:] (element-wise, shape (m*p, p*n, rank))
            # Then M_a[b,c] = U[a,:] @ Z_a[b,c,:]
            # This is a linear system: U[a,:] @ Z_a[b,c,:] = T[a,b,c] for all b,c
            
            # Let's build the linear system properly.
            # For each a, we want to find U[a,:] such that:
            # For all b,c: sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
            
            # Create matrix A_a of shape (m*p*p*n, rank) where
            # A_a[b*p*n+c, r] = V[b,r] * W[c,r]
            # And vector t_a of shape (m*p*p*n,) where t_a[b*p*n+c] = T[a,b,c]
            # Then solve A_a @ U[a,:] = t_a
            
            A = np.zeros((m*p * p*n, target_rank))
            t = T_target[a].flatten()
            
            for b in range(m*p):
                for c in range(p*n):
                    idx = b * p*n + c
                    A[idx, :] = V[b, :] * W[c, :]
            
            # Solve least squares
            U[a, :], residuals, rank_a, sv = np.linalg.lstsq(A, t, rcond=None)
            
            # Similarly for V
            for b in range(m*p):
                A_b = np.zeros((n*m * p*n, target_rank))
                t_b = T_target[:, b, :].flatten()
                for a in range(n*m):
                    for c in range(p*n):
                        idx = a * p*n + c
                        A_b[idx, :] = U[a, :] * W[c, :]
                V[b, :], residuals, rank_b, sv = np.linalg.lstsq(A_b, t_b, rcond=None)
            
            # Similarly for W
            for c in range(p*n):
                A_c = np.zeros((n*m * m*p, target_rank))
                t_c = T_target[:, :, c].flatten()
                for a in range(n*m):
                    for b in range(m*p):
                        idx = a * m*p + b
                        A_c[idx, :] = U[a, :] * V[b, :]
                W[c, :], residuals, rank_c, sv = np.linalg.lstsq(A_c, t_c, rcond=None)
        
        # Check if the solution is exact
        if _verify(U, V, W, n, m, p, tol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return None

def construct():
    n, m, p = 2, 3, 4
    
    # Try rank 23
    res = _try_rank_23(n, m, p)
    if res is not None:
        return res
        
    # Fallback: Schoolbook
    return _schoolbook(n, m, p)
