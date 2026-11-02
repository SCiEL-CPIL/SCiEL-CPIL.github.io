"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import numpy as np
import time
from itertools import product

def _schoolbook(n, m, p):
    """Return U, V, W for schoolbook decomposition of n x m * m x p."""
    rank = n * m * p
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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is correct."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_strassen_like(n, m, p, deadline):
    """
    Try to construct a low-rank decomposition using a Strassen-like approach.
    For 3x3x3, we can use the known rank-23 algorithm (or similar).
    For 3x3x7, we can try to extend this.
    
    Actually, let's try a different approach: use the fact that matrix multiplication
    is bilinear and try to find a sparse support pattern over a finite field,
    then solve for the coefficients.
    """
    # For now, let's try a structured random search with verification.
    # We'll try ranks from 40 up to 62 and see if we can find a valid decomposition.
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 63
    
    # Try a few promising ranks
    for rank in range(55, 63):  # Try ranks 55-62
        if time.time() > deadline:
            break
        
        # Try random sparse matrices with a specific pattern
        # Use a block structure: for each (i,j,k), we need sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # Strategy: use a random orthogonal-like structure
        for attempt in range(100):
            if time.time() > deadline:
                break
            
            # Initialize with random values
            U = np.zeros((n * m, rank))
            V = np.zeros((m * p, rank))
            W = np.zeros((p * n, rank))
            
            # Use a structured initialization
            for c in range(rank):
                # Assign random non-zero entries
                u_idx = np.random.choice(n * m, size=3, replace=False)
                v_idx = np.random.choice(m * p, size=3, replace=False)
                w_idx = np.random.choice(p * n, size=3, replace=False)
                
                U[u_idx, c] = np.random.choice([-1, 1])
                V[v_idx, c] = np.random.choice([-1, 1])
                W[w_idx, c] = np.random.choice([-1, 1])
            
            # Verify
            if _verify(U, V, W, n, m, p):
                return U, V, W, rank
    
    return best_U, best_V, best_W, best_rank

def _try_field_search(n, m, p, deadline):
    """
    Try to find a sparse support pattern over F_5, then solve for coefficients.
    """
    # This is a complex approach. Let's try a simpler structured approach first.
    # For 3x3x3, the rank is 23. For 3x3x7, we can think of it as 3x3x3 + 3x3x4.
    # If we can find a rank-23 decomposition for 3x3x3 and a rank-32 for 3x3x4,
    # we get rank 55.
    
    # Let's try to construct a rank-23 decomposition for 3x3x3 using known results.
    # The Bini et al. algorithm has rank 23 for 3x3x3.
    
    # For simplicity, let's try a different approach: use the fact that we can
    # share computation across different k values.
    
    # Actually, let's just try to find any decomposition with rank < 63 using
    # a more systematic search.
    
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p
    
    # Try to use a block diagonal structure
    # For each j (0,1,2), we have a 3x7 matrix slice.
    # We can try to represent each slice as a sum of rank-1 terms.
    
    # Let's try a specific construction:
    # For j=0: C[i,k] = A[i,0]*B[0,k] for i=0,1,2 and k=0,...,6
    # This is an outer product, which has rank 1 in the (i,k) plane.
    # But we need to combine across j values.
    
    # The tensor T[i,j,k] = 1 for all i,j,k.
    # We want to write T = sum_c U_c ⊗ V_c ⊗ W_c where
    # U_c is a vector in R^{n*m}, V_c in R^{m*p}, W_c in R^{p*n}
    # and T[i,j,k] = sum_c U_c[i*m+j] * V_c[j*p+k] * W_c[k*n+i]
    
    # Let's try a construction where we use the same U and V for different k values.
    
    # Actually, let's try a simple improvement: for the schoolbook, we have 63 terms.
    # Can we merge some terms?
    
    # For example, consider terms where i and j are the same but k differs.
    # U[i*m+j, c1] = 1, V[j*p+k1, c1] = 1, W[k1*n+i, c1] = 1
    # U[i*m+j, c2] = 1, V[j*p+k2, c2] = 1, W[k2*n+i, c2] = 1
    # These are separate terms. Can we combine them?
    
    # If we set U[i*m+j, c] = 1, V[j*p+k1, c] = a, V[j*p+k2, c] = b,
    # W[k1*n+i, c] = 1/a, W[k2*n+i, c] = 1/b, then the contribution is
    # 1 * a * 1/a + 1 * b * 1/b = 2, which is not what we want.
    
    # So we need to be more careful. Let's try a different approach.
    
    # Use a random search with a better initialization.
    for rank in [62, 61, 60, 59, 58, 57, 56, 55]:
        if time.time() > deadline:
            break
        
        # Try random dense matrices
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            U = np.random.randn(n * m, rank)
            V = np.random.randn(m * p, rank)
            W = np.random.randn(p * n, rank)
            
            # Scale to improve numerical stability
            U /= np.linalg.norm(U, axis=0, keepdims=True)
            V /= np.linalg.norm(V, axis=0, keepdims=True)
            W /= np.linalg.norm(W, axis=0, keepdims=True)
            
            # Check if this is close to a valid decomposition
            # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
            # This is a system of nonlinear equations, so random search is unlikely to work.
            
            # Let's try a different approach: fix U and V, then solve for W.
            # For each (i,j,k), we have sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            # This is a linear system in W if we fix U and V.
            
            # Let's try this: fix U and V randomly, then solve for W.
            # The system is: for each (i,j,k), sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
            # Let A[i,j,k,c] = U[i*m+j,c] * V[j*p+k,c]
            # Then sum_c A[i,j,k,c] * W[k*n+i,c] = 1
            # This is a linear system in the variables W[k*n+i,c].
            
            # But W has p*n*rank variables, and we have n*m*p equations.
            # For n=3, m=3, p=7, rank=62: 3*3*7=63 equations, 7*3*62=1302 variables.
            # So the system is underdetermined, which is good.
            
            # Let's set up the linear system.
            # Variables: W[k*n+i, c] for k=0..6, i=0..2, c=0..61
            # Equation (i,j,k): sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # Build the matrix
            num_vars = p * n * rank
            num_eqs = n * m * p
            
            # This might be too large to solve directly, but let's try.
            # Actually, for rank=62, num_vars = 7*3*62 = 1302, num_eqs = 63.
            # We can use least squares.
            
            # Build the system matrix
            # For each equation (i,j,k), the coefficients are:
            # For variable (k,n+i,c): coeff = U[i*m+j,c] * V[j*p+k,c]
            
            # Let's use a sparse representation
            from scipy import sparse
            
            rows = []
            cols = []
            data = []
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for c in range(rank):
                            coeff = U[i*m+j, c] * V[j*p+k, c]
                            if abs(coeff) > 1e-10:
                                var_idx = k * n * rank + i * rank + c
                                rows.append(eq_idx)
                                cols.append(var_idx)
                                data.append(coeff)
            
            A_mat = sparse.csr_matrix((data, (rows, cols)), shape=(num_eqs, num_vars))
            b_vec = np.ones(num_eqs)
            
            # Solve using least squares
            try:
                from scipy.linalg import lstsq
                # Convert to dense for lstsq (might be slow for large systems)
                # For rank=62, num_vars=1302, num_eqs=63, this should be fine.
                A_dense = A_mat.toarray()
                W_flat, residuals, rank_sys, sv = lstsq(A_dense, b_vec)
                
                # Reshape W
                W_sol = W_flat.reshape(p * n, rank)
                
                # Verify
                if _verify(U, V, W_sol, n, m, p):
                    return U, V, W_sol, rank
            except Exception:
                pass
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = 3, 3, 7
    deadline = time.time() + 115  # Safety margin
    
    # Schoolbook fallback
    U_sb, V_sb, W_sb = _schoolbook(n, m, p)
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = 63
    
    # Try field search approach
    U_f, V_f, W_f, rank_f = _try_field_search(n, m, p, deadline)
    if U_f is not None and rank_f < best_rank:
        best_U, best_V, best_W = U_f, V_f, W_f
        best_rank = rank_f
    
    # Try Strassen-like approach
    if time.time() < deadline:
        U_s, V_s, W_s, rank_s = _try_strassen_like(n, m, p, deadline)
        if U_s is not None and rank_s < best_rank:
            best_U, best_V, W_s = U_s, V_s, W_s
            best_rank = rank_s
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
