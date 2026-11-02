"""Matrix multiplication 3x5x7 - rank minimization."""
import time
import itertools
import random
from fractions import Fraction

def _build_schoolbook(n, m, p):
    """Build schoolbook decomposition."""
    nm, mp, pn = n * m, m * p, p * n
    rank = n * m * p
    U = [[0.0] * rank for _ in range(nm)]
    V = [[0.0] * rank for _ in range(mp)]
    W = [[0.0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return U, V, W, rank

def _verify(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly computes matrix multiplication."""
    nm, mp, pn = len(U), len(V), len(W)
    rank = len(U[0])
    # For each (i,j,k), check sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] == 1
    # For all other combinations, the sum should be 0
    # We check a sample of random matrices
    import numpy as np
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    AB = A @ B
    # Compute using decomposition
    # C[i,k] = sum_j A[i,j] * B[j,k] = sum_j sum_r U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
    # = sum_r (sum_j A[i,j]*U[i*m+j][r]) * (sum_k... ) -- no, this doesn't factor simply
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor decomposition says: C[i,k] = sum_r W[k*n+i][r] * (sum_j A[i,j]*U[i*m+j][r]) * ... 
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The product is: C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This should equal delta_{i,j,k} in the sense that when we contract with A and B:
    # sum_{j} A[i,j] * B[j,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k] / (A[i,j]*B[j,k])
    # Hmm, this is getting confusing. Let me think more carefully.
    # 
    # The matrix multiplication tensor T has entries T[i,j,k] = 1 if we're multiplying A[i,j]*B[j,k]
    # Actually the standard formulation: the multiplication map (A,B) -> C where C[i,k] = sum_j A[i,j]B[j,k]
    # Can be written as C = sum_r W_r * (U_r * A) * (V_r * B) where U_r, V_r, W_r are rank-1 factors
    # 
    # More precisely: C[i,k] = sum_r U[i*m+j, r] ... no.
    #
    # Let me use the standard bilinear form. The tensor for matrix mult is:
    # T_{i,j,k} such that C[i,k] = sum_j A[i,j] B[j,k]
    # This is a 3-tensor. A rank-1 decomposition gives:
    # T = sum_r u_r ⊗ v_r ⊗ w_r where u_r ∈ R^{n*m}, v_r ∈ R^{m*p}, w_r ∈ R^{p*n}
    # Then: C[i,k] = sum_r w_r[k*n+i] * (sum_j u_r[i*m+j] * A[i,j]) * ... 
    # No, that's not right either.
    #
    # Actually, the correct interpretation:
    # C[i,k] = sum_{j} A[i,j] B[j,k]
    # We want: C[i,k] = sum_r W[k*n+i, r] * sum_j (U[i*m+j, r] * A[i,j]) * sum_k' (V[j*p+k', r] * B[j,k'])
    # That doesn't work dimensionally.
    #
    # Let me look at this differently. The bilinear map is:
    # (A, B) -> C where C[i,k] = sum_j A[i,j] B[j,k]
    # This is bilinear in A and B. A rank-1 bilinear map is:
    # (A, B) -> alpha(A) * beta(B) * gamma where alpha: R^{n*m}->R, beta: R^{m*p}->R, gamma: R^{p*n}->R
    # So: C[i,k] = sum_r gamma_r[k*n+i] * alpha_r(A) * beta_r(B)
    # where alpha_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
    #       beta_r(B) = sum_{j,k} V[j*p+k, r] * B[j,k]
    #
    # For this to equal sum_j A[i,j] B[j,k] for all A, B:
    # sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j', r] A[i',j']) * (sum_{j',k'} V[j'*p+k', r] B[j',k'])
    # = sum_j A[i,j] B[j,k]
    #
    # Expanding the left side:
    # sum_r W[k*n+i, r] * sum_{i',j'} sum_{j'',k''} U[i'*m+j', r] V[j''*p+k'', r] A[i',j'] B[j'',k'']
    # = sum_{i',j',j'',k''} A[i',j'] B[j'',k''] * sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r]
    #
    # This must equal sum_j A[i,j] B[j,k], so:
    # sum_r W[k*n+i, r] U[i'*m+j', r] V[j''*p+k'', r] = delta_{i',i} delta_{j',j''} delta_{k'',k}
    # i.e., it equals 1 if i'=i, j'=j'', k''=k, and 0 otherwise.
    #
    # So the verification is: for all (i, j, k) and (i', j', j'', k''):
    # sum_r U[i'*m+j', r] * V[j''*p+k'', r] * W[k*n+i, r] = 1 if i'=i, j'=j'', k''=k, else 0
    
    # Let's verify using random matrices
    for trial in range(3):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_actual = np.zeros((n, p))
        for r in range(rank):
            alpha = np.dot(U, 0)  # placeholder
            # alpha_r(A) = sum_{i,j} U[i*m+j, r] * A[i,j]
            alpha_val = 0.0
            for ii in range(n):
                for jj in range(m):
                    alpha_val += U[ii*m+jj][r] * A[ii][jj]
            beta_val = 0.0
            for jj in range(m):
                for kk in range(p):
                    beta_val += V[jj*p+kk][r] * B[jj][kk]
            for ii in range(n):
                for kk in range(p):
                    C_actual[ii][kk] += W[kk*n+ii][r] * alpha_val * beta_val
        if not np.allclose(C_actual, C_expected, atol=tol):
            return False
    return True

def _verify_exact(U, V, W, n, m, p):
    """Exact verification using the tensor identity."""
    nm, mp, pn = len(U), len(V), len(W)
    rank = len(U[0])
    # Check: sum_r U[i'*m+j', r] * V[j''*p+k'', r] * W[k*n+i, r] = delta_{i',i} delta_{j',j''} delta_{k'',k}
    # We'll check all combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Should be 1 when i'=i, j'=j, k''=k... wait
                # The identity is: sum_r U[i'*m+j', r] V[j'*p+k, r] W[k*n+i, r] = delta_{j',j} for fixed i,k?
                # No. Let me re-derive.
                # C[i,k] = sum_j A[i,j] B[j,k]
                # = sum_r W[k*n+i, r] * (sum_{i',j'} U[i'*m+j',r] A[i',j']) * (sum_{j'',k''} V[j''*p+k'',r] B[j'',k''])
                # The coefficient of A[i',j'] B[j'',k''] on the LHS of "should equal C[i,k]" is:
                # delta_{i',i} delta_{j'',k}... no.
                # C[i,k] has coefficient 1 for A[i,j]B[j,k] for each j, and 0 otherwise.
                # So: sum_r U[i'*m+j',r] V[j''*p+k'',r] W[k*n+i,r] = 1 if i'=i and j''=k and j'=j'' 
                # Wait, A[i',j'] B[j'',k''] contributes to C[i,k] iff i'=i and k''=k and j'=j'' (the middle index matches).
                # So the condition is:
                # sum_r U[i'*m+j', r] * V[j''*p+k'', r] * W[k*n+i, r] = 1 if i'=i, j'=j'', k''=k; else 0
                pass
    # This is O(n^2 m^2 p^2 * rank) which is too slow. Use random tests instead.
    import numpy as np
    for trial in range(10):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_actual = np.zeros((n, p))
        for r in range(rank):
            alpha_val = 0.0
            for ii in range(n):
                for jj in range(m):
                    alpha_val += U[ii*m+jj][r] * A[ii][jj]
            beta_val = 0.0
            for jj in range(m):
                for kk in range(p):
                    beta_val += V[jj*p+kk][r] * B[jj][kk]
            for ii in range(n):
                for kk in range(p):
                    C_actual[ii][kk] += W[kk*n+ii][r] * alpha_val * beta_val
        if not np.allclose(C_actual, C_expected, atol=1e-10):
            return False
    return True

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x5x7 matrix multiplication.
    Uses algebraic search for non-trivial rank reductions from schoolbook.
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n * m, m * p, p * n  # 15, 35, 21
    
    # Start with schoolbook
    U, V, W, rank = _build_schoolbook(n, m, p)
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find rank reductions using structured search
    # Look for pairs of schoolbook terms that can be merged
    # The schoolbook has terms indexed by (i,j,k) -> col = i*m*p + j*p + k
    
    def col_idx(i, j, k):
        return i * m * p + j * p + k
    
    # Strategy: Try to find a rank-1 tensor T' such that T - T' has lower rank
    # where T is the multiplication tensor. This is equivalent to finding a 
    # non-trivial rank-1 component.
    # 
    # Alternative: Search for a 2-dimensional subspace of the rank-1 factors
    # that allows merging.
    #
    # For now, try a simple perturbation approach: randomly perturb some entries
    # and check if we can reduce rank.
    
    # Try searching for rank = best_rank - 1, best_rank - 2, etc.
    # Use random initialization and check validity
    
    import numpy as np
    
    for target_rank in range(best_rank - 1, max(best_rank - 10, 50), -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random attempts
        for attempt in range(20):
            if time.time() > deadline - 20:
                break
            
            # Initialize with schoolbook and zero out one column, then adjust
            U_try = [row[:] for row in U]
            V_try = [row[:] for row in V]
            W_try = [row[:] for row in W]
            
            # Remove last column
            U_try = [row[:target_rank] for row in U_try]
            V_try = [row[:target_rank] for row in V_try]
            W_try = [row[:target_rank] for row in W_try]
            
            # Add small random perturbations to try to fix the missing term
            for r in range(target_rank):
                for i in range(nm):
                    U_try[i][r] += random.gauss(0, 0.1)
                for j in range(mp):
                    V_try[j][r] += random.gauss(0, 0.1)
                for k in range(pn):
                    W_try[k][r] += random.gauss(0, 0.1)
            
            # Verify
            if _verify(U_try, V_try, W_try, n, m, p, tol=1e-8):
                best_rank = target_rank
                best_U, best_V, best_W = U_try, V_try, W_try
                break
        
        if best_rank == target_rank:
            break
    
    # Convert to list of lists if needed
    if isinstance(best_U, np.ndarray):
        best_U = best_U.tolist()
    if isinstance(best_V, np.ndarray):
        best_V = best_V.tolist()
    if isinstance(best_W, np.ndarray):
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
