import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Target Tensor T
    # T[i*m+j, j*p+k, k*n+i] = 1
    # Shape: (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    start_time = time.time()
    time_limit = 145.0  # Safety margin

    best_U = None
    best_V = None
    best_W = None
    best_rank = rank_schoolbook

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly reconstructs T."""
        T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def try_round_and_verify(U, V, W):
        """Try rounding to integers/half-integers and verify."""
        for scale in [1, 2]:
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        return None, None, None

    # Method 1: Strassen-style block decomposition
    # For 2x5x6, we can split into blocks and use Strassen on 2x2 blocks
    # The 5x6 matrix can be split into 2x2 blocks with some remainder
    # Key insight: n=2 is small, so we can use the 2x2 Strassen trick
    
    # Build schoolbook as fallback
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = rank_schoolbook

    # Try Strassen-based approach
    # For multiplying A (2x5) by B (5x6):
    # Split A into [A1, A2] where A1 is 2x3, A2 is 2x2
    # Split B into [B1; B2] where B1 is 3x6, B2 is 2x6
    # C = A1*B1 + A2*B2
    # Use Strassen on the 2x2 part: A2*B2 uses 7 multiplications instead of 4
    # Total: 2*3*6 (for A1*B1) + 7 (for A2*B2 via Strassen) = 36 + 7 = 43
    # But we need to be careful about the tensor structure
    
    # More systematically: try various rank values with structured initialization
    # and alternating least squares
    
    # Try ranks from a good lower bound upward
    # Slice rank lower bound: for each fixed k, the slice T[:,:,k*n+i] 
    # gives a matrix of size (n*m) x (m*p) = 10 x 30
    # The rank of each slice gives a lower bound
    
    # Compute slice ranks
    slice_ranks = []
    for i in range(n):
        for k in range(p):
            # Slice where output index is k*n+i
            # T[im+j, jp+k, kn+i] for all i,j
            # This is a matrix M where M[i*m+j, j*p+k] = 1 if the indices are consistent
            # Actually, for fixed i0, k0: T[i*m+j, j*p+k0, k0*n+i] 
            # The nonzero entries are when i=i0 and the j,k indices match
            # This is a permutation-like matrix
            pass
    
    # For n=2, m=5, p=6: 
    # The multiplication tensor has rank at most 60 (schoolbook)
    # Known: rank of 2x2x2 is 7 (Strassen)
    # For 2xm x px2 or similar, we can use block Strassen
    
    # Try alternating least squares with various initializations
    # Focus on ranks that are likely to work: 43, 44, 45, ...
    
    candidate_ranks = []
    # Start with Strassen-inspired ranks
    # 2x5x6: think of it as 2x(3+2) x (3+2)x6
    # Or better: use the fact that n=2 allows efficient 2x2 Strassen
    
    # Systematic approach: try ranks from 40 to 59
    # With good initialization, we might find a solution
    
    for r in range(40, min(rank_schoolbook, 60)):
        if time.time() - start_time > time_limit:
            break
        
        found = False
        
        for restart in range(50):
            if time.time() - start_time > time_limit:
                break
            
            # Random initialization
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            converged = False
            best_err = float('inf')
            
            for it in range(300):
                if time.time() - start_time > time_limit:
                    break
                
                # Update U: for each i, solve linear system
                # T[i, :, :] = sum_c U[i,c] * outer(V[:,c], W[:,c])
                # Vectorize: T_flat[i, :] = U[i, :] @ A_U.T
                # where A_U[j, c] = <V[:,c], W[:,c]>_elementwise = V[:,c] * W[:,c] flattened
                
                # Build A_U efficiently
                # A_U has shape (m*p*p*n, r)
                # A_U[flat_index, c] = V[j,c] * W[k,c] where flat_index = j*(p*n) + k
                
                # Use einsum to compute the basis
                # For speed, compute in chunks
                A_U = np.einsum('jc,kc->jkc', V, W).reshape(m*p*p*n, r)
                T_flat = T.reshape(n*m, m*p*p*n)
                
                try:
                    U_new = np.linalg.lstsq(A_U, T_flat.T, rcond=None)[0].T
                    U = U_new
                except:
                    pass
                
                # Update V
                A_V = np.einsum('ic,kc->ikc', U, W).reshape(n*m*p*n, r)
                T_perm_V = np.transpose(T, (1, 0, 2))
                T_flat_V = T_perm_V.reshape(m*p, n*m*p*n)
                
                try:
                    V_new = np.linalg.lstsq(A_V, T_flat_V.T, rcond=None)[0].T
                    V = V_new
                except:
                    pass
                
                # Update W
                A_W = np.einsum('ic,jc->ijc', U, V).reshape(n*m*m*p, r)
                T_perm_W = np.transpose(T, (2, 0, 1))
                T_flat_W = T_perm_W.reshape(p*n, n*m*m*p)
                
                try:
                    W_new = np.linalg.lstsq(A_W, T_flat_W.T, rcond=None)[0].T
                    W = W_new
                except:
                    pass
                
                # Check convergence
                T_rec = np.einsum('ic,jc,kc->ijk', U, V, W)
                err = np.linalg.norm(T - T_rec)
                best_err = min(best_err, err)
                
                if err < 1e-10:
                    converged = True
                    break
                
                # Early stopping if no improvement
                if it > 50 and err > best_err * 1.01:
                    break
            
            if converged:
                U_r, V_r, W_r = try_round_and_verify(U, V, W)
                if U_r is not None:
                    best_U, best_V, best_W = U_r, V_r, W_r
                    best_rank = r
                    found = True
                    break
        
        if found:
            # Found a solution at this rank, try lower
            continue
    
    # If no solution found below schoolbook, return schoolbook
    if best_rank == rank_schoolbook:
        best_U = U_school.tolist()
        best_V = V_school.tolist()
        best_W = W_school.tolist()
    else:
        best_U = best_U.tolist()
        best_V = best_V.tolist()
        best_W = best_W.tolist()
    
    return {"U": best_U, "V": best_V, "W": best_W}
