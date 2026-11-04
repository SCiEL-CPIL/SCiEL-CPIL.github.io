"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    # Precompute the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                I_idx = i*m + j
                J_idx = j*p + k
                K_idx = k*n + i
                T[I_idx, J_idx, K_idx] = 1.0

    def verify_half_integer(U, V, W, tol=1e-9):
        """Verify U, V, W have half-integer entries and exactly decompose T."""
        U2 = U * 2
        V2 = V * 2
        W2 = W * 2
        
        U_round = np.round(U2)
        V_round = np.round(V2)
        W_round = np.round(W2)
        
        if not (np.allclose(U2, U_round, atol=tol) and 
                np.allclose(V2, V_round, atol=tol) and 
                np.allclose(W2, W_round, atol=tol)):
            return False, None
            
        U_check = U_round / 2.0
        V_check = V_round / 2.0
        W_check = W_round / 2.0
        
        T_approx = np.einsum('ir, jr, kr -> ijk', U_check, V_check, W_check)
        if np.allclose(T_approx, T, atol=tol):
            return True, (U_check, V_check, W_check)
        return False, None

    def try_als_search(R, num_restarts, max_iters, time_limit):
        """ALS-based search for rank R decomposition."""
        for restart in range(num_restarts):
            if time.time() > time_limit:
                return None
            
            np.random.seed(12345 + restart)
            U = np.random.randn(n*m, R) * 0.5
            V = np.random.randn(m*p, R) * 0.5
            W = np.random.randn(p*n, R) * 0.5
            
            for it in range(max_iters):
                if time.time() > time_limit:
                    return None
                
                # Update U: for each row i, solve least squares
                for i in range(n*m):
                    # A_U[j,k,r] = V[j,r]*W[k,r], b_U = T[i,j,k]
                    A_U = np.einsum('jr, kr -> jkr', V, W).reshape(-1, R)
                    b_U = T[i, :, :].flatten()
                    U[i, :], _, _, _ = np.linalg.lstsq(A_U, b_U, rcond=None)
                
                # Update V
                for j in range(m*p):
                    A_V = np.einsum('ir, kr -> ikr', U, W).reshape(-1, R)
                    b_V = T[:, j, :].flatten()
                    V[j, :], _, _, _ = np.linalg.lstsq(A_V, b_V, rcond=None)
                
                # Update W
                for k in range(p*n):
                    A_W = np.einsum('ir, jr -> ijr', U, V).reshape(-1, R)
                    b_W = T[:, :, k].flatten()
                    W[k, :], _, _, _ = np.linalg.lstsq(A_W, b_W, rcond=None)
                
                # Check convergence
                T_approx = np.einsum('ir, jr, kr -> ijk', U, V, W)
                err = np.linalg.norm(T_approx - T)
                if err < 1e-8:
                    valid, result = verify_half_integer(U, V, W)
                    if valid:
                        return result
        return None

    def try_structured_kronecker():
        """
        Try to construct a decomposition using Kronecker product structure.
        3x4x4: We can decompose as (2+1)x4x4.
        Use rank-7 Strassen for 2x2x2, and combine with 2x2x2 for the 4x4 part.
        
        Strategy: 
        - Split the 3 rows into 2+1
        - Split the 4 cols into 2+2
        - Split the 4 output cols into 2+2
        
        This gives blocks:
        - 2x2x2 (top-left): use Strassen rank 7
        - 2x2x2 (top-right): use Strassen rank 7  
        - 1x2x2 (bottom-left): rank 2
        - 1x2x2 (bottom-right): rank 2
        
        Total: 7+7+2+2 = 18? That seems too low. Let me reconsider.
        
        Actually, the correct decomposition needs to handle the full 3x4x4 tensor.
        Let me use a different approach: direct construction with known patterns.
        """
        # Try a simpler structured approach: use the fact that we can share
        # computation between similar blocks.
        
        # For now, try rank 47 with a smarter initialization
        return None

    # Try structured approach first
    result = try_structured_kronecker()
    if result is not None:
        U, V, W = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try ALS search for rank 47
    if time.time() < deadline - 30:
        result = try_als_search(47, num_restarts=3, max_iters=20, time_limit=deadline - 30)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback to Schoolbook (rank 48)
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
