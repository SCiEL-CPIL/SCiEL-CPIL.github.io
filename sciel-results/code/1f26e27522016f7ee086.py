"""Fresh design: randomized bilinear rank reduction with time guard."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 6]
DEADLINE = 140.0  # safety margin before 150s kill

def _time_remaining():
    return DEADLINE - time.time()

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # The matrix multiplication tensor: T[i,j,k] = 1 if we multiply A[i,j]*B[j,k]
    # We want rank-r decomposition: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # where W uses trace convention W[k*n + i, r]
    
    # Build the full tensor T of shape (n*m, m*p, p*n)
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Start with schoolbook rank
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1.0
                    V[j*p + k, col] = 1.0
                    W[k*n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    U, V, W, best_rank = schoolbook()
    
    # Try to reduce rank using iterative optimization
    # Strategy: for a given target rank r, initialize randomly and do gradient-free refinement
    # by projecting the tensor onto rank-r manifold
    
    def try_rank_reduction(target_rank, U_init, V_init, W_init, max_iters=500):
        """Try to fit tensor T with rank target_rank using alternating minimization."""
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Flatten T for faster computation
        T_flat = T.reshape(n*m, -1)  # (n*m, m*p*p*n)
        
        for it in range(max_iters):
            if _time_remaining() < 5.0:
                break
            
            # Update W: for each column c, minimize ||T - U_c V_c^T||_F
            # T[i, a, b] = sum_c U[i,c] V[a,c] W[b,c]
            # Fix U, V: for each c, W[:,c] should satisfy:
            # sum_c W[b,c] * (U[i,c] V[a,c]) = T[i,a,b]
            # This is: for each (i,a), sum_c (U[i,c]*V[a,c]) W[b,c] = T[i,a,b]
            # Let M[i,a,c] = U[i,c]*V[a,c], then M @ W = T_flat
            M = np.einsum('ic,ac->iac', U, V)  # (n*m, m*p, target_rank)
            M_flat = M.reshape(n*m * m*p, target_rank)
            # Solve: M_flat @ W_flat = T_flat where W_flat is (m*p*p*n, target_rank)
            # Actually T_flat is (n*m, m*p*p*n), reshape to (n*m*m*p, p*n)
            T_reshaped = T.reshape(n*m * m*p, p*n)
            # M_flat: (n*m*m*p, target_rank), T_reshaped: (n*m*m*p, p*n)
            # We want W_flat: (p*n, target_rank) such that M_flat @ W_flat.T = T_reshaped
            # i.e., for each row idx: M_flat[idx,:] @ W_flat.T = T_reshaped[idx,:]
            # => W_flat.T = M_flat^{-1} T_reshaped (least squares)
            try:
                W_flat, residuals, rank_w, sv = np.linalg.lstsq(M_flat, T_reshaped, rcond=None)
                W = W_flat.T  # (p*n, target_rank)
            except np.linalg.LinAlgError:
                break
            
            # Update U: fix V, W
            # T[i, a, b] = sum_c U[i,c] V[a,c] W[b,c]
            # For each (a,b): sum_c U[i,c] (V[a,c]*W[b,c]) = T[i,a,b]
            # Let N[a,b,c] = V[a,c]*W[b,c], then for each (a,b): N[a,b,:] @ U[:,c]... 
            # Actually: T[i,a,b] = sum_c U[i,c] * (V[a,c] * W[b,c])
            # Let P[a,b,c] = V[a,c]*W[b,c], shape (m*p, p*n, target_rank)
            # T[i,a,b] = sum_c U[i,c] P[a,b,c]
            # => For each (a,b): P[a,b,:] @ U^T = T[:,a,b]
            # Let P_flat: (m*p*p*n, target_rank), T_col: (n*m, m*p*p*n)
            P = np.einsum('ac,bc->abc', V, W)  # (m*p, p*n, target_rank)
            P_flat = P.reshape(m*p * p*n, target_rank)
            T_col = T.reshape(n*m, m*p * p*n)
            # T_col[i, ab] = sum_c U[i,c] P_flat[ab, c]
            # => T_col = U @ P_flat.T
            # => U = T_col @ P_flat.T^{-1} (least squares)
            try:
                U, residuals, rank_u, sv = np.linalg.lstsq(P_flat.T, T_col.T, rcond=None)
                U = U.T  # (n*m, target_rank)
            except np.linalg.LinAlgError:
                break
            
            # Update V: fix U, W
            # T[i,a,b] = sum_c (U[i,c]*W[b,c]) V[a,c]
            # Let Q[i,b,c] = U[i,c]*W[b,c], shape (n*m, p*n, target_rank)
            # T[i,a,b] = sum_c Q[i,b,c] V[a,c]
            # For each (i,b): Q[i,b,:] @ V^T = T[i,:,b]
            Q = np.einsum('ic,bc->ibc', U, W)  # (n*m, p*n, target_rank)
            Q_flat = Q.reshape(n*m * p*n, target_rank)
            T_reshaped2 = T.reshape(n*m, m*p, p*n).transpose(0, 2, 1).reshape(n*m * p*n, m*p)
            # T_reshaped2[i*b, a] = sum_c Q_flat[i*b, c] V[a, c]
            # => T_reshaped2 = Q_flat @ V.T
            try:
                V, residuals, rank_v, sv = np.linalg.lstsq(Q_flat, T_reshaped2, rcond=None)
                V = V.T  # (m*p, target_rank)
            except np.linalg.LinAlgError:
                break
        
        # Compute final residual
        residual = 0.0
        for c in range(target_rank):
            residual_tensor = np.einsum('i,a,b->iab', U[:,c], V[:,c], W[:,c])
            residual = np.max(np.abs(T - residual))
        
        return U, V, W, residual
    
    # Try progressively lower ranks
    # Known: Strassen on 2x2 blocks gives rank 66. Try to go lower.
    # The theoretical lower bound for 3x4x6 is at least max(3*4, 4*6, 3*6) = 24
    # but likely higher. Let's try ranks from 66 down.
    
    np.random.seed(42)
    
    # Try a few target ranks with random initialization
    target_ranks = [66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24]
    
    best_U, best_V, best_W, best_r = U, V, W, best_rank
    
    for target_r in target_ranks:
        if _time_remaining() < 10.0:
            break
        
        if target_r >= best_rank:
            continue
        
        # Multiple random restarts for this rank
        max_restarts = 3 if target_r > 40 else 2
        for restart in range(max_restarts):
            if _time_remaining() < 5.0:
                break
            
            # Random initialization
            U_r = np.random.randn(n*m, target_r) * 0.1
            V_r = np.random.randn(m*p, target_r) * 0.1
            W_r = np.random.randn(p*n, target_r) * 0.1
            
            # Or use a structured init based on schoolbook
            if restart == 0:
                # Use first target_r columns of schoolbook as init
                U_r = U[:, :target_r].copy()
                V_r = V[:, :target_r].copy()
                W_r = W[:, :target_r].copy()
            
            U_t, V_t, W_t, res = try_rank_reduction(target_r, U_r, V_r, W_r, max_iters=300)
            
            if res < 1e-8:  # Found exact decomposition
                best_U, best_V, best_W, best_rank = U_t, V_t, W_t, target_r
                print(f"Found exact rank {target_r}!")
                break
        
        if _time_remaining() < 10.0:
            break
    
    # Polish: if we found a good rank, try to verify and clean up
    if _time_remaining() > 2.0:
        # Try to round to half-integers or integers
        def clean_and_verify(U_in, V_in, W_in, r):
            """Try to round entries to half-integers and verify."""
            U_c = np.round(U_in * 2) / 2
            V_c = np.round(V_in * 2) / 2
            W_c = np.round(W_in * 2) / 2
            
            # Verify
            max_err = 0.0
            for c in range(r):
                rt = np.einsum('i,a,b->iab', U_c[:,c], V_c[:,c], W_c[:,c])
                max_err = max(max_err, np.max(np.abs(T - rt)))
            
            if max_err < 1e-6:
                return U_c, V_c, W_c, True
            return U_in, V_in, W_in, False
        
        best_U, best_V, best_W, cleaned = clean_and_verify(best_U, best_V, best_W, best_rank)
    
    # Convert to list of lists with Fraction for exactness
    def to_fraction_list(arr):
        rows, cols = arr.shape
        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                val = arr[i, j]
                # Try to express as fraction
                frac = Fraction(val).limit_denominator(1000)
                row.append(frac)
            result.append(row)
        return result
    
    return {
        "U": to_fraction_list(best_U),
        "V": to_fraction_list(best_V),
        "W": to_fraction_list(best_W)
    }
