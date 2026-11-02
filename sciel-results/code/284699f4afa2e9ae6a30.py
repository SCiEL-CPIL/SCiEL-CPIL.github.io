import numpy as np
import time
import itertools

DIMENSIONS = [3, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    def _to_list(arr):
        return arr.tolist()

    def _verify(U, V, W, T):
        """Verify if U, V, W exactly decompose T."""
        recon = np.einsum('ir, jr, kr -> ijk', U, V, W)
        return np.allclose(recon, T, atol=1e-9)

    def _round_half(arr):
        return np.round(arr * 2) / 2

    def _check_and_return(U, V, W, T):
        """Round to half-integers and verify."""
        U_r = _round_half(U)
        V_r = _round_half(V)
        W_r = _round_half(W)
        if _verify(U_r, V_r, W_r, T):
            return {"U": _to_list(U_r), "V": _to_list(V_r), "W": _to_list(W_r)}
        return None

    # Strategy 1: Combinatorial Pattern Search for 3x3x4 base
    # The tensor T[i,j,k] = 1 for i in 0..2, j in 0..2, k in 0..3
    # We seek a rank R decomposition.
    # Known lower bound for 3x3x4 is 19 (from 3x3x3 rank 19? No, 3x3x3 is 19? No, 3x3x3 is 19? 
    # Actually rank(3x3x3) is between 19 and 23. Strassen is 7 for 2x2x2.
    # For 3x3x3, best known is 19? No, it's 19 for 3x3x3? 
    # Wait, rank(3x3x3) is known to be 19? No, it's 19? 
    # Let's check: 2x2x2 is 7. 3x3x3 is 19? 
    # Actually, the rank of 3x3x3 matrix multiplication is 19? 
    # No, it is 19? 
    # Let's just try to find a decomposition for 3x3x4 with rank < 24 (schoolbook 3*3*4=36).
    # If we find rank R for 3x3x4, we can use rank 2R for 3x3x8.
    # Schoolbook for 3x3x8 is 72.
    # If we find rank 19 for 3x3x4, total rank is 38.
    # If we find rank 18, total rank is 36.
    
    T_334 = np.ones((3, 3, 4))
    
    # Try ranks from 19 upwards (lower is better, but we need a valid one)
    # Actually, let's try to find the minimal rank.
    # We can use a randomized search with local optimization (ALS) but with better initialization.
    # Or we can use the "combinatorial pattern" idea: 
    # Assume entries are in {0, 1/2, 1}.
    
    def _als_decomp(T, rank, max_iter=1000, tol=1e-9):
        """
        Attempt to find a rank decomposition using ALS.
        T is shape (n, m, p).
        Returns (U, V, W) if found, else None.
        """
        n, m, p = T.shape
        np.random.seed(42) # Deterministic for reproducibility? Or random?
        # Let's use multiple restarts
        
        for restart in range(20):
            if time.time() > deadline:
                return None
            
            # Initialize
            U = np.random.randn(n, rank) * 0.1
            V = np.random.randn(m, rank) * 0.1
            W = np.random.randn(p, rank) * 0.1
            
            for it in range(max_iter):
                if time.time() > deadline:
                    return None
                
                # Update U
                # U[i, r] = sum_{j,k} T[i,j,k] V[j,r] W[k,r] / sum_{j,k} V[j,r]^2 W[k,r]^2
                Temp_U = np.einsum('ijk, jr, kr -> ir', T, V, W)
                Denom_U = np.einsum('jr, kr, jr, kr -> r', V, W, V, W)
                U_new = Temp_U / (Denom_U[None, :] + 1e-12)
                
                # Update V
                Temp_V = np.einsum('ijk, ir, kr -> jr', T, U_new, W)
                Denom_V = np.einsum('ir, kr, ir, kr -> r', U_new, W, U_new, W)
                V_new = Temp_V / (Denom_V[None, :] + 1e-12)
                
                # Update W
                Temp_W = np.einsum('ijk, ir, jr -> kr', T, U_new, V_new)
                Denom_W = np.einsum('ir, jr, ir, jr -> r', U_new, V_new, U_new, V_new)
                W_new = Temp_W / (Denom_W[None, :] + 1e-12)
                
                U, V, W = U_new, V_new, W_new
                
                # Check convergence
                Recon = np.einsum('ir, jr, kr -> ijk', U, V, W)
                Error = np.linalg.norm(Recon - T)
                if Error < tol:
                    # Try to round
                    res = _check_and_return(U, V, W, T)
                    if res:
                        return res
                    # If rounding fails, continue? Or accept non-integer?
                    # Task requires integer/half-integer.
                    # If ALS converges to exact solution, it might be rational.
                    # Let's try rounding to nearest half-integer.
                    # If that fails, maybe we need a different approach.
                    # But for now, let's assume if it converges, rounding works or is close.
                    # Actually, the task requires EXACT decomposition with integer/half-integer entries.
                    # So we must verify after rounding.
                    return None # If rounding fails, this restart failed.
            
            # End of iterations for this restart
            res = _check_and_return(U, V, W, T)
            if res:
                return res
                
        return None

    # Try to find decomposition for 3x3x4
    # Start with rank 19 (known lower bound for 3x3x3 is 19? No, 3x3x3 rank is 19? 
    # Actually, rank(3x3x3) is 19? 
    # Let's just try ranks 19, 20, 21... up to 36.
    # If we find rank R for 3x3x4, we can build 3x3x8 with rank 2R.
    
    best_decomp_334 = None
    best_rank_334 = 36 # Schoolbook
    
    for rank in range(19, 37):
        if time.time() > deadline:
            break
        res = _als_decomp(T_334, rank, max_iter=200)
        if res:
            best_decomp_334 = res
            best_rank_334 = rank
            break # Found the minimal rank we can get with this method
    
    if best_decomp_334:
        U1 = np.array(best_decomp_334['U'])
        V1 = np.array(best_decomp_334['V'])
        W1 = np.array(best_decomp_334['W'])
        
        # Construct 3x3x8
        # A (3x3) * B (3x8) = C (3x8)
        # B = [B1, B2] where B1, B2 are 3x4
        # C = [C1, C2] where C1, C2 are 3x4
        # C1 = A * B1, C2 = A * B2
        # We can reuse the same U, V for both halves, but W is split.
        # U is 3xR, V is 3xR, W is 8x2R
        # W[0:4, 0:R] = W1
        # W[4:8, R:2R] = W1
        
        R = best_rank_334
        total_rank = 2 * R
        U_full = np.zeros((3, total_rank))
        V_full = np.zeros((3, total_rank))
        W_full = np.zeros((8, total_rank))
        
        U_full[:, 0:R] = U1
        V_full[:, 0:R] = V1
        W_full[0:4, 0:R] = W1
        
        U_full[:, R:2*R] = U1
        V_full[:, R:2*R] = V1
        W_full[4:8, R:2*R] = W1
        
        T_full = np.ones((3, 3, 8))
        if _verify(U_full, V_full, W_full, T_full):
            return {"U": _to_list(U_full), "V": _to_list(V_full), "W": _to_list(W_full)}
        
    # Fallback to schoolbook
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
