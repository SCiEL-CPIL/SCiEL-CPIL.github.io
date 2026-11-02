"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying
    a 3x4 matrix by a 4x7 matrix.
    
    Strategy:
    1. Build the matrix multiplication tensor T
    2. Use structured initialization based on Kronecker products
    3. Try to find lower-rank decompositions using ALS with smart initialization
    4. Fall back to schoolbook if needed
    """
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p  # 84

    # Build the tensor T[a, b, c] where:
    # a = i*m + j, b = j*p + k, c = k*n + i
    # T[a, b, c] = 1 if the indices are consistent (same i, j, k), else 0
    T = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]"""
        T_recon = np.zeros((nm, mp, pn), dtype=float)
        for r in range(U.shape[1]):
            T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
        return np.max(np.abs(T_recon - T)) < tol

    def round_to_half_integer(x):
        """Round to nearest multiple of 0.5"""
        return np.round(2 * x) / 2.0

    def als_decompose(T, rank, max_iters=50, tol=1e-10, seed=None, U_init=None, V_init=None, W_init=None):
        """
        Alternating Least Squares for tensor decomposition.
        T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()
        
        # Initialize
        if U_init is not None and V_init is not None and W_init is not None:
            U = U_init.copy()
            V = V_init.copy()
            W = W_init.copy()
        else:
            U = rng.randn(nm, rank)
            V = rng.randn(mp, rank)
            W = rng.randn(pn, rank)
        
        prev_err = float('inf')
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            # Update U: for each a, U[a,:] = (V ⊙ W)^T T[a,:,:]
            VW = np.kron(V, W)  # shape (mp*pn, rank)
            T_unfold1 = T.reshape(nm, mp * pn)
            try:
                U_new = np.linalg.lstsq(VW.T, T_unfold1.T, rcond=None)[0].T
            except:
                break
            U = U_new
            
            # Update V: for each b, V[b,:] = (U ⊙ W)^T T[:,b,:]
            UW = np.kron(U, W)  # shape (nm*pn, rank)
            T_unfold2 = T.transpose(1, 2, 0).reshape(mp, nm * pn)
            try:
                V_new = np.linalg.lstsq(UW.T, T_unfold2.T, rcond=None)[0].T
            except:
                break
            V = V_new
            
            # Update W: for each c, W[c,:] = (U ⊙ V)^T T[:,:,c]
            UV = np.kron(U, V)  # shape (nm*mp, rank)
            T_unfold3 = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            try:
                W_new = np.linalg.lstsq(UV.T, T_unfold3.T, rcond=None)[0].T
            except:
                break
            W = W_new
            
            # Compute error
            T_recon = np.zeros((nm, mp, pn), dtype=float)
            for r in range(rank):
                T_recon += np.outer(np.outer(U[:, r], V[:, r]), W[:, r])
            err = np.linalg.norm(T_recon - T)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return U, V, W, err

    def try_round_and_verify(U, V, W, max_trials=200):
        """
        Try to find scalings and roundings that produce valid half-integer decomposition.
        Returns (U_r, V_r, W_r) if valid, else None.
        """
        if time.time() > deadline:
            return None
        
        # First try direct rounding
        U_r = round_to_half_integer(U)
        V_r = round_to_half_integer(V)
        W_r = round_to_half_integer(W)
        if verify_decomposition(U_r, V_r, W_r):
            return U_r, V_r, W_r
        
        # Try random scalings
        for _ in range(max_trials):
            if time.time() > deadline:
                break
            
            alpha = np.random.uniform(0.3, 3.0)
            beta = np.random.uniform(0.3, 3.0)
            gamma = 1.0 / (alpha * beta)
            
            U_s = U * alpha
            V_s = V * beta
            W_s = W * gamma
            
            U_r = round_to_half_integer(U_s)
            V_r = round_to_half_integer(V_s)
            W_r = round_to_half_integer(W_s)
            
            if verify_decomposition(U_r, V_r, W_r):
                return U_r, V_r, W_r
        
        return None

    def structured_init(rank, n, m, p):
        """
        Create structured initialization based on the Kronecker product structure.
        For matrix multiplication, we can use the fact that:
        T[i*m+j, j*p+k, k*n+i] = 1
        
        We can initialize U, V, W based on the index structure.
        """
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        # For each (i,j,k), we have a term that contributes to T
        # We can distribute these across rank components
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = idx % rank
                    U[i*m + j, r] += 1.0
                    V[j*p + k, r] += 1.0
                    W[k*n + i, r] += 1.0
                    idx += 1
        
        return U, V, W

    # Compute lower bound from unfolding ranks
    T_unfold1 = T.reshape(nm, mp * pn)
    T_unfold2 = T.reshape(nm * mp, pn)
    T_unfold3 = T.reshape(mp, nm * pn)
    
    try:
        rank_lb1 = np.linalg.matrix_rank(T_unfold1, tol=1e-10)
        rank_lb2 = np.linalg.matrix_rank(T_unfold2, tol=1e-10)
        rank_lb3 = np.linalg.matrix_rank(T_unfold3, tol=1e-10)
        rank_lb = max(rank_lb1, rank_lb2, rank_lb3)
    except Exception:
        rank_lb = 1

    rank_lb = max(1, int(rank_lb))
    
    best_result = None
    best_rank = schoolbook_rank
    
    # Try ranks from lower bound up to schoolbook
    max_rank_to_try = min(schoolbook_rank, rank_lb + 40)
    
    for rank in range(rank_lb, max_rank_to_try + 1):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple random restarts
        num_restarts = max(3, 20 - (rank - rank_lb))
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            try:
                # Use structured initialization for the first restart, random for others
                if restart == 0:
                    U_init, V_init, W_init = structured_init(rank, n, m, p)
                    U, V, W, err = als_decompose(T, rank, max_iters=80, tol=1e-12, seed=restart*1000+rank, 
                                                  U_init=U_init, V_init=V_init, W_init=W_init)
                else:
                    U, V, W, err = als_decompose(T, rank, max_iters=80, tol=1e-12, seed=restart*1000+rank)
                
                if err < 1e-8:
                    # Good fit, try to round
                    result = try_round_and_verify(U, V, W, max_trials=100)
                    if result is not None:
                        U_r, V_r, W_r = result
                        best_result = (U_r, V_r, W_r, rank)
                        found = True
                        break
            except Exception:
                continue
        
        if found:
            break  # Found valid decomposition at this rank
    
    # If no better decomposition found, fall back to schoolbook
    if best_result is None:
        rank = schoolbook_rank
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, column] = 1
                    V[j * p + k, column] = 1
                    W[k * n + i, column] = 1
                    column += 1
        
        best_result = (U, V, W, rank)

    U, V, W, rank = best_result
    
    # Convert to lists of lists for compatibility
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
