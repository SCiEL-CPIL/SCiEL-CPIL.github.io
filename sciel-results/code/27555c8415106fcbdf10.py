"""Program for matrix multiplication 4x4 * 4x8 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 4, 8]
DEADLINE_SECONDS = 110.0  # Safety margin from 120s limit

def construct():
    """
    Return a tensor decomposition for 4x4 * 4x8 matrix multiplication.
    Uses tensor completion: fix U,V to structured basis, solve for W.
    """
    start = time.time()
    n, m, p = DIMENSIONS
    
    def _time_left():
        return DEADLINE_SECONDS - (time.time() - start)
    
    # Strassen 2x2 factors
    S = [
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),
        ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),
        ([0,0,0,1], [-1,0,1,0], [1,0,1,0]),
        ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),
        ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),
        ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),
    ]
    
    # Build rank-98 Strassen decomposition as guaranteed baseline
    def _strassen_98():
        rank = 98
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for t in range(2):
            for s in range(7):
                u_block, v_block, w_block = S[s]
                for r in range(7):
                    u_scalar, v_scalar, w_scalar = S[r]
                    col = t * 49 + s * 7 + r
                    
                    for bi in range(2):
                        for bj in range(2):
                            cb = u_block[bi*2+bj]
                            if cb == 0:
                                continue
                            for ei in range(2):
                                for ej in range(2):
                                    cs = u_scalar[ei*2+ej]
                                    if cs == 0:
                                        continue
                                    i = 2*bi + ei
                                    j = 2*bj + ej
                                    U[i*m + j][col] += cb * cs
                    
                    for bi in range(2):
                        for bj in range(2):
                            cb = v_block[bi*2+bj]
                            if cb == 0:
                                continue
                            for ei in range(2):
                                for ej in range(2):
                                    cs = v_scalar[ei*2+ej]
                                    if cs == 0:
                                        continue
                                    j = 2*bi + ei
                                    k = 4*t + 2*bj + ej
                                    V[j*p + k][col] += cb * cs
                    
                    for bi in range(2):
                        for bj in range(2):
                            cb = w_block[bi*2+bj]
                            if cb == 0:
                                continue
                            for ei in range(2):
                                for ej in range(2):
                                    cs = w_scalar[ei*2+ej]
                                    if cs == 0:
                                        continue
                                    i = 2*bi + ei
                                    k = 4*t + 2*bj + ej
                                    W[k*n + i][col] += cb * cs
        
        return {"U": U, "V": V, "W": W}
    
    def _verify(U, V, W):
        rank = len(U[0])
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        return np.allclose(result, 1.0, atol=1e-10)
    
    # First, get the Strassen baseline
    best_result = _strassen_98()
    best_rank = 98
    
    # Tensor completion: fix U,V, solve for W
    def _solve_for_W(U, V, n, m, p):
        """Given U and V, solve for W such that einsum gives identity."""
        rank = U.shape[1]
        from scipy.sparse import lil_matrix
        from scipy.sparse.linalg import spsolve
        
        equations = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row = []
                    for r in range(rank):
                        coeff = U[i*m+j, r] * V[j*p+k, r]
                        if coeff != 0:
                            row.append((k*n + i, r, coeff))
                    if row:
                        equations.append((row, 1.0))
        
        if not equations:
            return None
        
        num_unknowns = p * n * rank
        A = lil_matrix((len(equations), num_unknowns), dtype=float)
        b = np.zeros(len(equations))
        
        for eq_idx, (row, rhs) in enumerate(equations):
            b[eq_idx] = rhs
            for (w_idx, r, coeff) in row:
                x_idx = w_idx * rank + r
                A[eq_idx, x_idx] = coeff
        
        try:
            A_csc = A.tocsr()
            x = spsolve(A_csc, b)
            if x is None:
                return None
            residual = np.abs(A_csc @ x - b)
            if np.max(residual) > 1e-8:
                return None
            W = x.reshape(p * n, rank)
            return W
        except Exception:
            return None
    
    def _try_decomposition(rank, n, m, p, seed=42):
        """Try to find a decomposition of given rank."""
        rng = np.random.RandomState(seed)
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        
        # Use structured initialization: each rank term covers a Strassen-like pattern
        for r in range(rank):
            # Strassen-inspired: each term touches a small number of (i,j) and (j,k)
            num_u = rng.randint(2, 5)
            u_indices = rng.choice(n*m, size=num_u, replace=False)
            for idx in u_indices:
                U[idx, r] = rng.choice([-1, 1])
            
            num_v = rng.randint(2, 5)
            v_indices = rng.choice(m*p, size=num_v, replace=False)
            for idx in v_indices:
                V[idx, r] = rng.choice([-1, 1])
        
        W = _solve_for_W(U, V, n, m, p)
        if W is None:
            return None
        
        # Verify
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        if np.allclose(result, 1.0, atol=1e-8):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        return None
    
    # Try to find lower rank decompositions
    for rank in range(80, 98):
        if _time_left() < 5.0:
            break
        
        for seed in range(30):
            if _time_left() < 2.0:
                break
            
            result = _try_decomposition(rank, n, m, p, seed=seed)
            if result is not None:
                if _verify(result["U"], result["V"], result["W"]):
                    if rank < best_rank:
                        best_rank = rank
                        best_result = result
                    break
    
    if _time_left() < 1.0:
        return best_result
    
    return best_result
