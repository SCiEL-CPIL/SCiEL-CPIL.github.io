import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 2, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Construct the target tensor T
    T = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                T[u_idx, v_idx, w_idx] += 1.0
                
    best_result = None
    best_rank = 20 # Fallback to schoolbook
    
    # Pre-compute schoolbook solution as fallback
    U_school = np.zeros((dim_u, 20))
    V_school = np.zeros((dim_v, 20))
    W_school = np.zeros((dim_w, 20))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1.0
                V_school[j*p+k, col] = 1.0
                W_school[k*n+i, col] = 1.0
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    best_rank = 20

    def clean_matrix(M, max_denom=10):
        """
        Attempts to convert a float matrix to a matrix of Fractions with small denominators.
        Returns None if not possible.
        """
        M_clean = np.zeros(M.shape, dtype=object)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                val = M[i, j]
                # Try to find a fraction p/q close to val
                found = False
                for q in range(1, max_denom + 1):
                    p = round(val * q)
                    frac = Fraction(p, q)
                    if abs(val - float(frac)) < 1e-8:
                        M_clean[i, j] = frac
                        found = True
                        break
                if not found:
                    return None
        return M_clean

    def verify_solution(U, V, W):
        """
        Verifies if the given matrices (as Fractions or floats) exactly reconstruct T.
        """
        T_rec = np.zeros_like(T)
        R = U.shape[1]
        for c in range(R):
            # Convert columns to floats for calculation
            uc = np.array([float(x) for x in U[:, c]])
            vc = np.array([float(x) for x in V[:, c]])
            wc = np.array([float(x) for x in W[:, c]])
            T_rec += np.outer(uc, np.outer(vc, wc))
        return np.allclose(T_rec, T, atol=1e-9)

    def try_rank(R, restarts=20, max_iters=100):
        nonlocal best_result, best_rank
        if R >= best_rank:
            return
            
        for r in range(restarts):
            if time.time() > deadline:
                return
                
            # Initialize
            np.random.seed(1000 + r)
            U = np.random.randn(dim_u, R)
            V = np.random.randn(dim_v, R)
            W = np.random.randn(dim_w, R)
            
            for it in range(max_iters):
                if time.time() > deadline:
                    return
                
                # Update U
                T_mat = T.reshape(dim_u, dim_v * dim_w)
                M = np.zeros((R, dim_v * dim_w))
                for c in range(R):
                    M[c, :] = (V[:, c] * W[:, c]).flatten()
                try:
                    U_new = np.linalg.lstsq(M, T_mat.T, rcond=None)[0].T
                except:
                    continue
                U = U_new
                
                # Update V
                T_mat_v = T.transpose(1, 0, 2).reshape(dim_v, dim_u * dim_w)
                M_v = np.zeros((R, dim_u * dim_w))
                for c in range(R):
                    M_v[c, :] = (U[:, c] * W[:, c]).flatten()
                try:
                    V_new = np.linalg.lstsq(M_v, T_mat_v.T, rcond=None)[0].T
                except:
                    continue
                V = V_new
                
                # Update W
                T_mat_w = T.transpose(2, 0, 1).reshape(dim_w, dim_u * dim_v)
                M_w = np.zeros((R, dim_u * dim_v))
                for c in range(R):
                    M_w[c, :] = (U[:, c] * V[:, c]).flatten()
                try:
                    W_new = np.linalg.lstsq(M_w, T_mat_w.T, rcond=None)[0].T
                except:
                    continue
                W = W_new
                
                # Check error
                T_rec = np.zeros_like(T)
                for c in range(R):
                    T_rec += np.outer(U[:, c], np.outer(V[:, c], W[:, c]))
                error = np.linalg.norm(T_rec - T)
                
                if error < 1e-8:
                    # Try to clean
                    U_clean = clean_matrix(U)
                    if U_clean is not None:
                        V_clean = clean_matrix(V)
                        if V_clean is not None:
                            W_clean = clean_matrix(W)
                            if W_clean is not None:
                                # Verify with cleaned values
                                if verify_solution(U_clean, V_clean, W_clean):
                                    best_result = {"U": U_clean, "V": V_clean, "W": W_clean}
                                    best_rank = R
                                    return
                                else:
                                    # Fallback: use floats if they are "exact enough" numerically
                                    # But the prompt wants integer/half-integer.
                                    # If cleaning failed verification, it's not a valid solution for the prompt's constraints.
                                    pass

    # Search ranks
    # Prioritize lower ranks with more restarts
    search_plan = [
        (14, 50),
        (15, 50),
        (16, 30),
        (17, 20),
        (18, 10),
        (19, 5),
        (20, 1)
    ]
    
    for R, restarts in search_plan:
        if time.time() > deadline:
            break
        try_rank(R, restarts)
        if best_rank == R:
            break
            
    # Return the best result found
    U_out = best_result["U"]
    V_out = best_result["V"]
    W_out = best_result["W"]
    
    # Convert to lists of floats (or ints)
    # The seed returns lists.
    # If U_out is a numpy array of objects (Fractions), convert to floats.
    # The prompt says "integer/half-integer entries".
    # Returning floats like 0.5, 1.0 is acceptable as they represent the rationals.
    
    def to_list(M):
        if isinstance(M, np.ndarray):
            if M.dtype == object:
                return [[float(x) for x in row] for row in M]
            else:
                return M.tolist()
        else:
            return M
            
    return {"U": to_list(U_out), "V": to_list(V_out), "W": to_list(W_out)}
