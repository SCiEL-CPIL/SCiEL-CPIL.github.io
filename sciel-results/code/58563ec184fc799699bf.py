import time
import random
import numpy as np
from fractions import Fraction

def _verify_exact(U, V, W, n, m, p):
    """
    Exact verification using fractions.
    Checks that the decomposition exactly matches the matrix multiplication tensor.
    T[u, v, w] = 1 if u//m == v//p and w%n == u//m and w//n == v%p
    T[u, v, w] = 0 otherwise
    """
    rank = len(U[0])
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Pre-compute expected values for all indices
    # We can just iterate and check
    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            j_v = v // p
            k = v % p
            for w in range(pn):
                i_w = w % n
                k_w = w // n
                
                expected = 0
                if j == j_v and i == i_w and k == k_w:
                    expected = 1
                
                val = Fraction(0)
                for c in range(rank):
                    val += U[u][c] * V[v][c] * W[w][c]
                
                if val != expected:
                    return False
    return True

def _verify_np(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using numpy."""
    U = np.array(U)
    V = np.array(V)
    W = np.array(W)
    rank = U.shape[1]
    nm, mp, pn = n*m, m*p, p*n
    
    # Tensorize
    # U is (nm, r), V is (mp, r), W is (pn, r)
    # We need to check sum_c U[u,c] V[v,c] W[w,c]
    
    # Efficient check:
    # Compute M_uv = sum_c U[u,c] V[v,c] for all u,v
    # Then for each w, check if M_uv * W[w,c] sum matches expected?
    # No, the sum is over c for the triple product.
    
    # Let's just loop over u, v, w if small enough.
    # 15 * 35 * 21 = 11025 triples.
    # For each triple, dot product of length r.
    # 11025 * r operations.
    # If r=100, 1.1M ops. Fast.
    
    for u in range(nm):
        i = u // m
        j = u % m
        for v in range(mp):
            j_v = v // p
            k = v % p
            for w in range(pn):
                i_w = w % n
                k_w = w // n
                
                expected = 1.0 if (j == j_v and i == i_w and k == k_w) else 0.0
                
                val = np.dot(U[u] * V[v], W[w])
                
                if abs(val - expected) > tol:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n, m, p = 3, 5, 7
    nm, mp, pn = n*m, m*p, p*n
    
    def schoolbook():
        rank = n * m * p
        U = [[0.0] * rank for _ in range(nm)]
        V = [[0.0] * rank for _ in range(mp)]
        W = [[0.0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1.0
                    V[j*p+k][col] = 1.0
                    W[k*n+i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    def try_als(r, num_iters=20, num_restarts=5):
        """
        Try to find a decomposition of rank r using Alternating Least Squares (ALS)
        followed by rounding.
        """
        if time.time() > deadline:
            return None
        
        # Build the target tensor T as a 3D array
        T = np.zeros((nm, mp, pn))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    T[u, v, w] = 1.0
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                return None
            
            random.seed(42 + r + restart)
            np.random.seed(42 + r + restart)
            
            # Initialize U, V, W randomly
            U = np.random.randn(nm, r)
            V = np.random.randn(mp, r)
            W = np.random.randn(pn, r)
            
            # Normalize to avoid scaling issues
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            
            for it in range(num_iters):
                if time.time() > deadline:
                    return None
                
                # Update U
                # For each u, U[u, :] is the solution to min || T[u, :, :] - sum_c U[u,c] (V[:,c] * W[:,c]) ||
                # This is a linear system: (V^T W) U[u, :]^T = (T[u, :, :] * (V * W^T))^T?
                # Let's derive:
                # Residual R[u, v, w] = T[u, v, w] - sum_c U[u,c] V[v,c] W[w,c]
                # We want to minimize sum_{v,w} R[u, v, w]^2
                # This is equivalent to solving for U[u, :] in:
                # sum_{v,w} (T[u, v, w] - sum_c U[u,c] V[v,c] W[w,c]) * d(U[u, c']) = 0
                # sum_{v,w} T[u, v, w] V[v, c'] W[w, c'] = sum_{v,w} sum_c U[u,c] V[v,c] W[w,c] V[v, c'] W[w, c']
                # LHS: (T[u] * (V * W^T)) flattened?
                # Let's use the mode-unfolding.
                # T_u = T[u, :, :] (mp x pn matrix)
                # V_c = V[:, c] (mp vector)
                # W_c = W[:, c] (pn vector)
                # Term is U[u,c] * V_c * W_c^T
                # We want to approximate T_u by sum_c U[u,c] (V_c * W_c^T)
                # This is a matrix factorization problem.
                # Flatten T_u to vector t_u (mp*pn).
                # Flatten V_c * W_c^T to vector v_w_c (mp*pn).
                # Then t_u = sum_c U[u,c] v_w_c.
                # So U[u, :] is the least squares solution of t_u = M_u U[u, :]^T
                # where M_u is a (mp*pn) x r matrix with columns v_w_c.
                
                # Constructing M_u for each u is expensive but n=3 is small.
                # Actually, M_u is the same for all u?
                # v_w_c = V[:, c] * W[:, c]^T.
                # Yes, the basis vectors v_w_c are independent of u.
                # So we can precompute M (mp*pn x r).
                
                # Precompute M
                M = np.zeros((mp * pn, r))
                for c in range(r):
                    M[:, c] = np.outer(V[:, c], W[:, c]).flatten()
                
                # Solve for U
                # T_flat is (nm, mp*pn)
                T_flat = T.reshape(nm, mp * pn)
                # We want T_flat @ M^T? No.
                # t_u = M @ U[u, :]^T
                # So T_flat = M @ U^T
                # U^T = (M^T M)^{-1} M^T T_flat^T
                # U = (M^T M)^{-1} M^T T_flat^T ... wait.
                # T_flat is nm x (mp*pn).
                # M is (mp*pn) x r.
                # We want to find U (nm x r) such that T_flat ≈ M @ U^T?
                # No, T_flat[u, :] = sum_c U[u,c] M[:, c]^T.
                # So T_flat = U @ M^T.
                # U = T_flat @ (M^T M)^{-1} M^T?
                # Least squares: min || T_flat - U M^T ||_F
                # This is equivalent to solving for each row u: min || T_flat[u,:] - U[u,:] M^T ||
                # Which is min || T_flat[u,:] M - U[u,:] (M^T M) ||? No.
                # It's min || T_flat[u,:] - U[u,:] M^T ||
                # Let A = M^T (r x mp*pn).
                # We want U[u,:] A = T_flat[u,:].
                # U[u,:] = T_flat[u,:] A^T (A A^T)^{-1} ?
                # Or simply use np.linalg.lstsq on the transposed system.
                
                try:
                    # Solve U M^T = T_flat
                    # M^T is r x (mp*pn)
                    # T_flat is nm x (mp*pn)
                    # U is nm x r
                    # This is equivalent to (M^T)^T U^T = T_flat^T
                    # M U^T = T_flat^T
                    # U^T = lstsq(M, T_flat^T)
                    U_sol, _, _, _ = np.linalg.lstsq(M, T_flat.T, rcond=None)
                    U = U_sol.T
                except np.linalg.LinAlgError:
                    continue
                
                # Update V
                # Similarly, T_flat_v is (mp, nm*pn)
                # T_flat_v[v, u*pn + w] = T[u, v, w]
                T_flat_v = np.zeros((mp, nm * pn))
                for v in range(mp):
                    for u in range(nm):
                        for w in range(pn):
                            T_flat_v[v, u * pn + w] = T[u, v, w]
                
                M_v = np.zeros((nm * pn, r))
                for c in range(r):
                    M_v[:, c] = np.outer(U[:, c], W[:, c]).flatten()
                
                try:
                    V_sol, _, _, _ = np.linalg.lstsq(M_v, T_flat_v.T, rcond=None)
                    V = V_sol.T
                except np.linalg.LinAlgError:
                    continue
                
                # Update W
                T_flat_w = np.zeros((pn, nm * mp))
                for w in range(pn):
                    for u in range(nm):
                        for v in range(mp):
                            T_flat_w[w, u * mp + v] = T[u, v, w]
                
                M_w = np.zeros((nm * mp, r))
                for c in range(r):
                    M_w[:, c] = np.outer(U[:, c], V[:, c]).flatten()
                
                try:
                    W_sol, _, _, _ = np.linalg.lstsq(M_w, T_flat_w.T, rcond=None)
                    W = W_sol.T
                except np.linalg.LinAlgError:
                    continue
            
            # Round U, V, W to half-integers
            U_r = np.round(U * 2.0) / 2.0
            V_r = np.round(V * 2.0) / 2.0
            W_r = np.round(W * 2.0) / 2.0
            
            # Check if rounding was close
            if np.max(np.abs(U - U_r)) > 0.01 or np.max(np.abs(V - V_r)) > 0.01 or np.max(np.abs(W - W_r)) > 0.01:
                continue
            
            # Verify
            if _verify_np(U_r, V_r, W_r, n, m, p):
                # Convert to fractions for exact check
                U_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in U_r]
                V_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in V_r]
                W_frac = [[Fraction(x).limit_denominator(100) for x in row] for row in W_r]
                
                if _verify_exact(U_frac, V_frac, W_frac, n, m, p):
                    return {
                        "U": [[float(x) for x in row] for row in U_frac],
                        "V": [[float(x) for x in row] for row in V_frac],
                        "W": [[float(x) for x in row] for row in W_frac]
                    }
        
        return None

    best_result = None
    best_rank = n * m * p
    
    # Try ranks from lower bound up
    for r in range(35, 105):
        if time.time() > deadline:
            break
        
        result = try_als(r, num_iters=10, num_restarts=3)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    if best_result is not None:
        return best_result
    
    return schoolbook()
