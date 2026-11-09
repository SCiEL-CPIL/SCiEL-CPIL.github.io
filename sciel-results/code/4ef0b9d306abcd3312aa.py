<<<<<<< SEARCH
DIMENSIONS = [3, 3, 7]
import time
import numpy as np

_START_TIME = time.time()
TIME_LIMIT = 140.0

def _check_time():
    return (time.time() - _START_TIME) < TIME_LIMIT
=======
DIMENSIONS = [3, 3, 7]
import time
import numpy as np
from itertools import product

_START_TIME = time.time()
TIME_LIMIT = 130.0  # Safety margin before 150s kill

def _check_time():
    return (time.time() - _START_TIME) < TIME_LIMIT
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build target tensor T[a, b, c] where:
    # a = i*m + j (A index), b = j'*p + k (B index), c = k'*n + i' (output index)
    # T[a, b, c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for same i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find lower rank using iterative optimization
    best_rank = rank_sb
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    
    # Try various target ranks
    for target_rank in [40, 45, 50, 55, 60]:
        if not _check_time():
            break
        
        found = False
        for attempt in range(3):
            if not _check_time():
                break
            
            np.random.seed(123 + target_rank * 10 + attempt)
            U = np.random.randn(nm, target_rank) * 0.5
            V = np.random.randn(mp, target_rank) * 0.5
            W = np.random.randn(pn, target_rank) * 0.5
            
            # ALS-style optimization
            for it in range(100):
                if not _check_time():
                    break
                
                # Update U: for each row a, solve linear system
                for a in range(nm):
                    # Minimize sum_{b,c} (T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r])^2
                    # = sum_{b,c} (T[a,b,c] - U[a,:] @ (V[b,:]*W[c,:]))^2
                    # Let M[a][bc, r] = V[b,r] * W[c,r], T_a[bc] = T[a,b,c]
                    # Solve: M[a] @ U[a,:] = T_a
                    M_a = np.zeros((mp * pn, target_rank))
                    T_a = np.zeros(mp * pn)
                    idx = 0
                    for b in range(mp):
                        for c in range(pn):
                            M_a[idx, :] = V[b, :] * W[c, :]
                            T_a[idx] = T[a, b, c]
                            idx += 1
                    # Solve least squares
                    U[a, :], residuals, rank_a, sv = np.linalg.lstsq(M_a, T_a, rcond=None)
                
                # Update V similarly
                for b in range(mp):
                    M_b = np.zeros((nm * pn, target_rank))
                    T_b = np.zeros(nm * pn)
                    idx = 0
                    for a in range(nm):
                        for c in range(pn):
                            M_b[idx, :] = U[a, :] * W[c, :]
                            T_b[idx] = T[a, b, c]
                            idx += 1
                    V[b, :], residuals, rank_b, sv = np.linalg.lstsq(M_b, T_b, rcond=None)
                
                # Update W similarly
                for c in range(pn):
                    M_c = np.zeros((nm * mp, target_rank))
                    T_c = np.zeros(nm * mp)
                    idx = 0
                    for a in range(nm):
                        for b in range(mp):
                            M_c[idx, :] = U[a, :] * V[b, :]
                            T_c[idx] = T[a, b, c]
                            idx += 1
                    W[c, :], residuals, rank_c, sv = np.linalg.lstsq(M_c, T_c, rcond=None)
            
            # Check if solution is good
            error = 0.0
            for a in range(nm):
                for b in range(mp):
                    for c in range(pn):
                        val = np.dot(U[a, :], V[b, :] * W[c, :])
                        error += (val - T[a, b, c]) ** 2
            
            if error < 1e-6:
                # Round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                err_r = 0.0
                for a in range(nm):
                    for b in range(mp):
                        for c in range(pn):
                            val = np.dot(U_r[a, :], V_r[b, :] * W_r[c, :])
                            err_r += (val - T[a, b, c]) ** 2
                
                if err_r < 1e-6:
                    # Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no good solution found for this rank, continue
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
=======
def construct():
    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    
    # Build target tensor T[a, b, c] where:
    # a = i*m + j (A index), b = j'*p + k (B index), c = k'*n + i' (output index)
    # T[a, b, c] = 1 iff a=(i,j), b=(j,k), c=(k,i) for same i,j,k
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Schoolbook decomposition
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(nm)]
    V_sb = [[0] * rank_sb for _ in range(mp)]
    W_sb = [[0] * rank_sb for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Precompute the tensor in a more efficient way for verification
    # T_flat[a, b, c] = 1 if (a//m, a%m, b%p, b//p, c%n, c//n) has matching indices
    
    def _verify_decomposition(U, V, W, T, tol=1e-8):
        """Verify that U, V, W exactly decompose T."""
        nm, r = U.shape
        mp, _ = V.shape
        pn, _ = W.shape
        # Compute tensor product: sum_r U[a,r] * V[b,r] * W[c,r]
        # This equals T[a,b,c] for all a,b,c
        # Vectorized: T_comp[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # = (U @ diag(V[b,:]) @ W.T)[c] ... not quite
        # Better: for each r, outer product U[:,r] outer V[:,r] outer W[:,r]
        # T_comp = sum_r np.einsum('ar,br,cr->abc', U, V, W)
        T_comp = np.zeros((nm, mp, pn), dtype=np.float64)
        for r in range(r):
            T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        err = np.max(np.abs(T_comp - T))
        return err < tol
    
    def _try_rank(target_rank, seed, max_iters=200):
        """Try to find a rank-`target_rank` decomposition using ALS."""
        if not _check_time():
            return None
        
        np.random.seed(seed)
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Precompute index arrays for faster access
        # T is sparse: only 63 non-zero entries
        nz_a, nz_b, nz_c = [], [], []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    nz_a.append(i * m + j)
                    nz_b.append(j * p + k)
                    nz_c.append(k * n + i)
        nz_a = np.array(nz_a)
        nz_b = np.array(nz_b)
        nz_c = np.array(nz_c)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(max_iters):
            if not _check_time():
                break
            
            # Update U: for each row a, solve least squares
            for a in range(nm):
                # M_a[bc, r] = V[b,r] * W[c,r]
                # T_a[bc] = T[a,b,c]
                # We only need to consider all (b,c) pairs
                M_a = np.einsum('br,cr->bcr', V, W).reshape(mp * pn, target_rank)
                T_a = T[a].reshape(mp * pn)
                try:
                    U[a, :], _, _, _ = np.linalg.lstsq(M_a, T_a, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Update V
            for b in range(mp):
                M_b = np.einsum('ar,cr->acr', U, W).reshape(nm * pn, target_rank)
                T_b = T[:, b, :].reshape(nm * pn)
                try:
                    V[b, :], _, _, _ = np.linalg.lstsq(M_b, T_b, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Update W
            for c in range(pn):
                M_c = np.einsum('ar,br->abr', U, V).reshape(nm * mp, target_rank)
                T_c = T[:, :, c].reshape(nm * mp)
                try:
                    W[c, :], _, _, _ = np.linalg.lstsq(M_c, T_c, rcond=None)
                except np.linalg.LinAlgError:
                    pass
            
            # Check convergence
            T_comp = np.zeros((nm, mp, pn), dtype=np.float64)
            for r in range(target_rank):
                T_comp += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.max(np.abs(T_comp - T))
            
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if err < 1e-6:
                # Try to round to half-integers
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-8):
                    return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        # Try rounding the best found
        if best_err < 0.1:
            U_r = np.round(best_U * 2) / 2
            V_r = np.round(best_V * 2) / 2
            W_r = np.round(best_W * 2) / 2
            if _verify_decomposition(U_r, V_r, W_r, T, tol=1e-8):
                return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}
        
        return None
    
    # Try various target ranks, starting from lower
    for target_rank in [40, 45, 50, 55, 60, 62]:
        if not _check_time():
            break
        
        for attempt in range(5):
            if not _check_time():
                break
            
            seed = 123 + target_rank * 100 + attempt
            result = _try_rank(target_rank, seed, max_iters=150)
            if result is not None:
                return result
    
    # Return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
>>>>>>> REPLACE
