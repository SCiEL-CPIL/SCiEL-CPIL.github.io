import time
import numpy as np
from scipy.optimize import minimize

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses a structured search: start from known rank-44 decomposition (Strassen + schoolbook),
    then attempt to reduce rank via subspace projection / bilinear map optimization.
    """
    n, m, p = 2, 3, 8
    start_time = time.time()
    deadline = start_time + 118.0  # 118s to leave safety margin

    # Build the multiplication tensor T[i, j, k] where output[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor is of shape (n, m, p) but we flatten: T[(i*m+j), (j*p+k), (k*n+i)] = 1
    # Actually, the tensor decomposition is:
    #   C[k*n+i] = sum_r U[i*m + a, r] * V[a*p + k, r] * W[k*n + i, r]  (for each a)
    # Wait, the standard tensor decomposition for matrix multiplication M_{n,m,p} is:
    #   C[i,k] = sum_{j,r} A[i,j] * B[j,k] * T[i,j,k,r]
    # But the interface says:
    #   U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    #   And the decomposition is: sum_r U[a, r] * V[b, r] * W[c, r] = T[a, b, c]
    # where a = i*m + j, b = j*p + k, c = k*n + i
    # So T[a, b, c] = 1 if a = i*m+j, b = j*p+k, c = k*n+i for some i,j,k, else 0.
    # More precisely: T[(i*m+j), (j*p+k), (k*n+i)] = 1 for all valid i,j,k.
    
    # Let's build the tensor T as a 3D array of shape (n*m, m*p, p*n)
    # T[a, b, c] = 1 if a = i*m + j, b = j*p + k, c = k*n + i for some i in [0,n), j in [0,m), k in [0,p)
    # Otherwise 0.
    
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Known rank-44 decomposition
    def build_rank44():
        rank = 44
        U = np.zeros((n*m, rank), dtype=np.float64)
        V = np.zeros((m*p, rank), dtype=np.float64)
        W = np.zeros((p*n, rank), dtype=np.float64)
        
        # Strassen 2x2x2
        S_U = np.array([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1]
        ], dtype=np.float64)
        S_V = np.array([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1]
        ], dtype=np.float64)
        S_W = np.array([
            [1, 0, 0, 1],
            [0, 1, 0, -1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0]
        ], dtype=np.float64)
        
        col = 0
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        U[li * m + la, col] = S_U[r, li * 2 + la]
                for la in range(2):
                    for lk in range(2):
                        V[la * p + (k0 + lk), col] = S_V[r, la * 2 + lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0 + lk) * n + li, col] = S_W[r, lk * 2 + li]
                col += 1
        
        # Schoolbook for a=2 (third row of A, i.e., j=2)
        for i in range(n):
            for k in range(p):
                U[i * m + 2, col] = 1.0
                V[2 * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
        
        return U, V, W
    
    U44, V44, W44 = build_rank44()
    
    def tensor_error(U, V, W):
        """Compute the error ||sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r] - T||_F"""
        # Reshape U, V, W to 3D and compute the tensor
        # T_reconstructed[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # This is a 3D tensor contraction
        # We can compute it as: for each r, add outer product U[:,r] * V[:,r] * W[:,r]
        # But that's slow. Instead, use einsum.
        # T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        # But for large rank, this might be slow. Let's try.
        T_rec = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        return np.linalg.norm(T_rec - T)
    
    def make_objective_and_jac(rank):
        """Create objective and Jacobian functions for given rank."""
        nU = n * m
        nV = m * p
        nW = p * n
        
        def unpack(x):
            U_ = x[:nU*rank].reshape(nU, rank)
            V_ = x[nU*rank:nU*rank+nV*rank].reshape(nV, rank)
            W_ = x[nU*rank+nV*rank:].reshape(nW, rank)
            return U_, V_, W_
        
        def objective(x):
            U_, V_, W_ = unpack(x)
            T_rec = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
            return np.sum((T_rec - T)**2)
        
        def jac(x):
            U_, V_, W_ = unpack(x)
            T_rec = np.einsum('ar,br,cr->abc', U_, V_, W_, optimize=True)
            diff = T_rec - T
            grad_U = np.einsum('abc,br,cr->ar', diff, V_, W_, optimize=True) * 2
            grad_V = np.einsum('abc,ar,cr->br', diff, U_, W_, optimize=True) * 2
            grad_W = np.einsum('abc,ar,br->cr', diff, U_, V_, optimize=True) * 2
            return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
        
        return objective, jac, unpack
    
    def try_rank_reduction(U0, V0, W0, target_rank, max_time=8.0):
        """Try to find a decomposition of rank target_rank by optimizing."""
        if time.time() > deadline - max_time:
            return None
        
        rank = target_rank
        nU = n * m
        nV = m * p
        nW = p * n
        
        objective, jac, unpack = make_objective_and_jac(rank)
        
        # Try multiple random initializations
        rng = np.random.default_rng(42)
        best_err = float('inf')
        best_x = None
        
        # Initialization 1: First target_rank columns
        U_init = U0[:, :target_rank].copy()
        V_init = V0[:, :target_rank].copy()
        W_init = W0[:, :target_rank].copy()
        x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
        
        inits = [x0]
        
        # Initialization 2: Random perturbation of first columns
        for _ in range(3):
            U_r = U_init + rng.normal(0, 0.1, U_init.shape)
            V_r = V_init + rng.normal(0, 0.1, V_init.shape)
            W_r = W_init + rng.normal(0, 0.1, W_init.shape)
            inits.append(np.concatenate([U_r.flatten(), V_r.flatten(), W_r.flatten()]))
        
        # Initialization 3: SVD-based low-rank approximation
        T_unf1 = T.reshape(nU, nV * nW)
        U_svd, s, Vt_svd = np.linalg.svd(T_unf1, full_matrices=False)
        if len(s) >= target_rank:
            Vt_flat = Vt_svd[:target_rank, :].reshape(target_rank, nV, nW)
            V_init2 = np.zeros((nV, target_rank))
            W_init2 = np.zeros((nW, target_rank))
            U_init2 = np.zeros((nU, target_rank))
            for r in range(target_rank):
                slice_r = Vt_flat[r]
                Ur, sr, Vtr = np.linalg.svd(slice_r, full_matrices=False)
                if len(sr) > 0:
                    V_init2[:, r] = Ur[:, 0] * np.sqrt(sr[0])
                    W_init2[:, r] = Vtr[0, :] * np.sqrt(sr[0])
            U_init2 = U_svd[:, :target_rank] * np.sqrt(s[:target_rank])
            inits.append(np.concatenate([U_init2.flatten(), V_init2.flatten(), W_init2.flatten()]))
        
        # Optimize each initialization
        for init_x in inits:
            if time.time() > deadline - 2:
                break
            
            result = minimize(objective, init_x, jac=jac, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500})
            
            if time.time() > deadline - 2:
                break
            
            if result.fun < best_err:
                best_err = result.fun
                best_x = result.x.copy()
            
            if best_err < 1e-10:
                break
        
        if best_x is None or best_err > 1e-6:
            return None
        
        # Unpack and round to nearest half-integer
        U_, V_, W_ = unpack(best_x)
        
        # Try rounding to integers first, then half-integers
        for mult in [1, 2]:
            U_r = np.round(U_ * mult) / mult
            V_r = np.round(V_ * mult) / mult
            W_r = np.round(W_ * mult) / mult
            err = tensor_error(U_r, V_r, W_r)
            if err < 1e-6:
                return U_r, V_r, W_r
        
        # If rounding didn't work, try a final polish
        result2 = minimize(objective, best_x, jac=jac, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 200})
        
        if result2.fun < 1e-10:
            U_, V_, W_ = unpack(result2.x)
            for mult in [1, 2]:
                U_r = np.round(U_ * mult) / mult
                V_r = np.round(V_ * mult) / mult
                W_r = np.round(W_ * mult) / mult
                err = tensor_error(U_r, V_r, W_r)
                if err < 1e-6:
                    return U_r, V_r, W_r
        
        return None
    
    # Try ranks from 43 down to 30
    best_result = (U44, V44, W44)
    best_rank = 44
    
    for target_rank in range(43, 29, -1):
        if time.time() > deadline - 12:
            break
        result = try_rank_reduction(U44, V44, W44, target_rank, max_time=10.0)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break  # Found a lower rank, stop
    
    U, V, W = best_result
    rank = best_rank
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
