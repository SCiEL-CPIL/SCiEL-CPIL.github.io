"""Matrix multiplication 2,3,6 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Uses a structured Kronecker-product approach exploiting 6 = 3*2,
    then attempts to reduce rank via targeted ALS search.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=6
    
    # Build the multiplication tensor
    T = np.zeros((n*m, m*p, p*n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    Tvec = T.reshape(-1)
    
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s margin
    
    # Helper: build Khatri-Rao matrix for a given pair
    def khatri_rao(A, B, size_a, size_b):
        """Column r is A[:,r] ⊗ B[:,r], shape (size_a*size_b, rank)"""
        rank = A.shape[1]
        KR = np.zeros((size_a * size_b, rank))
        for r in range(rank):
            KR[:, r] = np.kron(A[:, r], B[:, r])
        return KR
    
    # Helper: check if decomposition is exact
    def check_exact(U, V, W, rank, T, tol=1e-8):
        recon = np.zeros(T.shape)
        for r in range(rank):
            recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        return np.max(np.abs(recon - T)) < tol
    
    # Helper: ALS for given rank, return (U,V,W,converged)
    def als_decompose(rank, n_init=30, max_iter=100, seed_base=42):
        best_U = best_V = best_W = None
        for trial in range(n_init):
            if time.time() > deadline:
                break
            
            rng = np.random.RandomState(seed_base + trial)
            U = rng.randn(n*m, rank)
            V = rng.randn(m*p, rank)
            W = rng.randn(p*n, rank)
            
            # Normalize columns
            for r in range(rank):
                Un = np.linalg.norm(U[:, r])
                Vn = np.linalg.norm(V[:, r])
                Wn = np.linalg.norm(W[:, r])
                if Un > 1e-12:
                    U[:, r] /= Un
                if Vn > 1e-12:
                    V[:, r] /= Vn
                if Wn > 1e-12:
                    W[:, r] /= Wn
            
            prev_err = np.inf
            for iteration in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Solve for U
                VkrW = khatri_rao(V, W, m*p, p*n)
                try:
                    U_new, _, _, _ = np.linalg.lstsq(VkrW, Tvec, rcond=None)
                except:
                    break
                U = U_new.reshape(n*m, rank)
                
                # Solve for V
                WkrU = khatri_rao(W, U, p*n, n*m)
                try:
                    V_new, _, _, _ = np.linalg.lstsq(WkrU, Tvec, rcond=None)
                except:
                    break
                V = V_new.reshape(m*p, rank)
                
                # Solve for W
                UkrV = khatri_rao(U, V, n*m, m*p)
                try:
                    W_new, _, _, _ = np.linalg.lstsq(UkrV, Tvec, rcond=None)
                except:
                    break
                W = W_new.reshape(p*n, rank)
                
                # Check convergence
                recon = np.zeros(T.shape)
                for r in range(rank):
                    recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
                err = np.max(np.abs(recon - T))
                
                if err < 1e-10:
                    return U, V, W, True
                
                if abs(prev_err - err) < 1e-14 and iteration > 10:
                    break
                prev_err = err
        
        # Check best result from last trial
        if U is not None and check_exact(U, V, W, rank, T):
            return U, V, W, True
        return None, None, None, False
    
    # Try ranks from lower bound upward
    # Lower bound from matricization ranks
    T0 = T.reshape(n*m, m*p * p*n)
    T1 = T.reshape(m*p, n*m * p*n)
    T2 = T.reshape(p*n, n*m * m*p)
    
    r0 = np.linalg.matrix_rank(T0, tol=1e-10)
    r1 = np.linalg.matrix_rank(T1, tol=1e-10)
    r2 = np.linalg.matrix_rank(T2, tol=1e-10)
    lower_bound = max(r0, r1, r2)
    
    schoolbook = n * m * p  # 36
    
    best_rank = schoolbook
    best_U = best_V = best_W = None
    
    # Try ranks from lower_bound to schoolbook
    for rank in range(lower_bound, schoolbook + 1):
        if time.time() > deadline:
            break
        
        # Reduce number of trials for higher ranks (less likely to converge)
        n_trials = max(10, 50 - (rank - lower_bound) * 2)
        
        U, V, W, converged = als_decompose(rank, n_init=n_trials, max_iter=80)
        
        if converged:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
            break
    
    # If no better decomposition found, use schoolbook
    if best_U is None:
        rank = schoolbook
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Round to nearest integers or half-integers if possible
        U_rounded = np.round(best_U * 2) / 2
        V_rounded = np.round(best_V * 2) / 2
        W_rounded = np.round(best_W * 2) / 2
        
        # Verify the rounded version
        recon = np.zeros((n*m, m*p, p*n))
        for r in range(best_rank):
            recon += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
        
        if np.max(np.abs(recon - T)) < 1e-8:
            return {"U": U_rounded.tolist(), "V": V_rounded.tolist(), "W": W_rounded.tolist()}
        else:
            return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
