"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank = 32
    schoolbook_rank = n * m * p
    deadline = time.time() + 140  # safety margin before 150s kill

    # First, try to find a rank-27 decomposition via randomized algebraic search
    # The tensor T has entries T[i,j,k] = 1 if (i,j) in A, (j,k) in B, (k,i) in C
    # For rank r: T = sum_{t=1}^{r} u_t ⊗ v_t ⊗ w_t
    # where u_t is in R^{n*m}, v_t in R^{m*p}, w_t in R^{p*n}
    
    # Strategy: Use the fact that for matrix multiplication, we can exploit
    # the structure. Try r=27 with randomized initialization and local refinement.
    
    def build_schoolbook():
        """Return schoolbook decomposition of rank 32."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    # Build the tensor as a 3D array for verification
    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1.0
        return T

    T = build_tensor()

    def verify_decomposition(U, V, W, tol=1e-8):
        """Check if U, V, W form an exact decomposition of T."""
        r = U.shape[1]
        # Reconstruct tensor: T_recon[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
        # Use einsum for efficiency
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def try_random_decomposition(r, n_iters=5000, seed=42):
        """Try to find rank-r decomposition via randomized search with Newton refinement."""
        rng = np.random.RandomState(seed)
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        best_err = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for it in range(n_iters):
            if time.time() > deadline:
                break
            # Compute gradient of ||sum_t u_t v_t w_t - T||^2 w.r.t. U, V, W
            # F = sum_t U[:,t] outer V[:,t] outer W[:,t] - T
            # dF/du = sum_t v_t outer w_t (element-wise product in b,c dims)
            # Actually: F[a,b,c] = sum_t U[a,t]*V[b,t]*W[c,t]
            # dF/da U[a,t] = V[b,t]*W[c,t]
            # So gradient of 0.5*||F||^2 w.r.t. U[a,t] = sum_{b,c} (F[a,b,c]) * V[b,t] * W[c,t]
            
            F = np.einsum('at,bt,ct->abc', U, V, W) - T
            # grad_U[a,t] = sum_{b,c} F[a,b,c] * V[b,t] * W[c,t]
            grad_U = np.einsum('abc,bt,ct->at', F, V, W)
            grad_V = np.einsum('abc,at,ct->bt', F, U, W)
            grad_W = np.einsum('abc,at,bt->ct', F, U, V)
            
            # Normalize gradients for stable steps
            norm_U = np.linalg.norm(grad_U) + 1e-12
            norm_V = np.linalg.norm(grad_V) + 1e-12
            norm_W = np.linalg.norm(grad_W) + 1e-12
            
            # Adaptive learning rate
            err = np.sum(F**2)
            lr = 0.01 / (1.0 + err)
            
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
            
            # Periodically try to zero out small entries to get integer/half-integer
            if it % 500 == 499:
                current_err = np.max(np.abs(np.einsum('at,bt,ct->abc', U, V, W) - T))
                if current_err < best_err:
                    best_err = current_err
                    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                    
                # Try to snap to half-integers
                U_snap = np.round(U * 2) / 2
                V_snap = np.round(V * 2) / 2
                W_snap = np.round(W * 2) / 2
                snap_err = np.max(np.abs(np.einsum('at,bt,ct->abc', U_snap, V_snap, W_snap) - T))
                if snap_err < 1e-6:
                    return U_snap, V_snap, W_snap, snap_err
                if snap_err < best_err:
                    best_err = snap_err
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
        
        final_err = np.max(np.abs(np.einsum('at,bt,ct->abc', best_U, best_V, best_W) - T))
        return best_U, best_V, best_W, final_err

    # Try rank 27 first (known to be achievable for 2x4x4 based on literature)
    for r in [27, 28]:
        if time.time() > deadline:
            break
        U, V, W, err = try_random_decomposition(r, n_iters=3000, seed=42)
        if err < 1e-8:
            # Found valid decomposition, convert to list format
            U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
            V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
            W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
            return {"U": U_list, "V": V_list, "W": W_list}

    # Fallback to schoolbook
    U, V, W = build_schoolbook()
    U_list = [[float(U[i, t]) for t in range(schoolbook_rank)] for i in range(n * m)]
    V_list = [[float(V[i, t]) for t in range(schoolbook_rank)] for i in range(m * p)]
    W_list = [[float(W[i, t]) for t in range(schoolbook_rank)] for i in range(p * n)]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
