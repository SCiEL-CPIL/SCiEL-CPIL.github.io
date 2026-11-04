"""Matrix multiplication 3x4x6 - ALS with structured initialization."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
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


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify the decomposition: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol


def _als_decompose(n, m, p, r, max_sweeps=500, tol=1e-10, seed=42):
    """
    Alternating Least Squares for rank-r decomposition of the matrix multiplication tensor.
    
    We want: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    This is equivalent to: the tensor T (all ones, shape (n,m,p)) = sum_c u_c ⊗ v_c ⊗ w_c
    where u_c is indexed by (i,j), v_c by (j,k), w_c by (k,i).
    
    ALS: fix two factors, solve for the third via linear least squares.
    """
    np.random.seed(seed)
    
    # Initialize
    U = np.random.randn(n * m, r) * 0.5
    V = np.random.randn(m * p, r) * 0.5
    W = np.random.randn(p * n, r) * 0.5
    
    Ut = U.reshape(n, m, r)
    Vt = V.reshape(m, p, r)
    Wt = W.reshape(p, n, r)
    
    prev_loss = np.inf
    
    for sweep in range(max_sweeps):
        # --- Update U ---
        # For each (i,j), we want: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is a linear system: for fixed (i,j), define for each k:
        #   sum_c U[i*m+j,c] * B_{k,c} = 1
        # where B_{k,c} = V[j*p+k,c] * W[k*n+i,c]
        # So: U[i*m+j,:] @ B^T = ones(p)  =>  U[i*m+j,:] = (B^T B)^{-1} B^T ones(p)
        
        new_Ut = np.zeros_like(Ut)
        for i in range(n):
            for j in range(m):
                # B[k,c] = V[j*p+k,c] * W[k*n+i,c] for k in range(p)
                B = V[j*p:(j+1)*p, :] * W[:, i, :]  # shape (p, r)
                # Solve: U[i,j,:] @ B^T = ones(p)
                # i.e., B @ U[i,j,:]^T = ones(p)  ... wait
                # sum_c U[i*m+j,c] * B[k,c] = 1 for each k
                # So B @ U[i,j,:]^T = ones(p)
                # U[i,j,:]^T = (B^T B)^{-1} B^T ones(p)
                try:
                    Ut_ij = np.linalg.lstsq(B, np.ones(p), rcond=None)[0]
                    new_Ut[i, j, :] = Ut_ij
                except:
                    new_Ut[i, j, :] = Ut[i, j, :]
        
        U = new_Ut.reshape(n * m, r)
        Ut = U.reshape(n, m, r)
        
        # --- Update V ---
        # For each (j,k), we want: sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i
        # A[i,c] = U[i*m+j,c] * W[k*n+i,c] for i in range(n)
        # A @ V[j,k,:]^T = ones(n)
        
        new_Vt = np.zeros_like(Vt)
        for j in range(m):
            for k in range(p):
                A = U[:, j, :] * W[k, :, :]  # shape (n, r)
                try:
                    Vt_jk = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                    new_Vt[j, k, :] = Vt_jk
                except:
                    new_Vt[j, k, :] = Vt[j, k, :]
        
        V = new_Vt.reshape(m * p, r)
        Vt = V.reshape(m, p, r)
        
        # --- Update W ---
        # For each (k,i), we want: sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        # C[j,c] = U[i*m+j,c] * V[j*p+k,c] for j in range(m)
        # C @ W[k,i,:]^T = ones(m)
        
        new_Wt = np.zeros_like(Wt)
        for k in range(p):
            for i in range(n):
                C = U[i, :, :] * V[:, k, :]  # shape (m, r)
                try:
                    Wt_ki = np.linalg.lstsq(C, np.ones(m), rcond=None)[0]
                    new_Wt[k, i, :] = Wt_ki
                except:
                    new_Wt[k, i, :] = Wt[k, i, :]
        
        W = new_Wt.reshape(p * n, r)
        Wt = W.reshape(p, n, r)
        
        # Compute loss
        T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
        loss = np.sum((T - 1.0) ** 2)
        
        if loss < tol:
            break
        
        if abs(prev_loss - loss) < 1e-15:
            break
        
        prev_loss = loss
    
    # Final check
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    if np.max(np.abs(T - 1.0)) < 1e-5:
        return U, V, W
    return None


def _structured_init(n, m, p, r, seed=0):
    """
    Try structured initializations based on the algebraic structure.
    
    Idea: The matrix multiplication tensor has a natural structure. 
    For rank r, we can try to initialize based on partitions of the index sets.
    """
    np.random.seed(seed)
    
    # Strategy 1: Block diagonal-like structure
    # Divide the indices into r groups and assign each group to one term
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # For each term c, assign a subset of (i,j) to U, (j,k) to V, (k,i) to W
    # Simple approach: random but with some structure
    
    # Strategy: Use random initialization with scaling
    U = np.random.randn(n * m, r) * np.sqrt(1.0 / r)
    V = np.random.randn(m * p, r) * np.sqrt(1.0 / r)
    W = np.random.randn(p * n, r) * np.sqrt(1.0 / r)
    
    return U, V, W


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    # Try ALS for various ranks, from low to high
    # Use multiple random seeds for robustness
    
    max_seeds_per_rank = 5
    
    for r in range(40, 72):
        if time.time() > deadline - 10:
            break
        
        found = False
        for seed in range(max_seeds_per_rank):
            if time.time() > deadline - 5:
                break
            
            # Structured initialization
            U_init, V_init, W_init = _structured_init(n, m, p, r, seed=seed * 1000 + 42)
            
            # Run ALS
            result = _als_decompose(n, m, p, r, max_sweeps=300, tol=1e-10, seed=seed * 1000 + 42)
            
            if result is not None:
                U, V, W = result
                if _verify(U, V, W, n, m, p):
                    best = {'U': U, 'V': V, 'W': W}
                    best_rank = r
                    found = True
                    break
        
        if found:
            break
    
    # If we found a better solution, use it
    if best_rank < n * m * p:
        return {'U': best['U'].tolist(), 'V': best['V'].tolist(), 'W': best['W'].tolist()}
    else:
        # Return trivial decomposition
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
