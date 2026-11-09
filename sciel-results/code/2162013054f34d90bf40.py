"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the full multiplication tensor T of shape (nm, mp, pn)
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if U, V, W exactly reconstruct T."""
        r = U.shape[1]
        # Efficient check: T = sum_r U[:,r] outer V[:,r] outer W[:,r]
        # Use tensordot
        recon = np.tensordot(U, V, axes=([1], [1]))  # (nm, mp, r)
        recon = np.tensordot(recon, W, axes=([2], [1]))  # (nm, mp, pn)
        diff = np.abs(recon - T)
        return np.max(diff) < tol
    
    def to_rational_matrix(A, tol=1e-10):
        """Convert to half-integers if possible, else return as-is."""
        B = np.round(A * 2) / 2
        if np.max(np.abs(B - A)) < tol:
            return B
        B = np.round(A)
        if np.max(np.abs(B - A)) < tol:
            return B
        return A
    
    def als_decompose(T, rank, max_iter=50, tol=1e-12, seed=42):
        """Alternating least squares for tensor decomposition."""
        np.random.seed(seed)
        nm, mp, pn = T.shape
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.1
        V = np.random.randn(mp, rank) * 0.1
        W = np.random.randn(pn, rank) * 0.1
        
        for it in range(max_iter):
            # Fix V, W; solve for U
            M = np.outer(V, W).reshape(rank, mp * pn)
            T_flat = T.reshape(nm, mp * pn)
            U_new = T_flat @ np.linalg.pinv(M)
            
            # Fix U, W; solve for V
            M2 = np.outer(U, W).reshape(rank, nm * pn)
            T_flat2 = np.transpose(T, (1, 0, 2)).reshape(mp, nm * pn)
            V_new = T_flat2 @ np.linalg.pinv(M2)
            
            # Fix U, V; solve for W
            M3 = np.outer(U, V).reshape(rank, nm * mp)
            T_flat3 = np.transpose(T, (2, 0, 1)).reshape(pn, nm * mp)
            W_new = T_flat3 @ np.linalg.pinv(M3)
            
            # Check convergence
            if (np.max(np.abs(U_new - U)) < tol and
                np.max(np.abs(V_new - V)) < tol and
                np.max(np.abs(W_new - W)) < tol):
                U, V, W = U_new, V_new, W_new
                break
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W
    
    # Theoretical lower bound is max(nm, mp, pn)
    lower_bound = max(nm, mp, pn)
    
    best_rank = nm * mp # 96
    best_U = None
    best_V = None
    best_W = None
    
    # Schoolbook fallback
    U_school = np.zeros((nm, nm * mp))
    V_school = np.zeros((mp, nm * mp))
    W_school = np.zeros((pn, nm * mp))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    # Try to find a decomposition with rank close to the lower bound
    # We search from lower_bound up to schoolbook rank
    # Given time constraints, we limit the number of seeds and iterations
    for rank in range(lower_bound, nm * mp + 1):
        found = False
        # Use a few random seeds
        for seed in range(5):
            try:
                U, V, W = als_decompose(T, rank, max_iter=30, seed=seed)
                if check_decomposition(U, V, W):
                    # Try to convert to half-integers
                    U_r = to_rational_matrix(U)
                    V_r = to_rational_matrix(V)
                    W_r = to_rational_matrix(W)
                    if check_decomposition(U_r, V_r, W_r):
                        best_rank = rank
                        best_U = U_r
                        best_V = V_r
                        best_W = W_r
                        found = True
                        break
            except Exception:
                continue
        if found:
            break
            
    if best_U is not None:
        U_list = [[float(x) for x in row] for row in best_U]
        V_list = [[float(x) for x in row] for row in best_V]
        W_list = [[float(x) for x in row] for row in best_W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        U_list = [[float(x) for x in row] for row in U_school]
        V_list = [[float(x) for x in row] for row in V_school]
        W_list = [[float(x) for x in row] for row in W_school]
        return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
