"""Rewritten program for matrix multiplication tensor decomposition 3x3 x 3x5."""
import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    dim_a = n * m
    dim_b = m * p
    dim_c = p * n
    
    # Construct the tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i)
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Try ranks from low to high
    # Start with a reasonable lower bound. Trivial is 45.
    # We search for a decomposition with rank < 45.
    
    best_solution = None
    # Search range: 25 to 44
    for rank in range(25, 45):
        # Random initialization
        np.random.seed(rank) # Deterministic for reproducibility
        U = np.random.randn(dim_a, rank)
        V = np.random.randn(dim_b, rank)
        W = np.random.randn(dim_c, rank)
        
        # ALS (Alternating Least Squares)
        for it in range(50):
            # Update U
            M1 = np.einsum('br,cr->bcr', V, W).reshape(dim_b * dim_c, rank)
            T1 = T.reshape(dim_a, dim_b * dim_c)
            try:
                U = np.linalg.lstsq(M1.T, T1.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                break
            
            # Update V
            M2 = np.einsum('ar,cr->acr', U, W).reshape(dim_a * dim_c, rank)
            T2 = T.transpose(1, 0, 2).reshape(dim_b, dim_a * dim_c)
            try:
                V = np.linalg.lstsq(M2.T, T2.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                break
                
            # Update W
            M3 = np.einsum('ar,br->abr', U, V).reshape(dim_a * dim_b, rank)
            T3 = T.transpose(2, 0, 1).reshape(dim_c, dim_a * dim_b)
            try:
                W = np.linalg.lstsq(M3.T, T3.T, rcond=None)[0].T
            except np.linalg.LinAlgError:
                break

        # Check convergence and exactness
        # Reconstruct
        T_recon = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Round to nearest half-integer
        U_rounded = np.round(U * 2) / 2
        V_rounded = np.round(V * 2) / 2
        W_rounded = np.round(W * 2) / 2
        
        T_recon_rounded = np.einsum('ar,br,cr->abc', U_rounded, V_rounded, W_rounded)
        
        if np.allclose(T_recon_rounded, T, atol=1e-9):
            best_solution = {
                "U": U_rounded.tolist(),
                "V": V_rounded.tolist(),
                "W": W_rounded.tolist()
            }
            break
            
    if best_solution is None:
        # Fallback to trivial decomposition if search fails
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.float64)
        V = np.zeros((dim_b, rank), dtype=np.float64)
        W = np.zeros((dim_c, rank), dtype=np.float64)
        
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        best_solution = {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }

    return best_solution
