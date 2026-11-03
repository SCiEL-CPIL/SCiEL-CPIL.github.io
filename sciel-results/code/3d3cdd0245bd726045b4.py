import numpy as np
from scipy.optimize import minimize

def construct():
    n, m, p = 2, 4, 6
    rank = 40
    
    # Build the tensor T
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                T[a, b, c] = 1
    
    # Initialize U, V, W
    np.random.seed(42)
    U = np.random.randn(n*m, rank)
    V = np.random.randn(m*p, rank)
    W = np.random.randn(p*n, rank)
    
    # Alternating least squares
    for it in range(100):
        # Solve for U: T[:,:,c] = U @ (V*W[:,c])... 
        # Actually, the equation is: T[a,b,c] = sum_alpha U[a,alpha]*V[b,alpha]*W[c,alpha]
        # For fixed V,W: T[a,b,c] = sum_alpha U[a,alpha] * (V[b,alpha]*W[c,alpha])
        # This is: T_vec = U @ M where M[b,c,alpha] = V[b,alpha]*W[c,alpha]
        # Reshape: T is (nm, mp, pn), we want T[a,b,c] = sum_alpha U[a,alpha]*M[b,c,alpha]
        # This is a linear system in U for each (a).
        
        # For each a: T[a,:,:] = sum_alpha U[a,alpha] * (V[:,alpha]*W[:,alpha])
        # Let M = V * W (broadcasting over alpha): M[b,c,alpha] = V[b,alpha]*W[c,alpha]
        # T[a,b,c] = sum_alpha U[a,alpha] * M[b,c,alpha]
        # Reshape T[a] to (mp*pn,), M to (mp*pn, rank)
        # Then T[a] = U[a] @ M_flat
        
        M = V[:, :, None] * W[None, :, :]  # (mp, pn, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, m*p*p*n)  # (rank, mp*pn)
        T_flat = T.reshape(n*m, m*p*p*n)  # (nm, mp*pn)
        
        # T_flat = U @ M_flat.T → U = T_flat @ M_flat^{-1}... 
        # Actually: T_flat[a, :] = U[a, :] @ M_flat.T
        # So: U = T_flat @ pinv(M_flat.T)
        
        U = T_flat @ np.linalg.pinv(M_flat.T)
        
        # Similarly for V and W
        # T[a,b,c] = sum_alpha V[b,alpha] * (U[a,alpha]*W[c,alpha])
        M = U[:, :, None] * W[None, :, :]  # (nm, pn, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, n*m*p*n)
        T_flat = T.transpose(1, 0, 2).reshape(m*p, n*m*p*n)
        V = T_flat @ np.linalg.pinv(M_flat.T)
        
        M = U[:, :, None] * V[None, :, :]  # (nm, mp, rank)
        M_flat = M.transpose(2, 0, 1).reshape(rank, n*m*m*p)
        T_flat = T.transpose(2, 0, 1).reshape(p*n, n*m*m*p)
        W = T_flat @ np.linalg.pinv(M_flat.T)
    
    # Check residual
    residual = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                residual += (val - 1)**2
    
    # Check off-diagonal
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                i, j = divmod(a, m)
                j2, k = divmod(b, p)
                k2, i2 = divmod(c, n)
                if j != j2 or i != i2 or k != k2:
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    residual += val**2
    
    # Round to nearest half-integer
    U = np.round(U * 2) / 2
    V = np.round(V * 2) / 2
    W = np.round(W * 2) / 2
    
    # Convert to lists
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
