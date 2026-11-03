import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS  # n=3, m=4, p=5
    # Tensor T has shape (n*m, m*p, p*n) = (12, 20, 15)
    # T[(i*m+j), (j'*p+k), (k'*n+i')] = 1 if j==j', i==i', k==k' else 0
    
    # Build the tensor
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Try CP decomposition for various ranks
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # schoolbook
    
    # Schoolbook as fallback
    U_s = np.zeros((n*m, best_rank))
    V_s = np.zeros((m*p, best_rank))
    W_s = np.zeros((p*n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i*m+j, col] = 1.0
                V_s[j*p+k, col] = 1.0
                W_s[k*n+i, col] = 1.0
                col += 1
    
    # Try to find lower rank via numerical optimization
    # Use a simple ALS (Alternating Least Squares) for CP decomposition
    def cp_decompose(T, rank, max_iter=100, tol=1e-8):
        """Simple CP decomposition via ALS."""
        # Initialize randomly
        np.random.seed(42)
        A = np.random.randn(T.shape[0], rank)
        B = np.random.randn(T.shape[1], rank)
        C = np.random.randn(T.shape[2], rank)
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update A
            B_C = B * C  # element-wise, shape (n1, rank) * (n2, rank) -> need outer product
            # T reshaped to (n1, n2*n3), B_C reshaped
            T_mat = T.reshape(T.shape[0], -1)
            BC_mat = (B * C).reshape(B.shape[0], -1)  # wrong, need Kronecker
            # Actually: sum_r A[:,r] * (B[:,r] ⊗ C[:,r])
            # So we need: for each r, B[:,r] ⊗ C[:,r]
            # T_mat = A @ diag(B_C_flattened) ... this is complex
            
            # Simpler: use the fact that T = sum_r A_r ⊗ B_r ⊗ C_r
            # Reshape T to (n1, n2, n3)
            # A update: A = T_mat @ (B ⊙ C)^+ where ⊙ is Khatri-Rao
            B_kr_C = np.kron(B, C)  # This is wrong, Khatri-Rao is column-wise Kronecker
            # Khatri-Rao product: [B C]_kr = [B[:,0]⊗C[:,0], B[:,1]⊗C[:,1], ...]
            B_kr_C = np.zeros((T.shape[1]*T.shape[2], rank))
            for r in range(rank):
                B_kr_C[:, r] = np.kron(B[:, r], C[:, r])
            
            T_mat = T.reshape(T.shape[0], -1)
            A = np.linalg.lstsq(B_kr_C, T_mat, rcond=None)[0].T.T  # A shape (n1, rank)
            A = np.linalg.lstsq(B_kr_C, T_mat.T, rcond=None)[0].T  # (rank, n1) -> transpose
            # Actually: T_mat = A @ B_kr_C.T, so A = T_mat @ B_kr_C^+
            A = np.linalg.lstsq(B_kr_C.T, T_mat.T, rcond=None)[0].T
            
            # Update B
            A_kr_C = np.zeros((T.shape[0]*T.shape[2], rank))
            for r in range(rank):
                A_kr_C[:, r] = np.kron(A[:, r], C[:, r])
            T_mat_B = T.transpose(1, 0, 2).reshape(T.shape[1], -1)
            B = np.linalg.lstsq(A_kr_C.T, T_mat_B.T, rcond=None)[0].T
            
            # Update C
            A_kr_B = np.zeros((T.shape[0]*T.shape[1], rank))
            for r in range(rank):
                A_kr_B[:, r] = np.kron(A[:, r], B[:, r])
            T_mat_C = T.transpose(2, 0, 1).reshape(T.shape[2], -1)
            C = np.linalg.lstsq(A_kr_B.T, T_mat_C.T, rcond=None)[0].T
            
            # Check convergence
            err = np.linalg.norm(T - np.einsum('ir,jr,kr->ijk', A, B, C))
            if abs(prev_err - err) < tol:
                break
            prev_err = err
        
        return A, B, C, err
    
    # Try ranks from lower bound up
    # Lower bound: max of matrix ranks
    # Matricization 1: (n*m) x (m*p*p*n) -> rank <= n*m = 12
    # Matricization 2: (m*p) x (n*m*p*n) -> rank <= m*p = 20  
    # Matricization 3: (p*n) x (n*m*m*p) -> rank <= p*n = 15
    # So rank >= 20
    
    for rank in range(20, 60):
        try:
            A, B, C, err = cp_decompose(T, rank, max_iter=50)
            if err < 1e-6:
                # Check if it can be made exact with rationals
                # Try rounding to nearest 1/2
                A_r = np.round(A * 2) / 2
                B_r = np.round(B * 2) / 2
                C_r = np.round(C * 2) / 2
                T_recon = np.einsum('ir,jr,kr->ijk', A_r, B_r, C_r)
                if np.allclose(T_recon, T, atol=1e-9):
                    best_U, best_V, best_W = A_r, B_r, C_r
                    best_rank = rank
                    break
                # Try rounding to nearest integer
                A_r = np.round(A)
                B_r = np.round(B)
                C_r = np.round(C)
                T_recon = np.einsum('ir,jr,kr->ijk', A_r, B_r, C_r)
                if np.allclose(T_recon, T, atol=1e-9):
                    best_U, best_V, best_W = A_r, B_r, C_r
                    best_rank = rank
                    break
        except Exception:
            continue
    
    # Return schoolbook if no better found
    if best_U is None:
        return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
    else:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
