"""Numerical CP decomposition approach for 2x2 * 2x11 matrix multiplication tensor."""
import numpy as np
import time
from scipy.linalg import solve

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = 2, 2, 11
    
    # Build the matrix multiplication tensor T of shape (n*m, m*p, p*n) = (4, 22, 22)
    # T[i*m + j, j*p + k, k*n + i] = 1 for all valid i,j,k
    # Otherwise 0
    
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0
    
    # Try to find a CP decomposition of rank r
    # We'll try increasing ranks starting from a lower bound
    # Theoretical lower bound: For 2x2*2xp, the rank is at least 4*p - something
    # Actually, for 2x2*2x2, rank is 7 (Strassen)
    # For 2x2*2x11, a simple block decomposition gives 5*7 + 4 = 39
    # But maybe we can do better by sharing structure across columns?
    
    # Let's try ranks from 30 up to 44 (schoolbook is 44)
    # We'll use a simple ALS (Alternating Least Squares) approach for CP decomposition
    
    def cp_decompose(T, rank, max_iter=500, tol=1e-6):
        """Simple CP decomposition using ALS."""
        shape = T.shape
        # Initialize with random values
        np.random.seed(42)
        U = np.random.randn(shape[0], rank)
        V = np.random.randn(shape[1], rank)
        W = np.random.randn(shape[2], rank)
        
        prev_err = np.inf
        for it in range(max_iter):
            # Update U
            # T_u1 is T reshaped to (shape[0], shape[1]*shape[2])
            T_u1 = T.reshape(shape[0], -1)
            # Solve: U = T_u1 @ (V ⊙ W)^+ where ⊙ is Khatri-Rao product
            VW = np.kron(V, W)  # This is not quite right for multi-mode
            # Actually, for CP decomposition:
            # U = T_u1 @ (V ⊙ W)^+
            # where V ⊙ W is the Khatri-Rao product (column-wise Kronecker)
            # For matrices, Khatri-Rao product of V (n2 x r) and W (n3 x r) is (n2*n3 x r)
            # where column k is V[:,k] ⊗ W[:,k]
            
            # Build Khatri-Rao product
            def khatri_rao(A, B):
                """Khatri-Rao product of two matrices (column-wise Kronecker)."""
                r = A.shape[1]
                result = np.zeros((A.shape[0] * B.shape[0], r))
                for k in range(r):
                    result[:, k] = np.kron(A[:, k], B[:, k])
                return result
            
            VW = khatri_rao(V, W)
            # U = T_u1 @ pinv(VW)
            U_new = T_u1 @ np.linalg.pinv(VW)
            
            # Update V
            T_v1 = T.transpose(1, 0, 2).reshape(shape[1], -1)
            UW = khatri_rao(U_new, W)
            V_new = T_v1 @ np.linalg.pinv(UW)
            
            # Update W
            T_w1 = T.transpose(2, 0, 1).reshape(shape[2], -1)
            UV = khatri_rao(U_new, V_new)
            W_new = T_w1 @ np.linalg.pinv(UV)
            
            # Compute error
            # Reconstruct T_approx
            T_approx = np.zeros(shape)
            for k in range(rank):
                T_approx += np.einsum('i,j,l->ijl', U_new[:, k], V_new[:, k], W_new[:, k])
            err = np.linalg.norm(T - T_approx) / np.linalg.norm(T)
            
            if abs(prev_err - err) < tol:
                break
            prev_err = err
            
            U, V, W = U_new, V_new, W_new
        
        return U, V, W, err
    
    best_rank = None
    best_decomp = None
    
    # Try ranks from 35 to 44
    for rank in range(35, 45):
        if time.time() > deadline:
            break
        
        U, V, W, err = cp_decompose(T, rank, max_iter=200)
        
        # Check if the decomposition is close to exact
        if err < 1e-4:
            # Try to round to rational numbers
            # Scale to make entries close to integers or half-integers
            # Find a common scale factor
            all_entries = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            nonzero = all_entries[np.abs(all_entries) > 1e-6]
            
            # Try to find a scale that makes entries close to multiples of 0.5
            # First, try to normalize by the largest entry
            max_val = np.max(np.abs(all_entries))
            if max_val > 1e-6:
                scale = 1.0 / max_val
                U_scaled = U * scale
                V_scaled = V * scale
                W_scaled = W * scale
                
                # Try rounding to nearest 0.5
                U_rounded = np.round(U_scaled * 2) / 2
                V_rounded = np.round(V_scaled * 2) / 2
                W_rounded = np.round(W_scaled * 2) / 2
                
                # Verify the rounded decomposition
                T_rounded = np.zeros((n*m, m*p, p*n))
                for k in range(rank):
                    T_rounded += np.einsum('i,j,l->ijl', U_rounded[:, k], V_rounded[:, k], W_rounded[:, k])
                
                round_err = np.linalg.norm(T - T_rounded) / np.linalg.norm(T)
                if round_err < 1e-6:
                    best_rank = rank
                    best_decomp = (U_rounded, V_rounded, W_rounded)
                    break
            
            # If rounding failed, try with different scaling
            # Try to find a scale that makes entries close to integers
            for scale_mult in [1, 2, 4, 8, 16, 32]:
                U_scaled = U * scale_mult
                V_scaled = V * scale_mult
                W_scaled = W * scale_mult
                
                U_rounded = np.round(U_scaled)
                V_rounded = np.round(V_scaled)
                W_rounded = np.round(W_scaled)
                
                T_rounded = np.zeros((n*m, m*p, p*n))
                for k in range(rank):
                    T_rounded += np.einsum('i,j,l->ijl', U_rounded[:, k], V_rounded[:, k], W_rounded[:, k])
                
                round_err = np.linalg.norm(T - T_rounded) / np.linalg.norm(T)
                if round_err < 1e-6:
                    best_rank = rank
                    best_decomp = (U_rounded, V_rounded, W_rounded)
                    break
            
            if best_rank is not None:
                break
    
    # If no exact decomposition found, fall back to schoolbook
    if best_rank is None or best_decomp is None:
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        
        return {"U": U, "V": V, "W": W}
    
    # Convert numpy arrays to lists
    U_list = best_decomp[0].tolist()
    V_list = best_decomp[1].tolist()
    W_list = best_decomp[2].tolist()
    
    # Ensure entries are exactly 0, 0.5, 1, etc.
    def clean_matrix(mat):
        cleaned = []
        for row in mat:
            new_row = []
            for val in row:
                # Round to nearest 0.5
                rounded = round(val * 2) / 2
                # Clean up floating point artifacts
                if abs(rounded - round(rounded)) < 1e-9:
                    rounded = round(rounded)
                new_row.append(rounded)
            cleaned.append(new_row)
        return cleaned
    
    U_clean = clean_matrix(U_list)
    V_clean = clean_matrix(V_list)
    W_clean = clean_matrix(W_list)
    
    return {"U": U_clean, "V": V_clean, "W": W_clean}
