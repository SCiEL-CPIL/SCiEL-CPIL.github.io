import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if it's a multiplication term
    # T is (n*m) x (m*p) x (p*n) ? No.
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    # T[i*m+j, j*p+k, k*n+i] = 1 ?
    # The tensor is defined such that sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = T[i,j,k]
    # Where T[i,j,k] is 1 if the indices correspond to a valid term in C[i,k] = sum_j A[i,j] B[j,k].
    # Actually, the tensor is the identity map on the bilinear form.
    # T[i, j, k] is 1 if i, j, k are consistent?
    # No, the tensor M_{n,m,p} has entries M_{i, j, k} = 1 for all i, j, k?
    # No. The tensor is the structure constant tensor.
    # C_{i,k} = sum_j A_{i,j} B_{j,k}.
    # This is a bilinear map.
    # The tensor T is in (V*) x (W*) x Z.
    # T(e_i, e_j, e_k) = 1 if k corresponds to the output index?
    # Let's define the target matrix for each rank-1 term.
    # A rank-1 term is U_c (vec of A), V_c (vec of B), W_c (vec of C).
    # The product is sum_j A_j V_{j,c} * W_{k,c} ?
    # No.
    # The decomposition is:
    # C = sum_c (U_c^T A) * (V_c^T B) * W_c ?
    # No.
    # Standard tensor decomposition for matmul:
    # C_{i,k} = sum_c U_{i,j,c} V_{j,k,c} W_{k,i,c} ?
    # The indices in the prompt:
    # U: (n*m, rank). Index i*m+j.
    # V: (m*p, rank). Index j*p+k.
    # W: (p*n, rank). Index k*n+i.
    # The sum is over c.
    # Term: U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    # We want this sum to equal T[i,j,k].
    # T[i,j,k] is 1 if the term A[i,j] B[j,k] contributes to C[i,k].
    # Yes, T[i,j,k] = 1 for all valid i,j,k.
    # Wait, if T[i,j,k]=1 for all i,j,k, then the sum is 1.
    # But the sum is over c.
    # So for each (i,j,k), sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c] = 1.
    # This must hold for all i in 0..n-1, j in 0..m-1, k in 0..p-1.
    # Note that W index is k*n+i.
    # So for a fixed (i,k), the term W[k*n+i, c] is common to all j?
    # No, the sum is for a specific (i,j,k).
    # The equation is:
    # For all i, j, k:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1.
    
    # This is a system of equations.
    # Number of equations: n*m*p = 24.
    # Number of variables: rank * (n*m + m*p + p*n).
    
    # We can solve this.
    
    def solve_rank(rank):
        # Initialize
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        for _ in range(100):
            # Solve for U
            # For each (i,j,k): sum_c U[i*m+j, c] * (V[j*p+k, c] * W[k*n+i, c]) = 1
            # This is linear in U if V, W are fixed.
            # But U[i*m+j, c] appears in multiple equations (for different k).
            # Let A_c = V[j*p+k, c] * W[k*n+i, c].
            # sum_c U[i*m+j, c] * A_c = 1.
            
            # We can set up a linear system for all U variables.
            # Variables: U_flat (n*m * rank)
            # Equations: n*m*p
            # Matrix M: (n*m*p) x (n*m * rank)
            # M[row, col] = ...
            
            # This is large but n*m*p=24, n*m=6, rank~20.
            # 24 x 120. Easy.
            
            M_u = np.zeros((n*m*p, n*m * rank))
            rhs_u = np.ones(n*m*p)
            
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = idx
                        for c in range(rank):
                            # U[i*m+j, c] coefficient is V[j*p+k, c] * W[k*n+i, c]
                            M_u[row, (i*m+j)*rank + c] = V[j*p+k, c] * W[k*n+i, c]
                        idx += 1
            
            # Solve least squares
            U_flat, _, _, _ = np.linalg.lstsq(M_u, rhs_u, rcond=None)
            U = U_flat.reshape(n*m, rank)
            
            # Similarly for V
            M_v = np.zeros((n*m*p, m*p * rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = idx
                        for c in range(rank):
                            M_v[row, (j*p+k)*rank + c] = U[i*m+j, c] * W[k*n+i, c]
                        idx += 1
            V_flat, _, _, _ = np.linalg.lstsq(M_v, rhs_u, rcond=None)
            V = V_flat.reshape(m*p, rank)
            
            # Similarly for W
            M_w = np.zeros((n*m*p, p*n * rank))
            idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        row = idx
                        for c in range(rank):
                            M_w[row, (k*n+i)*rank + c] = U[i*m+j, c] * V[j*p+k, c]
                        idx += 1
            W_flat, _, _, _ = np.linalg.lstsq(M_w, rhs_u, rcond=None)
            W = W_flat.reshape(p*n, rank)
            
            # Snap to half-integers
            U = np.round(U * 2) / 2
            V = np.round(V * 2) / 2
            W = np.round(W * 2) / 2
            
            # Check error
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        error += (val - 1)**2
            if error < 1e-6:
                return U, V, W
                
        return None

    # Try ranks from 19 to 23
    for r in range(19, 24):
        start_time = time.time()
        while time.time() - start_time < 10:
            res = solve_rank(r)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    # Fallback to schoolbook
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
