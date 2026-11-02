import numpy as np
import time
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if it's the identity? 
    # No, matrix multiplication tensor T is defined such that 
    # C = A * B => C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor T is a 3-tensor where T[i, j, k] is the coefficient of A[i, j] B[j, k] in C[i, k].
    # So T[i, j, k] = 1 if the indices match the multiplication rule, 0 otherwise.
    # Specifically, T is a tensor of shape (n, m, p) such that 
    # T[i, j, k] = 1 if we are multiplying A[i,j] and B[j,k] to contribute to C[i,k].
    # Wait, the standard tensor for matrix multiplication $M_{n,m,p}$ has entries $T_{i,j,k}$.
    # The multiplication is $\langle A, B \rangle = C$.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # This can be written as $C = \text{mult}(A, B)$.
    # The tensor $T$ is such that $\text{vec}(C) = T \cdot (\text{vec}(A) \otimes \text{vec}(B))$? No.
    # The rank of the tensor is the minimal number of terms in the decomposition.
    # The entries of the tensor $T$ for $M_{n,m,p}$ are $T[i, j, k] = 1$ for all $i, j, k$?
    # No.
    # Let's check the seed code.
    # Seed code:
    # U[i*m+j, c] = 1
    # V[j*p+k, c] = 1
    # W[k*n+i, c] = 1
    # This corresponds to the term $A[i, j] B[j, k] W[k, i]$?
    # The output is $W$. The index is $k*n+i$. This is $C[k, i]$?
    # The task says: "output factor follows the official trace convention W[k*n+i,r]".
    # Usually $C[i, k]$ is row i, col k.
    # If $W$ index is $k*n+i$, then $W$ is flattened as $C^T$ or similar.
    # Let's trace the seed:
    # For a fixed (i, j, k):
    # U has 1 at row $i*m+j$ (which is $A[i, j]$).
    # V has 1 at row $j*p+k$ (which is $B[j, k]$).
    # W has 1 at row $k*n+i$.
    # The product $U \cdot V \cdot W$? No, the decomposition is $T = \sum U_c \otimes V_c \otimes W_c$.
    # The tensor entry $T[a, b, c]$ is sum over $r$ of $U[a, r] V[b, r] W[c, r]$.
    # Here $a$ is index of A, $b$ is index of B, $c$ is index of C.
    # $a = i*m+j$.
    # $b = j*p+k$.
    # $c = k*n+i$.
    # The seed sets $T[a, b, c] = 1$ for these specific tuples.
    # Are there any other non-zero entries?
    # For a different tuple $(i', j', k')$, the indices are different.
    # So the tensor $T$ has 1s at positions $(i*m+j, j*p+k, k*n+i)$ for all $i \in [0, n), j \in [0, m), k \in [0, p)$.
    # And 0 elsewhere.
    # This is the standard matrix multiplication tensor.
    
    # We need to find U, V, W of rank R such that 
    # T = sum_r U[:, r] * V[:, r] * W[:, r] (outer products)
    
    # Let's define the target tensor T_target.
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i*m+j, j*p+k, k*n+i] = 1.0

    # We want to minimize rank.
    # Start with R = 20.
    
    def objective(params, R):
        # params is a flat array of size R*(n*m + m*p + p*n)
        size_u = n*m * R
        size_v = m*p * R
        size_w = p*n * R
        
        U = params[:size_u].reshape(n*m, R)
        V = params[size_u:size_u+size_v].reshape(m*p, R)
        W = params[size_u+size_v:].reshape(p*n, R)
        
        # Reconstruct T
        # T_rec[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
        # This is efficient to compute as a loop over r or using einsum
        T_rec = np.einsum('ar,br,cr->abc', U, V, W)
        
        # Error
        err = np.sum((T_rec - T_target)**2)
        return err

    def check_solution(U, V, W):
        # Snap to half integers
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        
        # Check if snapped is close to original (optional, but good for verification)
        if not np.allclose(U, U_s, atol=1e-6):
            pass # It's okay if it's not close, we just need the snapped version to work exactly
        
        # Reconstruct with snapped values
        T_rec = np.einsum('ar,br,cr->abc', U_s, V_s, W_s)
        
        # Check exactness
        # The entries of T_target are 0 or 1.
        # The entries of T_rec should be 0 or 1.
        # Due to floating point, we check if they are close to integers.
        if np.allclose(T_rec, np.round(T_rec), atol=1e-9):
            T_rec_int = np.round(T_rec)
            if np.array_equal(T_rec_int, T_target):
                return U_s, V_s, W_s
        return None

    # Try ranks from 20 to 24
    for R in range(20, 25):
        # Multiple restarts
        for restart in range(10):
            # Initialize
            # Random initialization scaled appropriately
            # The entries of T are 1. The sum of products should be 1.
            # If R=20, average magnitude of U, V, W entries should be such that product sum is ~1.
            # If U, V, W are O(1), product is O(1), sum of 20 is O(20). Too big.
            # We need entries to be smaller.
            # Or, we can just let the optimizer figure it out.
            # Let's initialize with small random values.
            np.random.seed(42 + restart)
            params = np.random.randn(R * (n*m + m*p + p*n)) * 0.1
            
            res = minimize(objective, params, args=(R,), method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success:
                U = res.x[:n*m*R].reshape(n*m, R)
                V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
                W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
                
                sol = check_solution(U, V, W)
                if sol is not None:
                    U_s, V_s, W_s = sol
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
            
            # Also check if the raw solution is already close to half-integers
            # (Sometimes L-BFGS-B converges to the exact integer/half-integer point)
            U = res.x[:n*m*R].reshape(n*m, R)
            V = res.x[n*m*R:n*m*R + m*p*R].reshape(m*p, R)
            W = res.x[n*m*R + m*p*R:].reshape(p*n, R)
            sol = check_solution(U, V, W)
            if sol is not None:
                U_s, V_s, W_s = sol
                return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
                
    # If no solution found, return schoolbook
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
