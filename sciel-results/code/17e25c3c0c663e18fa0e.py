import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=5, p=5
    # U shape (n*m, R) = (20, R)
    # V shape (m*p, R) = (25, R)
    # W shape (p*n, R) = (20, R)
    
    # Target Rank: 4*m - 1 = 19
    target_rank = 4 * m - 1
    
    start_time = time.time()
    deadline = 110.0  # 10 second margin
    
    def build_tensor():
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # A index: i*m + j
                    # B index: j*p + k
                    # C index: k*n + i
                    T[i*m + j, j*p + k, k*n + i] = 1.0
        return T

    T = build_tensor()
    
    # Try to find decomposition of rank target_rank
    R = target_rank
    
    # Random initialization
    np.random.seed(42)
    U = np.random.randn(n*m, R)
    V = np.random.randn(m*p, R)
    W = np.random.randn(p*n, R)
    
    # Simple gradient descent or fixed point iteration
    # Since it's a bilinear problem, we can alternate minimization.
    # Fix U, V -> solve W. Fix U, W -> solve V. Fix V, W -> solve U.
    
    def solve_W(U, V, T):
        # T_{a,b,c} = sum_r U_{a,r} V_{b,r} W_{c,r}
        # For each c, T_{a,b,c} = sum_r (U_{a,r} V_{b,r}) W_{c,r}
        # Let M_{a,b} = T_{a,b,c}. Let X_{a,b,r} = U_{a,r} V_{b,r}.
        # This is a linear system for W_{c,r}.
        # But X depends on U, V.
        # We can vectorize.
        # T is (20, 25, 20).
        # For a fixed c, T[:,:,c] (20x25) = sum_r W[c,r] * (U[:,r] V[:,r].T)
        # This is a linear system for W[c,:].
        # Let A_c be a matrix of size (20*25) x R.
        # A_c[r, a*m+b] = U[a,r] * V[b,r]
        # b_vec = T[:,:,c].flatten()
        # Solve A_c W_c = b_vec
        W_sol = np.zeros((p*n, R))
        for c in range(p*n):
            A_mat = np.zeros((n*m * m*p, R))
            for r in range(R):
                # Outer product of U[:,r] and V[:,r]
                outer = np.outer(U[:, r], V[:, r])
                A_mat[:, r] = outer.flatten()
            b_vec = T[:, :, c].flatten()
            # Least squares
            W_c, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            W_sol[c, :] = W_c
        return W_sol

    def solve_U(V, W, T):
        # T_{a,b,c} = sum_r U_{a,r} (V_{b,r} W_{c,r})
        # For each a, T[a,:,:] (25x20) = sum_r U[a,r] * (V[:,r] W[:,r].T)
        U_sol = np.zeros((n*m, R))
        for a in range(n*m):
            A_mat = np.zeros((m*p * p*n, R))
            for r in range(R):
                outer = np.outer(V[:, r], W[:, r])
                A_mat[:, r] = outer.flatten()
            b_vec = T[a, :, :].flatten()
            U_a, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            U_sol[a, :] = U_a
        return U_sol

    def solve_V(U, W, T):
        # T_{a,b,c} = sum_r (U_{a,r} W_{c,r}) V_{b,r}
        # For each b, T[:,:,b] (20x20) = sum_r V[b,r] * (U[:,r] W[:,r].T)
        V_sol = np.zeros((m*p, R))
        for b in range(m*p):
            A_mat = np.zeros((n*m * p*n, R))
            for r in range(R):
                outer = np.outer(U[:, r], W[:, r])
                A_mat[:, r] = outer.flatten()
            b_vec = T[:, :, b].flatten()
            V_b, residuals, rank, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            V_sol[b, :] = V_b
        return V_sol

    # Iterate
    for iter in range(500):
        if time.time() > deadline:
            break
            
        W = solve_W(U, V, T)
        U = solve_U(V, W, T)
        V = solve_V(U, W, T)
        
        # Check convergence
        # Compute error
        # T_recon = sum_r U[:,r,None] * V[None,:,r] * W[None,None,r]?
        # Efficient check:
        # Error = || T - sum_r outer(U[:,r], outer(V[:,r], W[:,r])) ||
        # This is expensive.
        # Just check norm of U, V, W changes.
        if iter % 10 == 0:
             pass # print(iter)

    # Final check
    # Reconstruct T
    T_recon = np.zeros_like(T)
    for r in range(R):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    
    error = np.linalg.norm(T - T_recon)
    
    if error < 1e-4:
        # Try to round to integers/half-integers
        # Multiply by 2 to check for integers
        U_int = np.round(U * 2) / 2
        V_int = np.round(V * 2) / 2
        W_int = np.round(W * 2) / 2
        
        T_recon_int = np.zeros_like(T)
        for r in range(R):
            T_recon_int += np.einsum('a,b,c->abc', U_int[:, r], V_int[:, r], W_int[:, r])
        
        error_int = np.linalg.norm(T - T_recon_int)
        
        if error_int < 1e-9:
            return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
        
        # If integer rounding fails, maybe scale?
        # Or just return the float solution? 
        # Prompt says "integer/half-integer entries".
        # If it doesn't round, it's invalid.
        pass

    # Fallback to Schoolbook
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
# EVOLVE-BLOCK-END
