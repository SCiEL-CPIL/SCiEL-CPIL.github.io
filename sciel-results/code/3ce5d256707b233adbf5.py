"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0  # safety margin before 150s kill
    
    # Build schoolbook decomposition
    def build_schoolbook():
        terms = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
        rank = len(terms)
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for c, (i, j, k) in enumerate(terms):
            U[i * m + j, c] = 1.0
            V[j * p + k, c] = 1.0
            W[k * n + i, c] = 1.0
        return U, V, W

    # Verify decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    def verify(U, V, W, tol=1e-9):
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    if abs(total - 1.0) > tol:
                        return False
        return True

    # Compute tensor error (sum of squared errors)
    def tensor_sse(U, V, W):
        rank = U.shape[1]
        sse = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_val = U[i * m + j]
                    v_val = V[j * p + k]
                    w_val = W[k * n + i]
                    total = np.sum(u_val * v_val * w_val)
                    sse += (total - 1.0) ** 2
        return sse

    # Start with schoolbook
    U, V, W = build_schoolbook()
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if not verify(best_U, best_V, best_W):
        raise ValueError("Schoolbook decomposition failed verification")
    
    np.random.seed(42)
    
    # Vectorized Gradient Descent
    def vectorized_grad_descent(U_init, V_init, W_init, steps=100, lr=0.01):
        n, m, p = DIMENSIONS
        R = U_init.shape[1]
        
        U = U_init.copy()
        V = V_init.copy()
        W = W_init.copy()
        
        # Reshape for einsum
        # U: (n, m, R)
        # V: (m, p, R)
        # W: (p, n, R)
        U_res = U.reshape(n, m, R)
        V_res = V.reshape(m, p, R)
        W_res = W.reshape(p, n, R)
        
        best_sse = float('inf')
        
        for step in range(steps):
            if time.time() > deadline - 10:
                break
                
            # Compute P_ijk
            # P_ijk = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
            # Use einsum
            P = np.einsum('ijr,jkr,kr->ijk', U_res, V_res, W_res)
            
            # Compute D_ijk = P_ijk - 1
            D = P - 1.0
            
            # Compute SSE
            sse = np.sum(D**2)
            if sse < best_sse:
                best_sse = sse
                if sse < 1e-12:
                    break
            
            # Compute Gradients
            # G_U[i,j,r] = sum_k D[i,j,k] V[j,k,r] W[k,i,r]
            G_U = np.einsum('ijk,jkr,kr->ijr', D, V_res, W_res)
            # G_V[j,k,r] = sum_i D[i,j,k] U[i,j,r] W[k,i,r]
            G_V = np.einsum('ijk,ijr,ki->jkr', D, U_res, W_res)
            # G_W[k,i,r] = sum_j D[i,j,k] U[i,j,r] V[j,k,r]
            G_W = np.einsum('ijk,ijr,jkr->kir', D, U_res, V_res)
            
            # Update
            U_res -= lr * G_U
            V_res -= lr * G_V
            W_res -= lr * G_W
            
            # Reshape back to 2D for next iteration's reshape? 
            # Actually, we can keep them as 3D and just view them.
            # But we need to return 2D.
            U = U_res.reshape(n*m, R)
            V = V_res.reshape(m*p, R)
            W = W_res.reshape(p*n, R)
            
        return U, V, W, best_sse

    # Try to find lower rank
    for target_rank in [41, 40, 39]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(10):
            if time.time() > deadline - 20:
                break
            
            # Initialize
            # Start with random small values
            U_c = np.random.randn(n * m, target_rank) * 0.5
            V_c = np.random.randn(m * p, target_rank) * 0.5
            W_c = np.random.randn(p * n, target_rank) * 0.5
            
            U_res, V_res, W_res, sse = vectorized_grad_descent(U_c, V_c, W_c, steps=200, lr=0.05)
            
            if sse < 1e-10:
                # Try to snap to half-integers
                U_snap = np.round(U_res * 2.0) / 2.0
                V_snap = np.round(V_res * 2.0) / 2.0
                W_snap = np.round(W_res * 2.0) / 2.0
                
                if verify(U_snap, V_snap, W_snap):
                    best_rank = target_rank
                    best_U, best_V, best_W = U_snap.copy(), V_snap.copy(), W_snap.copy()
                    found = True
                    break
            
            # Also check continuous solution if very low error
            if sse < 1e-12:
                 # If continuous solution is perfect, we might be able to find a discrete one nearby
                 # But verify requires exact match.
                 pass

        if found:
            continue
        else:
            break
            
    # Final verification
    if not verify(best_U, best_V, best_W):
        # Fall back to schoolbook
        best_U, best_V, best_W = build_schoolbook()
        best_rank = n * m * p
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
