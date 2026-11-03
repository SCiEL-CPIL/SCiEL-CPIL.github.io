"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    
    # Build the target tensor T of shape (dim_W, dim_U, dim_V)
    T = np.zeros((dim_W, dim_U, dim_V))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = k * n + i
                b = i * m + j
                c = j * p + k
                T[a, b, c] = 1.0
    
    T_flat = T.flatten()

    def tensor_to_matrix(U, V, W):
        """Compute the reconstructed tensor from factor matrices."""
        T_rec = np.zeros((dim_W, dim_U, dim_V))
        for rr in range(U.shape[1]):
            T_rec += np.einsum('a,b,c->abc', W[:, rr], U[:, rr], V[:, rr])
        return T_rec

    def check_exact(U, V, W, tol=1e-10):
        """Check if U, V, W exactly decompose T."""
        T_rec = tensor_to_matrix(U, V, W)
        return np.allclose(T_rec, T, atol=tol)

    def recover_half_integer(U, V, W, max_denom=8):
        """
        Try to recover exact half-integer (or rational with small denom) solution.
        For each entry, find the closest fraction with denominator <= max_denom
        that, when used in the decomposition, gives exact T.
        """
        # First try simple rounding to 1/2
        for denom in [1, 2, 4]:
            scale = denom
            U_r = np.round(U * scale) / scale
            V_r = np.round(V * scale) / scale
            W_r = np.round(W * scale) / scale
            
            T_rec = tensor_to_matrix(U_r, V_r, W_r)
            if np.allclose(T_rec, T, atol=1e-10):
                return U_r, V_r, W_r
        
        # Try to find a global scaling that makes entries half-integers
        # The decomposition is homogeneous: (cU, V, W/c) also works
        # Try different scalings
        for c in [0.5, 1.0, 2.0, 4.0]:
            U_s = U * c
            W_s = W / c
            V_s = V
            
            for scale in [1, 2, 4, 8]:
                U_r = np.round(U_s * scale) / scale
                V_r = np.round(V_s * scale) / scale
                W_r = np.round(W_s * scale) / scale
                
                T_rec = tensor_to_matrix(U_r, V_r, W_r)
                if np.allclose(T_rec, T, atol=1e-10):
                    return U_r, V_r, W_r
        
        return None

    def als_step(U, V, W, r):
        """One ALS iteration: update U, V, W in place."""
        # Precompute kron products for efficiency
        # Update U: for each row b of U
        for b in range(dim_U):
            X = np.zeros((dim_W * dim_V, r))
            for rr in range(r):
                X[:, rr] = np.kron(W[:, rr], V[:, rr])
            y = T[:, b, :].flatten()
            try:
                U[b, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                pass
        
        # Update V: for each row c of V
        for c in range(dim_V):
            X = np.zeros((dim_W * dim_U, r))
            for rr in range(r):
                X[:, rr] = np.kron(W[:, rr], U[:, rr])
            y = T[:, :, c].flatten()
            try:
                V[c, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                pass
        
        # Update W: for each row a of W
        for a in range(dim_W):
            X = np.zeros((dim_U * dim_V, r))
            for rr in range(r):
                X[:, rr] = np.kron(U[:, rr], V[:, rr])
            y = T[a, :, :].flatten()
            try:
                W[a, :] = np.linalg.lstsq(X, y, rcond=None)[0]
            except:
                pass

    def polish(U, V, W, max_iters=100, tol=1e-12):
        """High-precision polish using more iterations."""
        r = U.shape[1]
        for _ in range(max_iters):
            if time.time() > deadline - 5:
                break
            als_step(U, V, W, r)
            T_rec = tensor_to_matrix(U, V, W)
            error = np.linalg.norm(T - T_rec)
            if error < tol:
                break
        return U, V, W

    def init_structured(r, restart):
        """Create diverse initializations."""
        if restart == 0:
            # Schoolbook-like partial initialization
            U = np.zeros((dim_U, r))
            V = np.zeros((dim_V, r))
            W = np.zeros((dim_W, r))
            for rr in range(r):
                i = rr % n
                j = (rr // n) % m
                k = (rr // (n * m)) % p
                U[i * m + j, rr] = 1.0
                V[j * p + k, rr] = 1.0
                W[k * n + i, rr] = 1.0
            # Add small random perturbation
            U += np.random.randn(dim_U, r) * 0.01
            V += np.random.randn(dim_V, r) * 0.01
            W += np.random.randn(dim_W, r) * 0.01
            return U, V, W
        elif restart == 1:
            # Random with normalization
            U = np.random.randn(dim_U, r)
            V = np.random.randn(dim_V, r)
            W = np.random.randn(dim_W, r)
            U /= np.linalg.norm(U)
            V /= np.linalg.norm(V)
            W /= np.linalg.norm(W)
            return U, V, W
        elif restart == 2:
            # Block diagonal structure
            U = np.zeros((dim_U, r))
            V = np.zeros((dim_V, r))
            W = np.zeros((dim_W, r))
            for rr in range(r):
                idx_u = rr % dim_U
                idx_v = rr % dim_V
                idx_w = rr % dim_W
                U[idx_u, rr] = 1.0
                V[idx_v, rr] = 1.0
                W[idx_w, rr] = 1.0
            U += np.random.randn(dim_U, r) * 0.05
            V += np.random.randn(dim_V, r) * 0.05
            W += np.random.randn(dim_W, r) * 0.05
            return U, V, W
        else:
            # Pure random
            U = np.random.randn(dim_U, r) * 0.5
            V = np.random.randn(dim_V, r) * 0.5
            W = np.random.randn(dim_W, r) * 0.5
            return U, V, W

    best_result = None
    
    # Try ranks from 18 to 23 (theoretical min is 18 for 2x3*3x4)
    for r in range(18, 24):
        if time.time() > deadline - 20:
            break
        
        found = False
        max_restarts = 12 if r >= 18 else 8
        max_iters = 50
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            U, V, W = init_structured(r, restart)
            
            # Run ALS with early stopping and polishing
            prev_error = float('inf')
            for it in range(max_iters):
                if time.time() > deadline - 20:
                    break
                
                als_step(U, V, W, r)
                T_rec = tensor_to_matrix(U, V, W)
                error = np.linalg.norm(T - T_rec)
                
                # Check for convergence or stagnation
                if error < 1e-8:
                    # Polish with higher precision
                    U, V, W = polish(U, V, W, max_iters=80, tol=1e-13)
                    T_rec = tensor_to_matrix(U, V, W)
                    error = np.linalg.norm(T - T_rec)
                    if error < 1e-10:
                        # Try to recover exact half-integer solution
                        recovered = recover_half_integer(U, V, W)
                        if recovered is not None:
                            U_r, V_r, W_r = recovered
                            best_result = (U_r, V_r, W_r, r)
                            found = True
                            break
                        # Also try direct rounding as fallback
                        for scale in [1, 2, 4]:
                            U_s = np.round(U * scale) / scale
                            V_s = np.round(V * scale) / scale
                            W_s = np.round(W * scale) / scale
                            T_rec_s = tensor_to_matrix(U_s, V_s, W_s)
                            if np.allclose(T_rec_s, T, atol=1e-10):
                                best_result = (U_s, V_s, W_s, r)
                                found = True
                                break
                        if found:
                            break
                    break
                elif it > 5 and error > prev_error * 1.01:
                    # Diverging or stuck, break early
                    break
                prev_error = error
            
            if found:
                break

        if found:
            break

    if best_result is not None:
        U_r, V_r, W_r, r = best_result
        return {"U": U_r.tolist(), "V": V_r.tolist(), "W": W_r.tolist()}

    # Fallback: Schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
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
