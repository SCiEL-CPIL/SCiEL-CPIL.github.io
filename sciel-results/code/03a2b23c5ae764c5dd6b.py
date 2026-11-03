import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m      # 10
    V_dim = m * p      # 25
    W_dim = p * n      # 10
    
    start_time = time.time()
    deadline = start_time + 115  # 115s limit to be safe within 120s

    # Build target tensor
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                alpha = i * m + j
                beta = j * p + k
                gamma = k * n + i
                T[alpha, beta, gamma] = 1.0

    # Precompute index lists for fast access
    indices_alpha = np.arange(U_dim)
    indices_beta = np.arange(V_dim)
    indices_gamma = np.arange(W_dim)

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(U_dim)]
        V = [[0] * rank for _ in range(V_dim)]
        W = [[0] * rank for _ in range(W_dim)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def compute_residual_fast(U, V, W):
        """Compute Frobenius residual using einsum for speed."""
        # prod[alpha, beta, gamma] = sum_r U[alpha,r]*V[beta,r]*W[gamma,r]
        prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
        diff = prod - T
        return np.sum(diff * diff)

    def exact_verify(U, V, W):
        """Verify exact decomposition with rational arithmetic."""
        R = U.shape[1] if hasattr(U, 'shape') else len(U[0])
        # Convert to fractions
        U_frac = np.zeros((U_dim, R), dtype=object)
        V_frac = np.zeros((V_dim, R), dtype=object)
        W_frac = np.zeros((W_dim, R), dtype=object)
        
        for i in range(U_dim):
            for r in range(R):
                U_frac[i, r] = Fraction(U[i, r]).limit_denominator(1000)
        for i in range(V_dim):
            for r in range(R):
                V_frac[i, r] = Fraction(V[i, r]).limit_denominator(1000)
        for i in range(W_dim):
            for r in range(R):
                W_frac[i, r] = Fraction(W[i, r]).limit_denominator(1000)
        
        # Only check non-zero entries of T for speed
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    alpha = i * m + j
                    beta = j * p + k
                    gamma = k * n + i
                    val = sum(U_frac[alpha, r] * V_frac[beta, r] * W_frac[gamma, r] for r in range(R))
                    if val != 1:
                        return False
        return True

    def snap_to_halfint(U, V, W):
        """Snap values to nearest half-integer and verify with float."""
        U_p = np.round(U * 2) / 2.0
        V_p = np.round(V * 2) / 2.0
        W_p = np.round(W * 2) / 2.0
        
        # Quick float check
        prod = np.einsum('ar,br,cr->abc', U_p, V_p, W_p, optimize=True)
        if np.max(np.abs(prod - T)) > 1e-9:
            return None
        return U_p, V_p, W_p

    def als_solve(R, max_iters=40, tol=1e-10, seed=None):
        """ALS with structured initialization."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(U_dim, R) * 0.1
        V = rng.randn(V_dim, R) * 0.1
        W = rng.randn(W_dim, R) * 0.1
        
        # Add schoolbook structure
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = (i * m + j) * p + k
                    if r < R:
                        U[i * m + j, r] += 1.0
                        V[j * p + k, r] += 1.0
                        W[k * n + i, r] += 1.0
        
        prev_res = float('inf')
        for it in range(max_iters):
            if time.time() > deadline:
                break
                
            # Update U
            for alpha in range(U_dim):
                A = np.empty((V_dim * W_dim, R))
                b = np.empty(V_dim * W_dim)
                idx = 0
                for beta in range(V_dim):
                    for gamma in range(W_dim):
                        A[idx, :] = V[beta] * W[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                U[alpha], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update V
            for beta in range(V_dim):
                A = np.empty((U_dim * W_dim, R))
                b = np.empty(U_dim * W_dim)
                idx = 0
                for alpha in range(U_dim):
                    for gamma in range(W_dim):
                        A[idx, :] = U[alpha] * W[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                V[beta], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Update W
            for gamma in range(W_dim):
                A = np.empty((U_dim * V_dim, R))
                b = np.empty(U_dim * V_dim)
                idx = 0
                for alpha in range(U_dim):
                    for beta in range(V_dim):
                        A[idx, :] = U[alpha] * V[beta]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                W[gamma], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            res = compute_residual_fast(U, V, W)
            if abs(prev_res - res) < tol:
                break
            prev_res = res
            
        return U, V, W, prev_res

    def gd_solve(R, max_iters=300, lr=0.01, momentum=0.9, seed=None):
        """Gradient descent with momentum, structured init."""
        rng = np.random.RandomState(seed)
        
        U = rng.randn(U_dim, R) * 0.1
        V = rng.randn(V_dim, R) * 0.1
        W = rng.randn(W_dim, R) * 0.1
        
        # Schoolbook init
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = (i * m + j) * p + k
                    if r < R:
                        U[i * m + j, r] += 1.0
                        V[j * p + k, r] += 1.0
                        W[k * n + i, r] += 1.0
        
        vU = np.zeros_like(U)
        vV = np.zeros_like(V)
        vW = np.zeros_like(W)
        prev_loss = float('inf')
        no_improve = 0
        patience = 50
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            prod = np.einsum('ar,br,cr->abc', U, V, W, optimize=True)
            diff = prod - T
            
            gradU = np.einsum('abc,br,cr->ar', diff, V, W, optimize=True)
            gradV = np.einsum('abc,ar,cr->br', diff, U, W, optimize=True)
            gradW = np.einsum('abc,ar,br->cr', diff, U, V, optimize=True)
            
            vU = momentum * vU + lr * gradU
            vV = momentum * vV + lr * gradV
            vW = momentum * vW + lr * gradW
            
            U -= vU
            V -= vV
            W -= vW
            
            loss = compute_residual_fast(U, V, W)
            
            if loss < prev_loss - 1e-10:
                prev_loss = loss
                no_improve = 0
            else:
                no_improve += 1
            
            if no_improve > patience:
                break
            if loss < 1e-10:
                break
        
        return U, V, W, loss

    def polish(U, V, W, max_iters=15):
        """High-precision polish: snap + ALS."""
        U_p, V_p, W_p = U.copy(), V.copy(), W.copy()
        R = U_p.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            
            U_s = np.round(U_p * 2) / 2.0
            V_s = np.round(V_p * 2) / 2.0
            W_s = np.round(W_p * 2) / 2.0
            
            if exact_verify(U_s, V_s, W_s):
                return U_s, V_s, W_s
            
            # One ALS round from snapped
            for alpha in range(U_dim):
                A = np.empty((V_dim * W_dim, R))
                b = np.empty(V_dim * W_dim)
                idx = 0
                for beta in range(V_dim):
                    for gamma in range(W_dim):
                        A[idx, :] = V_s[beta] * W_s[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                U_p[alpha], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            for beta in range(V_dim):
                A = np.empty((U_dim * W_dim, R))
                b = np.empty(U_dim * W_dim)
                idx = 0
                for alpha in range(U_dim):
                    for gamma in range(W_dim):
                        A[idx, :] = U_p[alpha] * W_s[gamma]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                V_p[beta], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            for gamma in range(W_dim):
                A = np.empty((U_dim * V_dim, R))
                b = np.empty(U_dim * V_dim)
                idx = 0
                for alpha in range(U_dim):
                    for beta in range(V_dim):
                        A[idx, :] = U_p[alpha] * V_p[beta]
                        b[idx] = T[alpha, beta, gamma]
                        idx += 1
                W_p[gamma], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        U_f = np.round(U_p * 2) / 2.0
        V_f = np.round(V_p * 2) / 2.0
        W_f = np.round(W_p * 2) / 2.0
        if exact_verify(U_f, V_f, W_f):
            return U_f, V_f, W_f
        return None

    # Fallback
    best_rank = n * m * p
    best_U, best_V, best_W = schoolbook()["U"], schoolbook()["V"], schoolbook()["W"]

    # Try ranks from lower bound
    lower_bound = max(U_dim, V_dim, W_dim)  # 25
    for R in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        
        found = False
        attempts = 0
        max_attempts = 12
        
        while not found and attempts < max_attempts and time.time() < deadline - 10:
            attempts += 1
            seed = attempts * 7919 + R * 31
            
            # Alternate between ALS and GD
            if attempts % 2 == 0:
                U, V, W, res = als_solve(R, max_iters=35, seed=seed)
            else:
                U, V, W, res = gd_solve(R, max_iters=250, seed=seed)
            
            if res < 1e-3:
                polished = polish(U, V, W, max_iters=12)
                if polished is not None:
                    U_p, V_p, W_p = polished
                    if exact_verify(U_p, V_p, W_p):
                        found = True
                        best_rank = R
                        best_U = U_p.tolist()
                        best_V = V_p.tolist()
                        best_W = W_p.tolist()
                        break
            elif res < 1e-2:
                snapped = snap_to_halfint(U, V, W)
                if snapped is not None:
                    U_p, V_p, W_p = snapped
                    if exact_verify(U_p, V_p, W_p):
                        found = True
                        best_rank = R
                        best_U = U_p.tolist()
                        best_V = V_p.tolist()
                        best_W = W_p.tolist()
                        break
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
