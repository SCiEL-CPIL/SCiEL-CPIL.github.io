"""Matrix multiplication 2x2x5 rank-minimal tensor decomposition."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 5]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 20


def _build_schoolbook():
    """Return (U, V, W) as numpy arrays for the schoolbook rank-20 decomposition."""
    rank = RANK_SCHOOLBOOK
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _residual_fast(U, V, W):
    """Faster residual computation using einsum."""
    T = np.ones((n, m, p))
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    result = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    return np.max(np.abs(T - result))


def _fourier_basis(r, seed=0):
    """Build a structured U matrix using Fourier modes of Z_r."""
    rng = np.random.RandomState(seed)
    U = np.zeros((n * m, r))
    # Use discrete Fourier transform matrix as basis
    F = np.fft.fft(np.eye(r))
    F = F / np.sqrt(r)  # normalize
    # Take first n*m rows of F^H to form U
    # Actually, we need U to be (n*m, r). Let's use a different approach:
    # Parameterize U using Fourier coefficients
    for row in range(n * m):
        for c in range(r):
            # Mix of Fourier modes
            U[row, c] = np.exp(2j * np.pi * row * c / r).real
    # Add small random perturbation to break symmetry
    U += 0.01 * rng.randn(*U.shape)
    return U


def _solve_for_V_W(U, r):
    """Given fixed U, solve linear systems for V and W.
    
    For the tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    We can fix U and solve for V and W alternately.
    """
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    
    # Initialize V and W randomly
    rng = np.random.RandomState(42)
    V = rng.randn(m * p, r) * 0.1
    W = rng.randn(p * n, r) * 0.1
    
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    # ALS iterations to converge V and W given fixed U
    for it in range(200):
        # Update V: for each (j,k), solve for V[j*p+k, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i
        for j in range(m):
            for k in range(p):
                # A is (n, r): A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = x
                    V3[j, k, :] = x
                except:
                    pass
        
        # Update W: for each (k,i), solve for W[k*n+i, :]
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
        for k in range(p):
            for i in range(n):
                # A is (m, r): A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = x
                    W3[k, i, :] = x
                except:
                    pass
    
    return V, W


def _try_rank_structured(r, deadline, max_restarts=50):
    """Try to find a rank-r decomposition using structured Fourier basis for U."""
    if r >= RANK_SCHOOLBOOK:
        U, V, W = _build_schoolbook()
        return U, V, W, 0.0
    
    best_U, best_V, best_W = None, None, None
    best_res = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        # Build structured U with different seeds/perturbations
        seed = restart
        U = _fourier_basis(r, seed=seed)
        
        # Solve for V and W
        V, W = _solve_for_V_W(U, r)
        
        res = _residual_fast(U, V, W)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_res < 1e-10:
            break
    
    if best_res < 1e-8:
        return best_U, best_V, best_W, best_res
    return None


def _als_step(U, V, W, lr=0.1, max_iter=5000, tol=1e-10, deadline=None):
    """ALS optimization to minimize residual. Returns (U, V, W, residual)."""
    T = np.ones((n, m, p))
    U3 = U.reshape(n, m, -1)
    V3 = V.reshape(m, p, -1)
    W3 = W.reshape(p, n, -1)
    
    best_res = _residual_fast(U, V, W)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U
        for i in range(n):
            for j in range(m):
                A = (V3[j, :, :] * W3[:, i, :]).T
                b = np.ones(p)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U[i * m + j, :] = x
                    U3[i, j, :] = x
                except:
                    continue
        
        # Update V
        for j in range(m):
            for k in range(p):
                A = (U3[:, j, :] * W3[k, :, :]).T
                b = np.ones(n)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V[j * p + k, :] = x
                    V3[j, k, :] = x
                except:
                    continue
        
        # Update W
        for k in range(p):
            for i in range(n):
                A = (U3[i, :, :] * V3[:, k, :]).T
                b = np.ones(m)
                try:
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W[k * n + i, :] = x
                    W3[k, i, :] = x
                except:
                    continue
        
        res = _residual_fast(U, V, W)
        if res < best_res:
            best_res = res
        if res < tol:
            break
    
    return U, V, W, best_res


def _try_rank_als(r, deadline, max_restarts=30):
    """Try to find a rank-r decomposition using ALS from random initializations."""
    if r >= RANK_SCHOOLBOOK:
        U, V, W = _build_schoolbook()
        return U, V, W, 0.0
    
    best_U, best_V, best_W = None, None, None
    best_res = float('inf')
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        if restart == 0:
            U0, V0, W0 = _build_schoolbook()
            U = U0[:, :r].copy()
            V = V0[:, :r].copy()
            W = W0[:, :r].copy()
        else:
            U0, V0, W0 = _build_schoolbook()
            cols = random.sample(range(RANK_SCHOOLBOOK), r)
            U = U0[:, cols].copy()
            V = V0[:, cols].copy()
            W = W0[:, cols].copy()
            U += 0.01 * np.random.randn(*U.shape)
            V += 0.01 * np.random.randn(*V.shape)
            W += 0.01 * np.random.randn(*W.shape)
        
        U, V, W, res = _als_step(U, V, W, deadline=deadline)
        
        if res < best_res:
            best_res = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if best_res < 1e-10:
            break
    
    if best_res < 1e-8:
        return best_U, best_V, best_W, best_res
    return None


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    
    # Try structured Fourier basis approach for rank 14
    result = _try_rank_structured(14, deadline, max_restarts=30)
    if result is not None:
        U, V, W, res = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 15 with structured approach
    result = _try_rank_structured(15, deadline, max_restarts=30)
    if result is not None:
        U, V, W, res = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 16 with structured approach
    result = _try_rank_structured(16, deadline, max_restarts=20)
    if result is not None:
        U, V, W, res = result
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try ALS-based search for lower ranks
    for r in [14, 15, 16, 17]:
        if time.time() > deadline:
            break
        result = _try_rank_als(r, deadline, max_restarts=20)
        if result is not None:
            U, V, W, res = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook rank 20
    U, V, W = _build_schoolbook()
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
