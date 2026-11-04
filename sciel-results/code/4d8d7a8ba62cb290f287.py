"""Matrix multiplication 3x4x6 - model-and-optimize approach."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def _trivial_decomposition(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
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

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check that sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    rank = U.shape[1]
    Ut = U.reshape(n, m, rank)
    Vt = V.reshape(m, p, rank)
    Wt = W.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    expected = np.ones((n, m, p))
    return np.max(np.abs(T - expected)) < tol

def _structured_init(n, m, p, r, seed=42):
    """
    Create structured initializations that respect bilinear structure.
    Uses block patterns inspired by Strassen-like decompositions.
    """
    rng = np.random.RandomState(seed)
    
    # Strategy 1: Kronecker-like block structure
    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))
    
    # Distribute the r factors across (i,j,k) indices with some overlap
    # For each factor c, assign it to cover a subset of (i,j,k) tuples
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if idx < r:
                    U0[i * m + j, idx] = 1.0
                    V0[j * p + k, idx] = 1.0
                    W0[k * n + i, idx] = 1.0
                    idx += 1
    # Add random perturbations to the remaining factors
    if r > n * m * p:
        for c in range(n * m * p, r):
            U0[:, c] = rng.randn(n * m) * 0.1
            V0[:, c] = rng.randn(m * p) * 0.1
            W0[:, c] = rng.randn(p * n) * 0.1
    elif r < n * m * p:
        # For rank reduction, we need to merge some terms
        # Add random components to existing factors
        for c in range(r):
            U0[:, c] += rng.randn(n * m) * 0.5
            V0[:, c] += rng.randn(m * p) * 0.5
            W0[:, c] += rng.randn(p * n) * 0.5
    
    return U0, V0, W0

def _random_init(n, m, p, r, seed=42):
    """Standard random initialization."""
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(r)
    U0 = rng.randn(n * m, r) * scale
    V0 = rng.randn(m * p, r) * scale
    W0 = rng.randn(p * n, r) * scale
    return U0, V0, W0

def _loss_and_grad_factory(n, m, p, r):
    """Create loss and gradient functions for optimization."""
    def loss_and_grad(x):
        U = x[:n*m*r].reshape(n, m, r)
        V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
        W = x[n*m*r + m*p*r :].reshape(p, n, r)
        
        T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        err = T - 1.0
        loss = np.sum(err ** 2)
        
        # Gradients
        dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
        dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
        dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2
        
        grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
        return loss, grad
    return loss_and_grad

def _polish(U, V, W, n, m, p, r, max_time=5.0):
    """High-precision polish pass with tighter tolerances."""
    start = time.time()
    x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    loss_fn = _loss_and_grad_factory(n, m, p, r)
    
    try:
        result = opt.minimize(loss_fn, x, method='L-BFGS-B', jac=True,
                              options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12,
                                       'maxfun': 5000})
        if time.time() - start > max_time:
            return None
        U_opt = result.x[:n*m*r].reshape(n*m, r)
        V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
        return U_opt, V_opt, W_opt
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
    n, m, p = DIMENSIONS
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = n * m * p  # 72
    
    def try_rank(r, max_time_per_attempt=8.0):
        """Try to find a rank-r decomposition using multiple strategies."""
        loss_fn = _loss_and_grad_factory(n, m, p, r)
        attempts = []
        
        # Strategy 1: Structured initialization
        U0, V0, W0 = _structured_init(n, m, p, r, seed=42)
        attempts.append((U0, V0, W0))
        
        # Strategy 2: Random initialization with different seeds
        for seed in [0, 1, 2, 3]:
            U0, V0, W0 = _random_init(n, m, p, r, seed=seed)
            attempts.append((U0, V0, W0))
        
        # Strategy 3: Perturbed structured initialization
        U0, V0, W0 = _structured_init(n, m, p, r, seed=99)
        U0 = U0 + np.random.randn(*U0.shape) * 0.3
        V0 = V0 + np.random.randn(*V0.shape) * 0.3
        W0 = W0 + np.random.randn(*W0.shape) * 0.3
        attempts.append((U0, V0, W0))
        
        best_result = None
        best_loss = np.inf
        
        for U_init, V_init, W_init in attempts:
            if time.time() > deadline - 5:
                break
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.minimize(loss_fn, x0, method='L-BFGS-B',
                                      jac=True, options={'maxiter': 1500, 'ftol': 1e-10, 'gtol': 1e-8})
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    U_opt = result.x[:n*m*r].reshape(n*m, r)
                    V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                    best_result = (U_opt, V_opt, W_opt)
                    
                    # Check if we've already found a valid solution
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        # Try to polish
                        polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=3.0)
                        if polished is not None:
                            U_p, V_p, W_p = polished
                            if _verify(U_p, V_p, W_p, n, m, p, tol=1e-8):
                                return U_p, V_p, W_p
                        return U_opt, V_opt, W_opt
                        
            except Exception:
                continue
        
        # If we got a good but not perfect solution, try polishing
        if best_result is not None and best_loss < 1.0:
            U_opt, V_opt, W_opt = best_result
            polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=5.0)
            if polished is not None:
                U_p, V_p, W_p = polished
                if _verify(U_p, V_p, W_p, n, m, p, tol=1e-6):
                    return U_p, V_p, W_p
        
        return None

    # Try ranks from low to high
    # Focus on ranks close to 72 first as they're more likely to succeed
    for r in [71, 70, 69, 68, 67, 66]:
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break
    
    if best_rank < n * m * p:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
# EVOLVE-BLOCK-END
