"""Matrix multiplication 3x4x6 - optimized tensor decomposition search.

Strategy:
- The trivial rank is 72 (3*4*6).
- We attempt to find a rank < 72 exact decomposition using:
  1. A Kronecker-structured initialization exploiting bilinear structure.
  2. Multiple diverse initializations (structured, block-diagonal, random).
  3. L-BFGS-B optimization with tight tolerances.
  4. A final high-precision polish pass.
- If no lower rank is found within time budget, return the trivial rank-72 solution.
"""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 4, 6]


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


def _verify_fast(U, V, W, n, m, p, tol=1e-5):
    """Faster verification with looser tolerance."""
    Ut = U.reshape(n, m, -1)
    Vt = V.reshape(m, p, -1)
    Wt = W.reshape(p, n, -1)
    T = np.einsum('ijc,jkc,kic->ijk', Ut, Vt, Wt)
    return np.max(np.abs(T - 1.0)) < tol


def _loss_and_grad(x, n, m, p, r):
    """Compute loss and gradient for given parameters."""
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    W = x[n*m*r + m*p*r :].reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    loss = np.sum(err ** 2)

    # Gradients using chain rule
    dU = np.einsum('ijk,jkc,kic->ijc', err, V, W) * 2
    dV = np.einsum('ijk,ijc,kic->jkc', err, U, W) * 2
    dW = np.einsum('ijk,ijc,jkc->kic', err, U, V) * 2

    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    return loss, grad


def _loss_only(x, n, m, p, r):
    """Compute loss only (no gradient) for BFGS."""
    U = x[:n*m*r].reshape(n, m, r)
    V = x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
    W = x[n*m*r + m*p*r :].reshape(p, n, r)

    T = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    err = T - 1.0
    return np.sum(err ** 2)


def _init_kronecker(n, m, p, r, seed=42):
    """
    Kronecker-structured initialization.
    
    For 3x4x6, we can think of the multiplication tensor as having structure
    where U depends on (i,j), V on (j,k), W on (k,i).
    
    We use a block structure: for each pair (i,k), we have factors that
    cover all j. This gives a natural grouping.
    
    Total (i,k) pairs: 3*6 = 18
    Total j: 4
    So 18*4 = 72 trivial factors.
    
    For rank reduction, we try to merge factors that share (i,k) or (j,k) or (i,j).
    """
    rng = np.random.RandomState(seed)

    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))

    if r >= n * m * p:
        # Use trivial decomposition for first n*m*p factors
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U0[i * m + j, col] = 1.0
                        V0[j * p + k, col] = 1.0
                        W0[k * n + i, col] = 1.0
                        col += 1
        # Add small random perturbations to remaining
        for c in range(n * m * p, r):
            U0[:, c] = rng.randn(n * m) * 0.05
            V0[:, c] = rng.randn(m * p) * 0.05
            W0[:, c] = rng.randn(p * n) * 0.05
    else:
        # For rank reduction: try to create structure
        # Strategy: group by (i, k) pairs, share across j
        
        # There are n*p = 18 (i,k) pairs
        # For each (i,k), we need to cover all m=4 j values
        # If we use ceil(r / (n*p)) factors per (i,k) group
        
        factors_per_ik = max(1, r // (n * p))
        extra = r % (n * p)
        
        col = 0
        for i in range(n):
            for k in range(p):
                block_size = factors_per_ik + (1 if i * p + k < extra else 0)
                for c in range(block_size):
                    # This factor serves (i, k) for all j
                    for j in range(m):
                        U0[i * m + j, col] += 1.0 / m
                        V0[j * p + k, col] += 1.0
                        W0[k * n + i, col] += 1.0 / m
                    col += 1
        
        # Fill remaining columns with random
        if col < r:
            for c in range(col, r):
                U0[:, c] = rng.randn(n * m) * 0.1
                V0[:, c] = rng.randn(m * p) * 0.1
                W0[:, c] = rng.randn(p * n) * 0.1
        
        # Add small perturbations to break symmetry
        U0 += rng.randn(n * m, r) * 0.05
        V0 += rng.randn(m * p, r) * 0.05
        W0 += rng.randn(p * n, r) * 0.05

    return U0, V0, W0


def _init_by_j(n, m, p, r, seed=42):
    """
    Initialize by grouping on j index.
    For each j, we have n*p (i,k) pairs.
    """
    rng = np.random.RandomState(seed)

    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))

    if r >= n * m * p:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U0[i * m + j, col] = 1.0
                        V0[j * p + k, col] = 1.0
                        W0[k * n + i, col] = 1.0
                        col += 1
        for c in range(n * m * p, r):
            U0[:, c] = rng.randn(n * m) * 0.05
            V0[:, c] = rng.randn(m * p) * 0.05
            W0[:, c] = rng.randn(p * n) * 0.05
    else:
        # Group by j: m=4 groups, each with n*p=18 factors
        factors_per_j = max(1, r // m)
        extra = r % m
        
        col = 0
        for j in range(m):
            block_size = factors_per_j + (1 if j < extra else 0)
            for c in range(block_size):
                for i in range(n):
                    for k in range(p):
                        U0[i * m + j, col] += 1.0 / (n * p)
                        V0[j * p + k, col] += 1.0 / (n * p)
                        W0[k * n + i, col] += 1.0 / (n * p)
                col += 1
        
        if col < r:
            for c in range(col, r):
                U0[:, c] = rng.randn(n * m) * 0.1
                V0[:, c] = rng.randn(m * p) * 0.1
                W0[:, c] = rng.randn(p * n) * 0.1
        
        U0 += rng.randn(n * m, r) * 0.05
        V0 += rng.randn(m * p, r) * 0.05
        W0 += rng.randn(p * n, r) * 0.05

    return U0, V0, W0


def _init_random(n, m, p, r, seed=42):
    """Random initialization with appropriate scaling."""
    rng = np.random.RandomState(seed)
    scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
    U0 = rng.randn(n * m, r) * scale
    V0 = rng.randn(m * p, r) * scale
    W0 = rng.randn(p * n, r) * scale
    return U0, V0, W0


def _init_structured_merge(n, m, p, r, seed=42):
    """
    Start from trivial and merge nearby terms.
    Each factor covers a subset of (i,j,k) tuples.
    """
    rng = np.random.RandomState(seed)

    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))

    if r >= n * m * p:
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if col < r:
                        U0[i * m + j, col] = 1.0
                        V0[j * p + k, col] = 1.0
                        W0[k * n + i, col] = 1.0
                        col += 1
        for c in range(n * m * p, r):
            U0[:, c] = rng.randn(n * m) * 0.05
            V0[:, c] = rng.randn(m * p) * 0.05
            W0[:, c] = rng.randn(p * n) * 0.05
    else:
        # Create assignments of (i,j,k) to factors
        assignments = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    assignments.append((i, j, k))
        
        rng.shuffle(assignments)
        
        # Assign round-robin to r factors
        for idx, (i, j, k) in enumerate(assignments):
            c = idx % r
            U0[i * m + j, c] += 1.0
            V0[j * p + k, c] += 1.0
            W0[k * n + i, c] += 1.0
        
        U0 += rng.randn(n * m, r) * 0.3
        V0 += rng.randn(m * p, r) * 0.3
        W0 += rng.randn(p * n, r) * 0.3

    return U0, V0, W0


def _polish(U, V, W, n, m, p, r, max_time=5.0):
    """High-precision polish pass using L-BFGS-B."""
    start = time.time()
    x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])

    try:
        result = opt.minimize(
            lambda x: _loss_and_grad(x, n, m, p, r),
            x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 5000, 'ftol': 1e-16, 'gtol': 1e-14, 'maxfun': 20000}
        )

        if time.time() - start > max_time:
            return None

        U_opt = result.x[:n*m*r].reshape(n*m, r)
        V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)

        return U_opt, V_opt, W_opt
    except Exception:
        return None


def _optimize_from_init(U_init, V_init, W_init, n, m, p, r, max_time=8.0):
    """Run L-BFGS-B from a given initialization."""
    start = time.time()
    x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])

    try:
        result = opt.minimize(
            lambda x: _loss_and_grad(x, n, m, p, r),
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )

        if time.time() - start > max_time:
            return None, np.inf

        U_opt = result.x[:n*m*r].reshape(n*m, r)
        V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
        loss = result.fun

        return (U_opt, V_opt, W_opt), loss

    except Exception:
        return None, np.inf


def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 72

    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank

    def try_rank(r, max_time_per_attempt=8.0):
        """Try to find a rank-r decomposition using multiple strategies."""
        if time.time() > deadline - 15:
            return None

        best_result = None
        best_loss = np.inf

        # Generate multiple diverse initializations
        inits = []

        # Strategy 1: Kronecker-structured by (i,k)
        U0, V0, W0 = _init_kronecker(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))

        # Strategy 2: Group by j
        U0, V0, W0 = _init_by_j(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))

        # Strategy 3: Structured merge
        U0, V0, W0 = _init_structured_merge(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))

        # Strategy 4-6: Random with different seeds
        for seed in [0, 1, 2]:
            U0, V0, W0 = _init_random(n, m, p, r, seed=seed)
            inits.append((U0, V0, W0))

        # Strategy 7-8: More structured with different seeds
        U0, V0, W0 = _init_kronecker(n, m, p, r, seed=99)
        inits.append((U0, V0, W0))
        U0, V0, W0 = _init_by_j(n, m, p, r, seed=99)
        inits.append((U0, V0, W0))

        for U_init, V_init, W_init in inits:
            if time.time() > deadline - 15:
                break

            result, loss = _optimize_from_init(U_init, V_init, W_init, n, m, p, r,
                                                max_time=max_time_per_attempt)

            if result is not None:
                U_opt, V_opt, W_opt = result
                if loss < best_loss:
                    best_loss = loss
                    best_result = (U_opt, V_opt, W_opt)

                    # Check if we've found a valid solution
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        # Try to polish for higher precision
                        polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=5.0)
                        if polished is not None:
                            U_p, V_p, W_p = polished
                            if _verify(U_p, V_p, W_p, n, m, p, tol=1e-8):
                                return U_p, V_p, W_p
                        return U_opt, V_opt, W_opt

        # If we got a good but not perfect solution, try polishing
        if best_result is not None and best_loss < 0.5:
            U_opt, V_opt, W_opt = best_result
            polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=5.0)
            if polished is not None:
                U_p, V_p, W_p = polished
                if _verify(U_p, V_p, W_p, n, m, p, tol=1e-6):
                    return U_p, V_p, W_p

        return None

    # Try ranks from 71 down to 60
    for r in range(trivial_rank - 1, max(trivial_rank - 15, 55), -1):
        if time.time() > deadline - 20:
            break

        result = try_rank(r)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W, n, m, p, tol=1e-6):
                best = {'U': U, 'V': V, 'W': W}
                best_rank = r
                break

    if best_rank < trivial_rank:
        return best
    else:
        return {'U': U_trivial.tolist(), 'V': V_trivial.tolist(), 'W': W_trivial.tolist()}
