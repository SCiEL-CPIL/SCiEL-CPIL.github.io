"""Matrix multiplication 3x4x6 - optimized tensor decomposition search."""
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

def _structured_init_v2(n, m, p, r, seed=42):
    """
    Improved structured initialization that tries to exploit bilinear structure.
    Uses a more principled approach: start from trivial and merge factors
    that share structure in the (i,j,k) tensor.
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
        # For rank reduction: use a smarter merging strategy
        # Start with trivial factors, then merge pairs that can be combined
        
        # First, create all trivial factors
        all_factors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    all_factors.append((i, j, k))
        
        # We need to reduce from n*m*p factors to r factors
        # Strategy: greedily merge factors that share indices
        
        # Create a cost matrix for merging two factors
        # Factors (i1,j1,k1) and (i2,j2,k2) can be merged if they share
        # at least one index (i, j, or k)
        
        # Start with all factors, then iteratively merge the "cheapest" pairs
        current_factors = list(all_factors)
        
        while len(current_factors) > r:
            if len(current_factors) <= 1:
                break
                
            # Find the pair with the most shared indices
            best_pair = None
            best_score = -1
            
            for idx1 in range(len(current_factors)):
                for idx2 in range(idx1 + 1, len(current_factors)):
                    i1, j1, k1 = current_factors[idx1]
                    i2, j2, k2 = current_factors[idx2]
                    
                    # Score based on shared indices
                    score = 0
                    if i1 == i2:
                        score += 2
                    if j1 == j2:
                        score += 2
                    if k1 == k2:
                        score += 2
                    
                    if score > best_score:
                        best_score = score
                        best_pair = (idx1, idx2)
            
            if best_pair is None or best_score == 0:
                # No good merges left, just take first r factors
                current_factors = current_factors[:r]
                break
            
            # Merge the best pair: combine their contributions
            idx1, idx2 = best_pair
            i1, j1, k1 = current_factors[idx1]
            i2, j2, k2 = current_factors[idx2]
            
            # Remove both and add a merged factor
            # The merged factor will have contributions from both
            # We'll represent this by keeping track of multiple (i,j,k) per factor
            current_factors.pop(idx2)
            current_factors.pop(idx1)
            
            # Add a special marker for merged factors
            # For now, just keep one of them and we'll handle the rest in init
            current_factors.append((i1, j1, k1))  # Keep first
            # The second factor's contribution will be added as perturbation
        
        # Now initialize: each factor in current_factors gets a base contribution
        # If we have fewer than r factors, we can split some
        if len(current_factors) < r:
            # Split some factors to reach r
            while len(current_factors) < r:
                # Pick a factor and split it
                idx = rng.randint(len(current_factors))
                i, j, k = current_factors[idx]
                # Create a perturbed version
                current_factors.append((i, j, k))
        
        # Take first r factors
        current_factors = current_factors[:r]
        
        # Initialize each factor
        for c, (i, j, k) in enumerate(current_factors):
            U0[i * m + j, c] = 1.0
            V0[j * p + k, c] = 1.0
            W0[k * n + i, c] = 1.0
        
        # Add structured perturbations based on shared structure
        # Factors that share indices should have correlated perturbations
        for c in range(r):
            i, j, k = current_factors[c]
            # Perturbation that respects the structure
            U0[i * m + j, c] += rng.randn() * 0.1
            V0[j * p + k, c] += rng.randn() * 0.1
            W0[k * n + i, c] += rng.randn() * 0.1
    
    return U0, V0, W0

def _random_init_v2(n, m, p, r, seed=42):
    """Random initialization with better scaling."""
    rng = np.random.RandomState(seed)
    # Scale based on expected contribution per factor
    scale = 1.0 / np.sqrt(max(1, r / (n * m * p)))
    U0 = rng.randn(n * m, r) * scale
    V0 = rng.randn(m * p, r) * scale
    W0 = rng.randn(p * n, r) * scale
    return U0, V0, W0

def _block_diagonal_init(n, m, p, r, seed=42):
    """
    Block diagonal initialization: group by i-index, then j-index.
    This respects the structure that U depends on (i,j), V on (j,k), W on (k,i).
    """
    rng = np.random.RandomState(seed)
    
    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))
    
    # Try to create block structure where factors are grouped
    # by which (i,j) or (j,k) or (k,i) they primarily serve
    
    # Assign factors to blocks based on i
    factors_per_i = r // n
    extra = r % n
    
    col = 0
    for i in range(n):
        block_size = factors_per_i + (1 if i < extra else 0)
        for c in range(block_size):
            # This factor primarily serves row i
            # It can contribute to any (i, j, k) tuple
            for j in range(m):
                for k in range(p):
                    # Weight based on how well this factor fits
                    weight = 1.0 / (m * p)
                    U0[i * m + j, col] += weight
                    V0[j * p + k, col] += weight
                    W0[k * n + i, col] += weight
            col += 1
    
    # Add small random perturbations
    U0 += rng.randn(n * m, r) * 0.1
    V0 += rng.randn(m * p, r) * 0.1
    W0 += rng.randn(p * n, r) * 0.1
    
    return U0, V0, W0

def _polish(U, V, W, n, m, p, r, max_time=5.0):
    """High-precision polish pass with multiple stages."""
    start = time.time()
    x = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    try:
        # Stage 1: L-BFGS-B with moderate tolerance
        result1 = opt.minimize(
            lambda x: _loss_and_grad(x, n, m, p, r),
            x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 2000}
        )
        
        if time.time() - start > max_time * 0.5:
            return None
        
        x = result1.x
        
        # Stage 2: Continue with tighter tolerance
        result2 = opt.minimize(
            lambda x: _loss_and_grad(x, n, m, p, r),
            x,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13, 'maxfun': 5000}
        )
        
        if time.time() - start > max_time:
            return None
        
        U_opt = result2.x[:n*m*r].reshape(n*m, r)
        V_opt = result2.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
        W_opt = result2.x[n*m*r + m*p*r :].reshape(p*n, r)
        
        return U_opt, V_opt, W_opt
    except Exception:
        return None

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    n, m, p = DIMENSIONS
    trivial_rank = n * m * p  # 72
    
    # Start with trivial decomposition as fallback
    U_trivial, V_trivial, W_trivial = _trivial_decomposition(n, m, p)
    best = {'U': U_trivial, 'V': V_trivial, 'W': W_trivial}
    best_rank = trivial_rank
    
    def try_rank(r, max_time_per_attempt=6.0):
        """Try to find a rank-r decomposition using multiple strategies."""
        if time.time() > deadline - 10:
            return None
        
        best_result = None
        best_loss = np.inf
        
        # Generate multiple initialization strategies
        inits = []
        
        # Strategy 1: Structured v2 (primary)
        U0, V0, W0 = _structured_init_v2(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))
        
        # Strategy 2: Block diagonal
        U0, V0, W0 = _block_diagonal_init(n, m, p, r, seed=42)
        inits.append((U0, V0, W0))
        
        # Strategy 3: Random with different seeds
        for seed in [0, 1, 2]:
            U0, V0, W0 = _random_init_v2(n, m, p, r, seed=seed)
            inits.append((U0, V0, W0))
        
        # Strategy 4: Structured with different seed
        U0, V0, W0 = _structured_init_v2(n, m, p, r, seed=99)
        inits.append((U0, V0, W0))
        
        # Strategy 5: Near-trivial perturbation
        U0, V0, W0 = _trivial_decomposition(n, m, p)
        rng = np.random.RandomState(123)
        # Perturb only a subset of entries to create near-trivial init
        for c in range(r):
            if c < n * m * p:
                # Keep most entries, perturb some
                mask = rng.rand(n * m) < 0.3
                U0[:, c] += mask * rng.randn(n * m) * 0.1
                mask = rng.rand(m * p) < 0.3
                V0[:, c] += mask * rng.randn(m * p) * 0.1
                mask = rng.rand(p * n) < 0.3
                W0[:, c] += mask * rng.randn(p * n) * 0.1
        inits.append((U0[:n*m, :r], V0[:m*p, :r], W0[:p*n, :r]))
        
        for U_init, V_init, W_init in inits:
            if time.time() > deadline - 10:
                break
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                result = opt.minimize(
                    lambda x: _loss_and_grad(x, n, m, p, r),
                    x0,
                    method='L-BFGS-B',
                    jac=True,
                    options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 8000}
                )
                
                if result.fun < best_loss:
                    best_loss = result.fun
                    U_opt = result.x[:n*m*r].reshape(n*m, r)
                    V_opt = result.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_opt = result.x[n*m*r + m*p*r :].reshape(p*n, r)
                    best_result = (U_opt, V_opt, W_opt)
                    
                    # Check if we've found a valid solution
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        # Try to polish for higher precision
                        polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=3.0)
                        if polished is not None:
                            U_p, V_p, W_p = polished
                            if _verify(U_p, V_p, W_p, n, m, p, tol=1e-8):
                                return U_p, V_p, W_p
                        return U_opt, V_opt, W_opt
                        
            except Exception:
                continue
        
        # If we got a good but not perfect solution, try polishing
        if best_result is not None and best_loss < 0.5:
            U_opt, V_opt, W_opt = best_result
            polished = _polish(U_opt, V_opt, W_opt, n, m, p, r, max_time=5.0)
            if polished is not None:
                U_p, V_p, W_p = polished
                if _verify(U_p, V_p, W_p, n, m, p, tol=1e-6):
                    return U_p, V_p, W_p
        
        return None
    
    # Try ranks from 71 down to 65
    # Focus more time on ranks close to 72, less on lower ranks
    # Use adaptive time allocation: more time for higher ranks
    time_per_rank = 8.0  # Base time per rank attempt
    
    for r in range(trivial_rank - 1, max(trivial_rank - 10, 60), -1):
        if time.time() > deadline - 15:
            break
        
        # Allocate more time to higher ranks (closer to trivial)
        rank_priority = (trivial_rank - r) / 10.0  # 0 for r=71, 1 for r=61
        adjusted_time = time_per_rank * (1.0 + 0.5 * rank_priority)
        
        result = try_rank(r, max_time_per_attempt=adjusted_time)
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
