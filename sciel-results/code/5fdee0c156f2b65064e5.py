"""
Combined approach: Structural Kronecker construction (Strassen 3x3x3) +
Optimization polish with analytic gradients and structured initialization.

The key insight is that the 3x3 x 3x6 multiplication tensor can be viewed
as two independent 3x3x3 blocks (for k=0..2 and k=3..5) sharing the same
U matrix. If we have a rank-7 decomposition for 3x3x3 (Strassen), we can
construct a rank-14 decomposition for 3x3x6 by Kronecker product with a
trivial rank-2 decomposition of the 1x1x2 block selector.

However, Strassen's rank-7 algorithm for 3x3x3 doesn't exist directly.
The best known rank for 3x3x3 over reals is 23 (or possibly lower).

Let's use a different structural approach: 
- For 3x3x3, we can use a rank-19 or rank-20 decomposition if available.
- But since we don't have hardcoded optimal decompositions, let's use
  a hybrid: start with a structured initialization that exploits the
  block structure, then polish with gradient descent.

Actually, let's think about this more carefully. The tensor for 3x3x6 has
54 non-zero entries (one for each (i,j,k) with i,j in 0..2, k in 0..5).
The lower bound is max(9, 18, 18) = 18.

A key structural observation: we can decompose the 3x6 matrix as two 3x3
blocks. The multiplication A(3x3) * B(3x6) = A * [B1 B2] where B1, B2 are
3x3. So the result is [A*B1 A*B2]. This means we need to compute two
3x3 matrix multiplications. If we have a rank-r decomposition for 3x3x3,
we can get a rank-2r decomposition for 3x3x6 by:
- U_shared (9 x 2r) = [U_333, U_333]
- V (18 x 2r): rows 0-8 (for B1) = [V_333, 0], rows 9-17 (for B2) = [0, V_333]
- W (18 x 2r): rows 0-8 (for B1) = [W_333, 0], rows 9-17 (for B2) = [0, W_333]

This gives rank 2r. If we can find a rank-23 decomposition for 3x3x3, we
get rank 46 for 3x3x6. But finding rank-23 for 3x3x3 is itself hard.

Let's try a practical approach: use gradient-based optimization with
structured initialization that exploits the block structure, and try to
find decompositions of rank 46-53.
"""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _build_target_tensor(n, m, p):
    """Build the target tensor T[a,b,c] for matrix multiplication."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify(U, V, W, n, m, p, T_target, tol=1e-9):
    """Verify that the decomposition exactly matches the target tensor."""
    R = np.einsum('ar,br,cr->abc', U, V, W)
    return np.allclose(R, T_target, atol=tol)

def _snap_to_half_ints(U, V, W):
    """Snap entries to nearest half-integer (multiple of 0.5)."""
    U_s = np.round(U * 2) / 2
    V_s = np.round(V * 2) / 2
    W_s = np.round(W * 2) / 2
    return U_s, V_s, W_s

def _obj_and_grad(params, dim_U, dim_V, dim_W, rank, T_target):
    """Objective and analytic gradient for tensor decomposition."""
    U = params[:dim_U * rank].reshape(dim_U, rank)
    V = params[dim_U * rank : dim_U * rank + dim_V * rank].reshape(dim_V, rank)
    W = params[dim_U * rank + dim_V * rank :].reshape(dim_W, rank)
    
    R = np.einsum('ar,br,cr->abc', U, V, W)
    diff = R - T_target
    obj = np.sum(diff ** 2)
    
    # Analytic gradients
    grad_U = 2.0 * np.einsum('abc,br,cr->ar', diff, V, W)
    grad_V = 2.0 * np.einsum('abc,ar,cr->br', diff, U, W)
    grad_W = 2.0 * np.einsum('abc,ar,br->cr', diff, U, V)
    
    grad = np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
    return obj, grad

def _try_rank_optimize(n, m, p, target_rank, T_target, max_restarts=10, max_iter=300):
    """
    Try to find a rank-r decomposition using L-BFGS-B with analytic gradients.
    Uses structured initialization based on the block structure of 3x3x6.
    """
    dim_U, dim_V, dim_W = n * m, m * p, p * n
    best_loss = float('inf')
    best_params = None
    
    for restart in range(max_restarts):
        if time.time() > DEADLINE - 15:
            break
        
        # Structured initialization: exploit the two 3x3x3 block structure
        if restart == 0:
            # Initialize with the "two independent 3x3x3" structure
            # U is shared, V and W are block-diagonal
            U = np.zeros((dim_U, target_rank))
            V = np.zeros((dim_V, target_rank))
            W = np.zeros((dim_W, target_rank))
            
            half = target_rank // 2
            rem = target_rank % 2
            
            # First block (k=0,1,2)
            for r in range(half):
                # Random values for the first block
                U[:, r] = np.random.randn(dim_U) * 0.3
                V[:9, r] = np.random.randn(9) * 0.3  # k=0,1,2 part of V
                W[:9, r] = np.random.randn(9) * 0.3   # k=0,1,2 part of W
                # V[9:, r] = 0, W[9:, r] = 0 (already zero)
            
            # Second block (k=3,4,5)
            for r in range(half, target_rank):
                U[:, r] = np.random.randn(dim_U) * 0.3
                V[9:, r] = np.random.randn(9) * 0.3   # k=3,4,5 part of V
                W[9:, r] = np.random.randn(9) * 0.3   # k=3,4,5 part of W
                # V[:9, r] = 0, W[:9, r] = 0 (already zero)
            
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        else:
            # Random initialization with moderate scale
            np.random.seed(42 + restart * 1000 + target_rank)
            scale = 0.5 / np.sqrt(target_rank)
            U = np.random.randn(dim_U, target_rank) * scale
            V = np.random.randn(dim_V, target_rank) * scale
            W = np.random.randn(dim_W, target_rank) * scale
            params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        # Optimize with L-BFGS-B
        try:
            result = opt.minimize(
                _obj_and_grad,
                params,
                args=(dim_U, dim_V, dim_W, target_rank, T_target),
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iter,
                    'ftol': 1e-14,
                    'gtol': 1e-8,
                    'maxls': 50,
                    'maxfun': 1000
                }
            )
            
            if result.fun < best_loss:
                best_loss = result.fun
                best_params = result.x.copy()
                
            if best_loss < 1e-6:
                break
                
        except Exception as e:
            continue
    
    if best_params is not None and best_loss < 1e-4:
        # Extract and snap to half-integers
        U = best_params[:dim_U * target_rank].reshape(dim_U, target_rank)
        V = best_params[dim_U * target_rank : dim_U * target_rank + dim_V * target_rank].reshape(dim_V, target_rank)
        W = best_params[dim_U * target_rank + dim_V * target_rank :].reshape(dim_W, target_rank)
        
        U_s, V_s, W_s = _snap_to_half_ints(U, V, W)
        
        if _verify(U_s, V_s, W_s, n, m, p, T_target, tol=1e-9):
            return U_s, V_s, W_s
    
    return None

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
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

def construct():
    n, m, p = DIMENSIONS
    T_target = _build_target_tensor(n, m, p)
    
    # Lower bound: max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # Schoolbook rank: 54
    # Try to find decompositions with rank < 54
    lower_bound = max(n * m, m * p, p * n)
    schoolbook_rank = n * m * p
    
    # Try ranks from a reasonable range. Based on the block structure,
    # if 3x3x3 has rank r, then 3x3x6 has rank 2r.
    # Known: 3x3x3 rank is at most 27 (schoolbook), possibly 23.
    # So 3x3x6 rank is at most 54, possibly 46.
    # Try ranks 46 to 53 first, then fall back to 54.
    
    # Order of ranks to try (lower is better)
    ranks_to_try = list(range(46, 54))
    # Also try some ranks between lower bound and 46 in case of surprises
    for r in range(lower_bound, 46):
        if r not in ranks_to_try:
            ranks_to_try.append(r)
    ranks_to_try.sort()
    
    for target_rank in ranks_to_try:
        if time.time() > DEADLINE - 20:
            break
        
        # Number of restarts depends on how far we are from the deadline
        time_remaining = DEADLINE - 20 - time.time()
        if time_remaining < 10:
            break
            
        # More restarts for lower ranks (harder problems)
        if target_rank < 50:
            max_restarts = 8
        else:
            max_restarts = 5
            
        result = _try_rank_optimize(n, m, p, target_rank, T_target, 
                                     max_restarts=max_restarts, max_iter=200)
        if result is not None:
            U, V, W = result
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fall back to schoolbook
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
