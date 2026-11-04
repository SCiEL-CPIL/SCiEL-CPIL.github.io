"""Matrix multiplication 2x4x5 rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]
n, m, p = DIMENSIONS
RANK_SCHOOLBOOK = n * m * p  # 40

# Deadline guard
_START = time.time()
DEADLINE = 150  # seconds, evaluator kills at 150s, keep safety margin
SAFETY_MARGIN = 10  # seconds


def _elapsed():
    return time.time() - _START


def _time_left():
    return DEADLINE - SAFETY_MARGIN - _elapsed()


def _validate(U, V, W):
    """Check that U,V,W exactly decompose the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # Tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For exact matrix multiplication, T[i,j,k] = 1 for all (i,j,k)
    # Build the tensor from the decomposition
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # This is a 3-tensor contraction.
    # Efficient: for each c, outer product of U[:,c], V[:,c], W[:,c]
    # But we can check by evaluating on random inputs or directly.
    # Direct check: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    # Number of (i,j,k) = 2*4*5 = 40, rank up to 40, so 40*40 = 1600 mults. Fast.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _schoolbook():
    """Return the schoolbook decomposition of rank 40."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(r):
    """
    Try to find an exact decomposition of rank r using integer/half-integer entries.
    We use a structured search: start from schoolbook and try to merge/eliminate terms.
    
    For 2x4x5, the known lower bound is around 20-25. We attempt to find
    a decomposition with rank < 40.
    
    Strategy: Use the fact that we can combine terms. The matrix multiplication
    tensor T_{i,j,k} = 1 for all i,j,k. We need:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    This is equivalent to: the bilinear form (in the middle index j) for each (i,k)
    must be the all-ones matrix of size m x p... wait no.
    
    Actually, let's think of it differently. For fixed i and k, the equation is:
      sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all j.
    
    Let's define for each c: u_c is a vector of length n*m, v_c of length m*p, w_c of length p*n.
    
    Alternative approach: use the "partition" method. If we can write the multiplication
    as a sum of r rank-1 tensors, we're done.
    
    For small dimensions, we can try to find a decomposition using the following:
    - The tensor is T[i,j,k] = 1 (all ones).
    - We need sum_c (a_c ⊗ b_c ⊗ c_c)[i,j,k] = 1 where
      a_c[i] = U[i*m + j, c]... no, this doesn't factor cleanly because U depends on both i and j.
    
    Let me re-index. The standard tensor for matrix mult of (n x m) * (m x p) is:
      T[i, j, k] = 1  (the coefficient of A[i,j]*B[j,k] in C[i,k])
    
    A rank-r decomposition means:
      T[i,j,k] = sum_{c=1}^r U_c[i,j] * V_c[j,k] * W_c[k,i]
    
    where U_c is n x m, V_c is m x p, W_c is p x n.
    
    For all-ones tensor, one approach: use the fact that 1 = sum of products.
    
    Key insight: If we can find matrices U_c (n x m), V_c (m x p), W_c (p x n) such that
    for all (i,j,k): sum_c U_c[i,j] * V_c[j,k] * W_c[k,i] = 1
    
    A simple construction: let U_c[i,j] = a_c[i] * b_c[j], V_c[j,k] = b_c[j] * c_c[k], W_c[k,i] = c_c[k] * a_c[i].
    Then sum_c a_c[i]^2 * b_c[j]^2 * c_c[k]^2 = 1 for all i,j,k.
    This requires: sum_c a_c[i]^2 * b_c[j]^2 * c_c[k]^2 = 1.
    By separability, we need sum_c a_c[i]^2 = 1 for all i, sum_c b_c[j]^2 = 1 for all j, sum_c c_c[k]^2 = 1 for all k,
    AND the product must equal 1. This is only possible if each term is 1, which gives rank = n*m*p.
    
    So the separable approach doesn't help. We need non-separable (non-diagonal) U_c, V_c, W_c.
    
    Let's try a different approach: use the known results. For 2x2x2, rank 7 (Strassen).
    For 2x2x3, the rank is known to be 11 or 12.
    For 2x4x5, we can try to compose smaller decompositions.
    
    Actually, let's try a direct search for small rank improvements. Start with rank 39 and see if we can
    make one of the schoolbook terms vanish by adjusting others.
    
    Given the time constraint, let's try a more practical approach:
    Use the "border rank" or try to find a decomposition with rank slightly less than 40.
    
    For this specific problem, let me try to use the following trick:
    Note that T[i,j,k] = 1. Consider grouping by i. For fixed i, we have an m x p all-ones matrix.
    The bilinear complexity of multiplying a 2x4 by 4x5 is what we want.
    
    Let's try rank 35-39 with a structured perturbation.
    """
    if r >= RANK_SCHOOLBOOK:
        return None
    
    # Try a structured approach: use the fact that we can share computations.
    # For 2x4x5, think of it as two 1x4x5 multiplications (for the two rows of A).
    # A 1x4x5 multiplication has rank 20 (schoolbook). Two of them would be 40.
    # But we can share the 4x5 part partially.
    
    # Alternative: think of A as 2x4, B as 4x5.
    # C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0] + A[0,2]B[2,0] + A[0,3]B[3,0]
    # C[0,1] = A[0,0]B[0,1] + A[0,1]B[1,1] + A[0,2]B[2,1] + A[0,3]B[3,1]
    # ... etc.
    
    # Let's try to find a decomposition using numpy optimization.
    # We'll parameterize U, V, W with rank r and minimize the error, then check for exactness.
    
    if r < 1:
        return None
    
    # Initialize with schoolbook truncated (first r terms)
    U0 = np.zeros((n * m, r))
    V0 = np.zeros((m * p, r))
    W0 = np.zeros((p * n, r))
    for c in range(min(r, RANK_SCHOOLBOOK)):
        i = c // (m * p)
        rem = c % (m * p)
        j = rem // p
        k = rem % p
        U0[i * m + j, c] = 1
        V0[j * p + k, c] = 1
        W0[k * n + i, c] = 1
    
    # Use scipy to optimize
    try:
        from scipy.optimize import minimize
    except ImportError:
        return None
    
    # Flatten parameters
    x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
    
    def objective(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r:].reshape(p*n, r)
        # Compute T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # Error = sum of (T[i,j,k] - 1)^2
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (s - 1.0) ** 2
        return err
    
    def gradient(x):
        U = x[:n*m*r].reshape(n*m, r)
        V = x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
        W = x[n*m*r + m*p*r:].reshape(p*n, r)
        gU = np.zeros_like(U)
        gV = np.zeros_like(V)
        gW = np.zeros_like(W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    d = 2 * (s - 1.0)
                    gU[i*m+j, :] += d * V[j*p+k, :] * W[k*n+i, :]
                    gV[j*p+k, :] += d * U[i*m+j, :] * W[k*n+i, :]
                    gW[k*n+i, :] += d * U[i*m+j, :] * V[j*p+k, :]
        return np.concatenate([gU.flatten(), gV.flatten(), gW.flatten()])
    
    # Try a few random initializations
    best_result = None
    np.random.seed(42)
    n_attempts = 3
    for attempt in range(n_attempts):
        if _time_left() < 5:
            break
        if attempt == 0:
            x_start = x0.copy()
        else:
            # Random perturbation of schoolbook
            U_rand = np.zeros((n*m, r))
            V_rand = np.zeros((m*p, r))
            W_rand = np.zeros((p*n, r))
            for c in range(min(r, RANK_SCHOOLBOOK)):
                i = c // (m * p)
                rem = c % (m * p)
                j = rem // p
                k = rem % p
                U_rand[i * m + j, c] = 1 + np.random.normal(0, 0.1)
                V_rand[j * p + k, c] = 1 + np.random.normal(0, 0.1)
                W_rand[k * n + i, c] = 1 + np.random.normal(0, 0.1)
            # Add some randomness to other entries
            U_rand += np.random.normal(0, 0.05, (n*m, r))
            V_rand += np.random.normal(0, 0.05, (m*p, r))
            W_rand += np.random.normal(0, 0.05, (p*n, r))
            x_start = np.concatenate([U_rand.flatten(), V_rand.flatten(), W_rand.flatten()])
        
        try:
            res = minimize(objective, x_start, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            if res.success and res.fun < 1e-8:
                U_opt = res.x[:n*m*r].reshape(n*m, r)
                V_opt = res.x[n*m*r:n*m*r + m*p*r].reshape(m*p, r)
                W_opt = res.x[n*m*r + m*p*r:].reshape(p*n, r)
                # Round to nearest half-integer
                U_round = np.round(2 * U_opt) / 2
                V_round = np.round(2 * V_opt) / 2
                W_round = np.round(2 * W_opt) / 2
                if _validate(U_round.tolist(), V_round.tolist(), W_round.tolist()):
                    return {"U": U_round.tolist(), "V": V_round.tolist(), "W": W_round.tolist()}
        except Exception:
            pass
    
    return None


def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4x5 matrix multiplication."""
    # Start with schoolbook (rank 40)
    best = _schoolbook()
    best_rank = RANK_SCHOOLBOOK
    
    # Try to find lower rank decompositions
    for r in range(RANK_SCHOOLBOOK - 1, 15, -1):
        if _time_left() < 15:
            break
        result = _try_rank(r)
        if result is not None:
            best = result
            best_rank = r
            break  # Found a lower rank, stop
    
    return best
