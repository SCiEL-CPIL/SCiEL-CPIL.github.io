"""Matrix multiplication 4,4,7 tensor decomposition - structured search."""
import time
import numpy as np

DIMENSIONS = [4, 4, 7]
SEED = 42

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that the decomposition exactly reproduces matrix multiplication."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    if U_mat.shape != (n * m, W_mat.shape[1]):
        return False
    if V_mat.shape != (m * p, W_mat.shape[1]):
        return False
    if W_mat.shape != (p * n, U_mat.shape[1]):
        return False
    if U_mat.shape[1] != V_mat.shape[1] or V_mat.shape[1] != W_mat.shape[1]:
        return False
    
    rank = U_mat.shape[1]
    
    # Verify target entries: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    
    # Verify non-target entries are zero
    # Total possible (alpha, beta, gamma) triples: (n*m)*(m*p)*(p*n) = 16*28*28 = 12544
    # Target entries: 112
    # We check a subset of non-target entries for efficiency
    target_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                target_set.add((i * m + j, j * p + k, k * n + i))
    
    # Check all non-target entries (12544 - 112 = 12432, manageable)
    for alpha in range(n * m):
        for beta in range(m * p):
            for gamma in range(p * n):
                if (alpha, beta, gamma) not in target_set:
                    val = np.sum(U_mat[alpha, :] * V_mat[beta, :] * W_mat[gamma, :])
                    if abs(val) > tol:
                        return False
    return True


def _verify_partial(U, V, W, n, m, p, tol=1e-9):
    """Faster verification - only checks target entries."""
    U_mat = np.array(U, dtype=float)
    V_mat = np.array(V, dtype=float)
    W_mat = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U_mat[i * m + j, :] * V_mat[j * p + k, :] * W_mat[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _build_strassen_4x4_4x7(n, m, p):
    """
    Try a Strassen-inspired decomposition for 4x4 * 4x7.
    
    Split A (4x4) into four 2x2 blocks:
      A = [[A11, A12], [A21, A22]]
    
    Split B (4x7) into four 2x? blocks by rows:
      B = [[B1], [B2]] where B1 is 2x7, B2 is 2x7
    
    Then C = A*B = [[A11*B1 + A12*B2], [A21*B1 + A22*B2]]
    
    Each 2x2 * 2x7 product can be done with Strassen (7 multiplications instead of 8).
    So total: 4 * 7 = 28 scalar multiplications... but that's for the block structure.
    
    Actually, let me think differently. The tensor rank for 2x2*2x2 is 7 (Strassen).
    For 2x2*2x7, we can view it as 2x2*2x2 done 3.5 times... not quite.
    
    Alternative: use the fact that 4 = 2+2, and apply Strassen at the block level.
    
    Let me try a different approach: direct algebraic construction.
    """
    # For now, return None and use other methods
    return None


def _random_search(rank, n, m, p, rng, max_iters=1000, deadline=None):
    """Random search for a valid decomposition at given rank."""
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        U = rng.randn(n * m, rank)
        V = rng.randn(m * p, rank)
        W = rng.randn(p * n, rank)
        
        # Quick check on target entries
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U = U.copy()
            best_V = V.copy()
            best_W = W.copy()
        
        if best_err < 1e-9:
            return best_U, best_V, best_W
    
    return None, None, None


def _gradient_descent(U, V, W, n, m, p, lr=0.01, max_iters=500, deadline=None):
    """Simple gradient descent to minimize the error."""
    U = U.astype(float).copy()
    V = V.astype(float).copy()
    W = W.astype(float).copy()
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    # d/dU[u_idx,r] = (U[u_idx,r]*V[v_idx,r]*W[w_idx,r] - 1) * V[v_idx,r] * W[w_idx,r]
                    # Simplified: gradient of (sum_r U[u,r]*V[v,r]*W[w,r] - 1)^2
                    val = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
                    diff = val - 1.0
                    dU[u_idx, :] += 2 * diff * V[v_idx, :] * W[w_idx, :]
                    dV[v_idx, :] += 2 * diff * U[u_idx, :] * W[w_idx, :]
                    dW[w_idx, :] += 2 * diff * U[u_idx, :] * V[v_idx, :]
        
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    err += (val - 1.0) ** 2
        
        if err < 1e-12:
            break
    
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Start with schoolbook
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_rank = U_school.shape[1]
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    
    rng = np.random.RandomState(SEED)
    
    # Try structured approach: exploit the 4x4 symmetry
    # The key insight: for each fixed j, the terms A[i,j]*B[j,k] for all i,k form a rank-1 
    # contribution pattern. We can try to share computation across different j values.
    
    # Strategy: Try ranks from 111 down, using random initialization + gradient descent
    # Focus on ranks where structured solutions might exist
    
    # First, try a few promising rank values with structured initialization
    candidate_ranks = [111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 100, 96, 90, 84, 80]
    
    for trial_rank in candidate_ranks:
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        found = False
        for restart in range(5):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            U_try = rng.randn(n * m, trial_rank)
            V_try = rng.randn(m * p, trial_rank)
            W_try = rng.randn(p * n, trial_rank)
            
            # Gradient descent
            U_try, V_try, W_try = _gradient_descent(
                U_try, V_try, W_try, n, m, p,
                lr=0.001, max_iters=200, deadline=deadline - 10
            )
            
            # Verify
            if _verify_partial(U_try, V_try, W_try, n, m, p, tol=1e-8):
                # Full verification
                if _verify(U_try, V_try, W_try, n, m, p, tol=1e-9):
                    best_rank = trial_rank
                    best_result = {"U": U_try.tolist(), "V": V_try.tolist(), "W": W_try.tolist()}
                    found = True
                    break
        
        if found:
            # Try to go even lower
            continue
    
    # If we found a better solution, try to polish it
    if best_rank < n * m * p:
        U_polish = np.array(best_result["U"], dtype=float)
        V_polish = np.array(best_result["V"], dtype=float)
        W_polish = np.array(best_result["W"], dtype=float)
        
        U_polish, V_polish, W_polish = _gradient_descent(
            U_polish, V_polish, W_polish, n, m, p,
            lr=0.0001, max_iters=100, deadline=deadline - 5
        )
        
        if _verify(U_polish, V_polish, W_polish, n, m, p, tol=1e-10):
            best_result = {"U": U_polish.tolist(), "V": V_polish.tolist(), "W": W_polish.tolist()}
    
    return best_result
# EVOLVE-BLOCK-END
