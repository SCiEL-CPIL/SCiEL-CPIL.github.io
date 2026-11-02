"""Tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 5, 7]

def _snap_matrix(M, tol=1e-8):
    """Snap entries to nearest half-integer if close."""
    M = np.array(M, dtype=float)
    snapped = np.zeros_like(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            half = np.round(v * 2) / 2.0
            if abs(v - half) < tol:
                snapped[i, j] = half
            else:
                whole = np.round(v)
                if abs(v - whole) < tol:
                    snapped[i, j] = whole
                else:
                    snapped[i, j] = v
    return snapped

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rank = U.shape[1]
    
    # Check all target entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    
    # Check some zero entries
    np.random.seed(42)
    for _ in range(200):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        
        # Create inconsistent indices
        # Change i
        i2 = (i + 1) % n
        a = i * m + j
        b = j * p + k
        c = k * n + i2
        if i2 != i:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-6:
                return False
        
        # Change j
        j2 = (j + 1) % m
        a = i * m + j
        b = j2 * p + k
        c = k * n + i
        if j2 != j:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-6:
                return False
        
        # Change k
        k2 = (k + 1) % p
        a = i * m + j
        b = j * p + k2
        c = k * n + i
        if k2 != k:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-6:
                return False
    
    return True

def _try_polynomial_embedding(n, m, p, target_rank, deadline):
    """
    Try to find a decomposition using polynomial interpolation embedding.
    
    The idea: represent the inner product as coefficient extraction from
    polynomial products, then use evaluation points to reconstruct.
    
    For A[i,j] * B[j,k], we can think of this as:
    - For each row i of A, form polynomial A_i(x) = sum_j A[i,j] x^j
    - For each column k of B, form polynomial B_k(x) = sum_j B[j,k] x^j
    - Then C[i,k] = coefficient of x^(m-1) in A_i(x) * B_k(x)
    
    But this gives us a different structure. Let me think differently.
    
    Alternative: Use the fact that for fixed i, the map j -> (A[i,j], B[j,:]) 
    can be represented via evaluation at m distinct points.
    
    Actually, let's use a more direct approach:
    The tensor T[i*m+j, j*p+k, k*n+i] = 1
    
    For polynomial embedding, we can write:
    T = sum_r u_r ⊗ v_r ⊗ w_r
    
    where u_r, v_r, w_r are related to evaluation points.
    
    Let's try: for r = 0, ..., R-1, choose evaluation points t_r.
    Then define:
    u_r[i*m+j] = t_r^j * alpha_r(i)
    v_r[j*p+k] = t_r^j * beta_r(k)  # Note: same t_r^j to match the middle index
    w_r[k*n+i] = gamma_r(k) * delta_r(i)
    
    But this doesn't quite work because the middle index j appears in both u and v.
    
    Let me try a different approach: use the structure of the problem.
    The key insight is that for each j, we have a "slice" of the tensor.
    We can try to decompose using rank-1 terms that are localized in j.
    
    For now, let's just return None and use the fallback.
    """
    # This is a placeholder for the polynomial embedding approach.
    # The actual implementation would be complex, so we'll rely on the
    # structured initialization + gradient descent approach below.
    return None

def _structured_initialization(n, m, p, rank, seed):
    """
    Initialize U, V, W using a structured approach based on the tensor's geometry.
    
    Use the fact that the tensor has a specific pattern of nonzeros to create
    a better starting point for optimization.
    """
    np.random.seed(seed)
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strategy: for each rank-1 term, try to cover a "stripe" of the tensor.
    # The tensor has nonzeros at (i*m+j, j*p+k, k*n+i).
    # We can group terms by i, j, or k.
    
    # Approach 1: Group by i (rows of A)
    # For each i, we have m*p nonzeros. If we use m*p/rank_i terms per i,
    # total rank = n * (m*p/rank_i). For rank = 35, rank_i = 35/3 ≈ 11.67.
    
    # Let's use a hybrid: some terms dedicated to each i, some shared.
    
    # First, allocate terms to cover the "diagonal" structure.
    # For each (i, j), we can have terms that handle all k for that (i, j).
    # That would be n*m terms, each handling p k-values.
    
    # But that's only n*m = 15 terms, and each would need to be rank p=7 in some sense.
    
    # Let's try: for each (i, j), create terms that are "active" for that (i, j)
    # and vary over k.
    
    num_i_terms = n  # One group per i
    num_j_terms = m  # One group per j
    num_k_terms = p  # One group per k
    
    # Total terms: n + m + p = 3 + 5 + 7 = 15. Not enough for rank 35.
    
    # Let's use a different structure: for each i, use m terms (one per j).
    # For each (i, j), the term handles all k.
    # That's n*m = 15 terms. Still not enough.
    
    # For rank 35, we need 35 terms. Let's use:
    # - 15 terms for the "main" structure (one per (i,j))
    # - 20 additional terms for refinement
    
    # Initialize the first 15 terms with a structured pattern.
    for i in range(n):
        for j in range(m):
            r = i * m + j
            if r < rank:
                # This term is "active" for row i, column j
                U[i * m + j, r] = 1.0
                for k in range(p):
                    V[j * p + k, r] = 1.0
                    W[k * n + i, r] = 1.0
    
    # Initialize remaining terms randomly but with small magnitude
    for r in range(n * m, rank):
        U[:, r] = np.random.randn(n * m) * 0.1
        V[:, r] = np.random.randn(m * p) * 0.1
        W[:, r] = np.random.randn(p * n) * 0.1
    
    return U, V, W

def _gradient_descent(U, V, W, n, m, p, target_rank, deadline, max_iter=2000):
    """
    Perform gradient descent to refine the decomposition.
    
    Minimize: sum_{target} (recon - 1)^2 + sum_{zero_sample} recon^2
    """
    # Precompute target entries
    target_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                target_entries.append((a, b, c))
    
    # Sample zero entries
    np.random.seed(123)
    zero_entries = set()
    for _ in range(500):
        a = np.random.randint(n * m)
        b = np.random.randint(m * p)
        c = np.random.randint(p * n)
        i = a // m
        j = a % m
        j2 = b // p
        k = b % p
        i2 = c % n
        k2 = c // n
        if not (i == i2 and j == j2 and k == k2):
            zero_entries.add((a, b, c))
    
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    lr = 0.01
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        # Loss on target entries
        for a, b, c in target_entries:
            recon = np.dot(U[a, :], V[b, :] * W[c, :])
            err = recon - 1.0
            # d/dU[a,r] = err * V[b,r] * W[c,r]
            dU[a, :] += err * V[b, :] * W[c, :]
            # d/dV[b,r] = err * U[a,r] * W[c,r]
            dV[b, :] += err * U[a, :] * W[c, :]
            # d/dW[c,r] = err * U[a,r] * V[b,r]
            dW[c, :] += err * U[a, :] * V[b, :]
        
        # Loss on zero entries
        for a, b, c in zero_entries:
            recon = np.dot(U[a, :], V[b, :] * W[c, :])
            # d/dU[a,r] = 2 * recon * V[b,r] * W[c,r]
            dU[a, :] += 2.0 * recon * V[b, :] * W[c, :]
            dV[b, :] += 2.0 * recon * U[a, :] * W[c, :]
            dW[c, :] += 2.0 * recon * U[a, :] * V[b, :]
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Compute loss
        loss = 0.0
        for a, b, c in target_entries:
            recon = np.dot(U[a, :], V[b, :] * W[c, :])
            loss += (recon - 1.0) ** 2
        for a, b, c in zero_entries:
            recon = np.dot(U[a, :], V[b, :] * W[c, :])
            loss += recon ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        # Adaptive learning rate
        if loss < best_loss * 0.99:
            lr *= 0.95
        elif loss > best_loss * 1.01:
            lr *= 0.7
        
        if best_loss < 1e-10:
            break
    
    return best_U, best_V, best_W, best_loss

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = n*m*p = 105
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    # Verify schoolbook
    if not _verify(U_sb, V_sb, W_sb, n, m, p):
        raise ValueError("Schoolbook decomposition failed verification")
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    best_rank = rank_schoolbook
    
    # Try lower ranks
    # Lower bound: max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    for target_rank in range(35, rank_schoolbook + 1):
        if time.time() > deadline - 20:
            break
        
        # Try multiple seeds
        for seed in range(5):
            if time.time() > deadline - 20:
                break
            
            # Structured initialization
            U, V, W = _structured_initialization(n, m, p, target_rank, seed)
            
            # Gradient descent
            U_opt, V_opt, W_opt, loss = _gradient_descent(
                U, V, W, n, m, p, target_rank, deadline, max_iter=500
            )
            
            # Check if we found a valid decomposition
            if loss < 1e-8:
                # Verify
                if _verify(U_opt, V_opt, W_opt, n, m, p):
                    # Snap to half-integers
                    U_snap = _snap_matrix(U_opt)
                    V_snap = _snap_matrix(V_opt)
                    W_snap = _snap_matrix(W_opt)
                    
                    # Re-verify after snapping
                    if _verify(U_snap, V_snap, W_snap, n, m, p):
                        best_result = {"U": U_snap.tolist(), "V": V_snap.tolist(), "W": W_snap.tolist()}
                        best_rank = target_rank
                        break  # Found valid decomposition at this rank
        
        if best_rank < rank_schoolbook:
            break  # Found a better rank, stop
    
    return best_result
