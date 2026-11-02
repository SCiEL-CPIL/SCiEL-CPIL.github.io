"""Rewritten program for 4x4 @ 4x5 matrix multiplication tensor decomposition."""
import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication.
    
    Each term is a dict with 'U', 'V', 'W' keys, each mapping (row, col) -> coefficient.
    U is indexed by (i, j) for A[i,j]
    V is indexed by (j, k) for B[j,k]
    W is indexed by (k, i) for C[k,i] -- but we use W[k*n+i] in the flat index
    
    For C = A*B:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    For 2x2x2 (n=m=p=2):
    M1 = (A11+A22)(B11+B22)
    M2 = (A21+A22)B11
    M3 = A11(B21-B22)
    M4 = A22(B12-B11)
    M5 = (A11+A12)B22
    M6 = (A21-A11)(B11+B12)
    M7 = (A12-A22)(B21+B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    """
    terms = []
    # M1 = (A11+A22)(B11+B22)
    terms.append({'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}})
    # M2 = (A21+A22)B11
    terms.append({'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1, (1,1): -1}})
    # M3 = A11(B21-B22)
    terms.append({'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): 1}})
    # M4 = A22(B12-B11)
    terms.append({'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): 1, (1,0): 1}})
    # M5 = (A11+A12)B22
    terms.append({'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,0): -1, (0,1): 1}})
    # M6 = (A21-A11)(B11+B12)
    terms.append({'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(1,1): 1}})
    # M7 = (A12-A22)(B21+B22)
    terms.append({'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1}})
    return terms

def _build_4x4x4_strassen():
    """Build a 4x4x4 decomposition using recursive Strassen (rank 49).
    
    Decompose 4x4x4 into 2x2x2 blocks.
    Outer: 7 Strassen terms for the 2x2 block structure
    Inner: 7 Strassen terms for each 2x2x2 sub-problem
    Total: 49 terms
    """
    n = 4
    terms_2x2 = _strassen_2x2()
    
    U = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    V = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    W = [[0] * 49 for _ in range(16)]  # 4*4 = 16
    
    for t1, term1 in enumerate(terms_2x2):
        for t2, term2 in enumerate(terms_2x2):
            idx = t1 * 7 + t2
            
            # U: indexed by (i, j) where i,j in 0..3
            # i = bi*2 + li, j = bj*2 + lj
            for (bi, bj), coeff_u in term1['U'].items():
                for (li, lj), coeff_v in term2['U'].items():
                    i = bi * 2 + li
                    j = bj * 2 + lj
                    U[i*4 + j][idx] += coeff_u * coeff_v
            
            # V: indexed by (j, k) where j,k in 0..3
            # j = bj*2 + lj, k = bk*2 + lk
            for (bj, bk), coeff_v in term1['V'].items():
                for (lj, lk), coeff_w in term2['V'].items():
                    j = bj * 2 + lj
                    k = bk * 2 + lk
                    V[j*4 + k][idx] += coeff_v * coeff_w
            
            # W: indexed by (k, i) where k,i in 0..3
            # k = bk*2 + lk, i = bi*2 + li
            for (bk, bi), coeff_w in term1['W'].items():
                for (lk, li), coeff_u2 in term2['W'].items():
                    k = bk * 2 + lk
                    i = bi * 2 + li
                    W[k*4 + i][idx] += coeff_w * coeff_u2
    
    return {"U": U, "V": V, "W": W}

def _build_4x4x5_strassen():
    """Build a 4x4x5 decomposition: 4x4x4 (rank 49) + 4x4x1 (rank 4) = rank 53.
    
    The 4x4x4 part handles C[0:4, 0:4] = A @ B[:, 0:4].
    The 4x4x1 part handles C[0:4, 4] = A @ B[:, 4].
    """
    n, m, p = DIMENSIONS
    
    base = _build_4x4x4_strassen()
    rank_base = 49
    
    rank = rank_base + 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Copy 4x4x4 part (handles k=0..3)
    for i in range(n * m):
        for r in range(rank_base):
            U[i][r] = base["U"][i][r]
    for j in range(m * 4):  # j*p+k for k=0..3, so j*5+0..3 but we use j*4+k in base
        # V in base is indexed by (j, k) with k in 0..3, flat index j*4+k
        # In the full problem, V is indexed by (j, k) with k in 0..4, flat index j*5+k
        # So base V[j*4+k] maps to full V[j*5+k] for k=0..3
        for k in range(4):
            V[j*5 + k][r] = base["V"][j*4 + k][r]
    for k in range(4 * n):  # k*n+i for k=0..3
        # W in base is indexed by (k, i) with k in 0..3, flat index k*4+i
        # In full problem, W is indexed by (k, i) with k in 0..4, flat index k*5+i
        for i in range(n):
            W[k*5 + i][r] = base["W"][k*4 + i][r]
    
    # Add 4x4x1 part: C[i,4] = sum_j A[i,j] * B[j,4] for i=0..3
    # This requires m=4 terms (one per j), using schoolbook for this slice
    for j in range(m):
        r = rank_base + j
        for i in range(n):
            U[i*m + j][r] = 1
        V[j*p + 4][r] = 1  # B[j,4]
        for i in range(n):
            W[4*n + i][r] = 1  # C[4,i] where k=4, so index = 4*4+i = 16+i
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U_np[i*m+j, :]
                v_val = V_np[j*p+k, :]
                w_val = W_np[k*n+i, :]
                total = np.dot(u_val * v_val, w_val)
                if abs(total - 1.0) > tol:
                    return False
    return True

def _verify_integer_half_integer(U, V, W, tol=1e-9):
    """Check that all entries are integer or half-integer (multiple of 0.5)."""
    for name, M in [("U", U), ("V", V), ("W", W)]:
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True

def _try_merge_terms(U, V, W, n, m, p):
    """Try to merge or eliminate terms to reduce rank.
    
    This function attempts to find linear dependencies among the rank-1 terms
    and eliminate them, potentially reducing the rank.
    """
    rank = len(U[0])
    
    # Convert to numpy for linear algebra
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # For each pair of terms, check if one can be expressed as a combination of others
    # This is a simplified approach - in practice, we'd use more sophisticated methods
    
    # Try to find if any column is a linear combination of others
    # For each factor, check column rank
    for name, M_np in [("U", U_np), ("V", V_np), ("W", W_np)]:
        # Check if any column is zero
        for r in range(rank):
            if np.allclose(M_np[:, r], 0, atol=1e-12):
                # This term is zero, can be removed
                return True
    
    return False

def _build_4x4x5_optimized():
    """Build a 4x4x5 decomposition with optimization attempts.
    
    Start with the Strassen-based rank 53 and try to reduce it.
    """
    n, m, p = DIMENSIONS
    
    # Start with Strassen-based
    result = _build_4x4x5_strassen()
    
    # Try to find and remove zero or redundant terms
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = len(U[0])
    
    # Check for zero terms
    new_U = []
    new_V = []
    new_W = []
    kept_indices = []
    
    for r in range(rank):
        u_col = [U[i][r] for i in range(len(U))]
        v_col = [V[i][r] for i in range(len(V))]
        w_col = [W[i][r] for i in range(len(W))]
        
        # Skip if any factor is entirely zero
        if any(abs(x) > 1e-12 for x in u_col) and \
           any(abs(x) > 1e-12 for x in v_col) and \
           any(abs(x) > 1e-12 for x in w_col):
            kept_indices.append(r)
    
    if len(kept_indices) < rank:
        U = [row[r] for r in kept_indices]
        V = [row[r] for r in kept_indices]
        W = [row[r] for r in kept_indices]
        result = {"U": U, "V": V, "W": W}
    
    return result

def _slice_wise_linear_solve(target_rank, n, m, p, seed=42, time_limit=60):
    """Use slice-wise linear system approach to find lower-rank decomposition.
    
    Strategy: Fix W to be a random matrix, then for each output slice k,
    solve a linear system for the corresponding U and V columns.
    This exploits the structure that each output slice C[:,k] depends
    on specific combinations of A and B entries.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize W with random values
    W = np.random.randn(p * n, target_rank) * 0.5
    
    # For each target rank, try to find U and V such that:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # We'll use an iterative approach: fix W, solve for U,V alternately
    U = np.random.randn(n*m, target_rank) * 0.1
    V = np.random.randn(m*p, target_rank) * 0.1
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for iteration in range(200):
        if time.time() - start_time > time_limit:
            break
        
        # Compute current loss
        loss = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    s = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                    loss += (s - 1.0) ** 2
        
        if loss < best_loss:
            best_loss = loss
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        if loss < 1e-10:
            break
        
        # Update U: for each column r, solve least squares
        for r in range(target_rank):
            # Fix all other columns, solve for U[:,r]
            U_old = U[:, r].copy()
            
            # Build the linear system: for each (i,j,k),
            # U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] should contribute to make sum = 1
            # This is complex; use gradient step instead
            
            # Gradient-based update for U[:,r]
            grad_U = np.zeros(n*m)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        s = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                        resid = s - 1.0
                        grad_U[u_idx] += 2.0 * resid * V[v_idx, r] * W[w_idx, r]
            
            # Gradient descent step
            lr = 0.01
            U[:, r] -= lr * grad_U
        
        # Update V similarly
        for r in range(target_rank):
            grad_V = np.zeros(m*p)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        s = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                        resid = s - 1.0
                        grad_V[v_idx] += 2.0 * resid * U[u_idx, r] * W[w_idx, r]
            
            lr = 0.01
            V[:, r] -= lr * grad_V
        
        # Update W similarly
        for r in range(target_rank):
            grad_W = np.zeros(p*n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i*m + j
                        v_idx = j*p + k
                        w_idx = k*n + i
                        s = np.dot(U[u_idx, :] * V[v_idx, :], W[w_idx, :])
                        resid = s - 1.0
                        grad_W[w_idx] += 2.0 * resid * U[u_idx, r] * V[v_idx, r]
            
            lr = 0.01
            W[:, r] -= lr * grad_W
    
    if best_loss < 1e-6:
        # Try to snap to half-integers
        U_snap = np.round(best_U * 2.0) / 2.0
        V_snap = np.round(best_V * 2.0) / 2.0
        W_snap = np.round(best_W * 2.0) / 2.0
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # Try without snapping
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _try_gradient_optimization(target_rank, n, m, p, max_iter=500, seed=42, time_limit=30):
    """Try to find a decomposition of given rank using gradient-based optimization.
    
    Uses L-BFGS-B with vectorized loss computation and rational reconstruction.
    """
    start_time = time.time()
    np.random.seed(seed)
    
    # Initialize with small random values
    U = np.random.randn(n*m, target_rank) * 0.3
    V = np.random.randn(m*p, target_rank) * 0.3
    W = np.random.randn(p*n, target_rank) * 0.3
    
    def loss_and_grad(params):
        U_p = params[:n*m*target_rank].reshape(n*m, target_rank)
        V_p = params[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_p = params[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Compute residual: for each (i,j,k), residual = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1
        # Vectorized computation
        loss_val = 0.0
        U_grad = np.zeros_like(U_p)
        V_grad = np.zeros_like(V_p)
        W_grad = np.zeros_like(W_p)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i*m + j
                    v_idx = j*p + k
                    w_idx = k*n + i
                    
                    # residual = sum_r U[u_idx,r]*V[v_idx,r]*W[w_idx,r] - 1
                    prod = U_p[u_idx, :] * V_p[v_idx, :]
                    s = np.dot(prod, W_p[w_idx, :])
                    resid = s - 1.0
                    
                    loss_val += resid * resid
                    
                    # Gradients
                    # d(loss)/d(U[u_idx,r]) = 2*resid * V[v_idx,r] * W[w_idx,r]
                    U_grad[u_idx, :] += 2.0 * resid * V_p[v_idx, :] * W_p[w_idx, :]
                    V_grad[v_idx, :] += 2.0 * resid * U_p[u_idx, :] * W_p[w_idx, :]
                    W_grad[w_idx, :] += 2.0 * resid * U_p[u_idx, :] * V_p[v_idx, :]
        
        grad = np.concatenate([U_grad.flatten(), V_grad.flatten(), W_grad.flatten()])
        return loss_val, grad
    
    params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    from scipy.optimize import minimize
    try:
        result = minimize(loss_and_grad, params, jac=True, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-12, 'maxls': 50})
    except Exception:
        return None
    
    if time.time() - start_time > time_limit:
        return None
    
    if result.fun < 1e-6:
        U_opt = result.x[:n*m*target_rank].reshape(n*m, target_rank)
        V_opt = result.x[n*m*target_rank:n*m*target_rank + m*p*target_rank].reshape(m*p, target_rank)
        W_opt = result.x[n*m*target_rank + m*p*target_rank:].reshape(p*n, target_rank)
        
        # Rational reconstruction: snap to nearest half-integer
        U_snap = np.round(U_opt * 2.0) / 2.0
        V_snap = np.round(V_opt * 2.0) / 2.0
        W_snap = np.round(W_opt * 2.0) / 2.0
        
        U_list = U_snap.tolist()
        V_list = V_snap.tolist()
        W_list = W_snap.tolist()
        
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
        
        # If snapping failed, try without snapping
        U_list = U_opt.tolist()
        V_list = V_opt.tolist()
        W_list = W_opt.tolist()
        if _verify_decomposition(U_list, V_list, W_list, n, m, p):
            return {"U": U_list, "V": V_list, "W": W_list}
    
    return None

def _build_schoolbook():
    """Fallback schoolbook decomposition with rank n*m*p = 80."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    best_result = None
    best_rank = float('inf')
    
    # Approach 1: Strassen-based (rank 53)
    try:
        result = _build_4x4x5_strassen()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            candidates = [(rank, result)]
        else:
            candidates = []
    except Exception:
        candidates = []
    
    # Approach 2: Optimized Strassen
    try:
        result = _build_4x4x5_optimized()
        rank = len(result["U"][0])
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
            candidates.append((rank, result))
    except Exception:
        pass
    
    # Pick the best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        best_rank, best_result = candidates[0]
    
    # Try to find even lower rank through optimization if time permits
    if best_result is not None and time.time() < deadline - 45:
        # Try several random seeds for each target rank
        for target_rank in range(best_rank - 1, max(best_rank - 6, 35), -1):
            if time.time() > deadline - 25:
                break
            found = False
            for seed in [target_rank, target_rank + 100, target_rank + 200]:
                if time.time() > deadline - 20:
                    break
                try:
                    # Try slice-wise approach first (faster)
                    result = _slice_wise_linear_solve(target_rank, n, m, p, seed=seed, time_limit=10)
                    if result is None:
                        # Fall back to gradient optimization
                        result = _try_gradient_optimization(target_rank, n, m, p, max_iter=300, seed=seed, time_limit=10)
                    if result is not None:
                        best_result = result
                        best_rank = target_rank
                        found = True
                        break
                except Exception:
                    pass
            if found:
                break
    
    if best_result is not None:
        # Ensure all values are proper Python floats/ints
        U = [[float(x) for x in row] for row in best_result["U"]]
        V = [[float(x) for x in row] for row in best_result["V"]]
        W = [[float(x) for x in row] for row in best_result["W"]]
        return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return _build_schoolbook()

# EVOLVE-BLOCK-END
