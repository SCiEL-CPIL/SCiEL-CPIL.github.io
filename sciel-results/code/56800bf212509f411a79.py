"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def _build_rank33():
    """Construct the known rank-33 decomposition using 3 blocks of 2x3x2."""
    n, m, p = 2, 3, 6
    rank = 33
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    terms = []
    terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
    terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
    terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
    terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
    terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
    terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
    terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
    terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
    terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
    terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})
    
    for block in range(3):
        for t_idx, term in enumerate(terms):
            col = block * 11 + t_idx
            for idx, val in term['u'].items():
                U[idx, col] = val
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx, col] = val
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return U, V, W


def _verify_decomp(U, V, W, n, m, p, tol=1e-9):
    """Check if decomposition is valid."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _alternating_minimize(U, V, W, n, m, p, max_iters=200, tol=1e-10, deadline=None):
    """
    Alternating minimization: fix two matrices, solve for the third via linear least squares.
    For U: for each (i,j), solve min ||A_{ij} U[i*m+j,:]^T - 1|| where A_{ij}[k,:] = V[j*p+k,:] * W[k*n+i,:]
    For V: for each (j,k), solve min ||B_{jk} V[j*p+k,:]^T - 1|| where B_{jk}[i,:] = U[i*m+j,:] * W[k*n+i,:]
    For W: for each (k,i), solve min ||C_{ki} W[k*n+i,:]^T - 1|| where C_{ki}[j,:] = U[i*m+j,:] * V[j*p+k,:]
    """
    rank = U.shape[1]
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Update U: for each (i,j), solve A_{ij} @ u = 1
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A @ u = ones(p)
                u, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                U[i*m+j, :] = u
        
        # Update V: for each (j,k), solve B_{jk} @ v = 1
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    B[i, :] = U[i*m+j, :] * W[k*n+i, :]
                v, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                V[j*p+k, :] = v
        
        # Update W: for each (k,i), solve C_{ki} @ w = 1
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, rank))
                for j in range(m):
                    C[j, :] = U[i*m+j, :] * V[j*p+k, :]
                w, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                W[k*n+i, :] = w
        
        # Check convergence
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        if max_err < tol:
            break
    
    return U, V, W, max_err


def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Start with the known rank-33 decomposition
    U33, V33, W33 = _build_rank33()
    best_U, best_V, best_W = U33.copy(), V33.copy(), W33.copy()
    best_rank = 33
    best_err = 0.0
    
    if not _verify_decomp(U33, V33, W33, n, m, p):
        raise ValueError("Initial rank-33 decomposition failed verification")
    
    # Try to find lower-rank decompositions using alternating minimization
    # Try ranks from 32 down to 20
    for target_rank in range(32, 19, -1):
        if time.time() > deadline:
            break
        
        # Initialize with random values, scaled to help convergence
        np.random.seed(42 + target_rank)
        U = np.random.randn(n * m, target_rank) * 0.5
        V = np.random.randn(m * p, target_rank) * 0.5
        W = np.random.randn(p * n, target_rank) * 0.5
        
        # Run alternating minimization
        U_new, V_new, W_new, err = _alternating_minimize(
            U, V, W, n, m, p, max_iters=300, tol=1e-12, deadline=deadline
        )
        
        if err < 1e-9:
            # Verify with exact check
            if _verify_decomp(U_new, V_new, W_new, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                break  # Found a valid lower-rank decomposition
        
        # Also try initializing from the rank-33 decomposition by truncating
        if time.time() > deadline:
            break
        U_init = U33[:, :target_rank].copy()
        V_init = V33[:, :target_rank].copy()
        W_init = W33[:, :target_rank].copy()
        
        U_new, V_new, W_new, err = _alternating_minimize(
            U_init, V_init, W_init, n, m, p, max_iters=300, tol=1e-12, deadline=deadline
        )
        
        if err < 1e-9:
            if _verify_decomp(U_new, V_new, W_new, n, m, p, tol=1e-9):
                best_U, best_V, best_W = U_new, V_new, W_new
                best_rank = target_rank
                break
    
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}


def _verify(result, n, m, p):
    """Verify the decomposition is correct by checking all (i,j,k) entries."""
    U = result['U']
    V = result['V']
    W = result['W']
    rank = len(U[0])
    
    assert len(U) == n * m, f"U shape {len(U)} != {n*m}"
    assert len(V) == m * p, f"V shape {len(V)} != {m*p}"
    assert len(W) == p * n, f"W shape {len(W)} != {p*n}"
    assert all(len(row) == rank for row in U), "U column count mismatch"
    assert all(len(row) == rank for row in V), "V column count mismatch"
    assert all(len(row) == rank for row in W), "W column count mismatch"
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    raise ValueError(f"Verification failed at i={i},j={k}: got {s}")
    
    return True
# EVOLVE-BLOCK-END
