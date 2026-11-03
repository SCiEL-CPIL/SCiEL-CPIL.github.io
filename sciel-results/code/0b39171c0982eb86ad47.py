"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # P1 = (A[0,j0]+A[1,j1]) * (B[j0,k0]+B[j1,k1])
    U[0*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
    U[1*m + j0][col] += 1
    U[1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + 1][col] += 1
    W[k1*n + 1][col] -= 1
    col += 1
    
    # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
    U[0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + 0][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
    U[1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + 0][col] += 1
    W[k0*n + 1][col] += 1
    col += 1
    
    # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
    U[0*m + j0][col] += 1
    U[0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] -= 1
    W[k1*n + 0][col] += 1
    col += 1
    
    # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
    U[1*m + j0][col] += 1
    U[0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + 1][col] += 1
    col += 1
    
    # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
    U[0*m + j1][col] += 1
    U[1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + 0][col] += 1
    col += 1
    
    return col


def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    if U_arr.shape != (n*m, rank) or V_arr.shape != (m*p, rank) or W_arr.shape != (p*n, rank):
        return False
    
    # Check half-integer constraint
    for arr in [U_arr, V_arr, W_arr]:
        doubled = arr * 2
        if not np.allclose(doubled, np.round(doubled), atol=1e-10):
            return False
    
    # Verify tensor identity
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for r in range(rank):
                    total += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(total - 1.0) > 1e-9:
                    return False
    return True


def _build_strassen_54(n, m, p):
    """Build the rank-54 Strassen block decomposition."""
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p)
    
    # Remaining j=4: schoolbook for 2x1x6 block
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    return U, V, W, rank


def _try_lower_rank(n, m, p, target_rank, deadline, seed=42):
    """
    Attempt to find a rank-target_rank decomposition via randomized algebraic search.
    Uses the structure: try to express the tensor as sum of target_rank rank-1 terms
    with entries in {0, ±1, ±1/2}.
    """
    np.random.seed(seed)
    
    nU = n * m
    nV = m * p
    nW = p * n
    
    # Number of tensor entries
    n_tensor = n * m * p  # 60
    
    # Strategy: random initialization with structured sparsity, then local refinement
    # We work in the space of half-integers by scaling by 2
    
    max_attempts = 3
    for attempt in range(max_attempts):
        if time.time() > deadline - 10:
            return None
        
        # Initialize with random sparse half-integer values
        # Each entry is 0 with prob 0.6, ±1/2 with prob 0.2, ±1 with prob 0.1
        def rand_sparse(nrows, rank):
            mat = np.zeros((nrows, rank))
            for i in range(nrows):
                for r in range(rank):
                    u = np.random.random()
                    if u < 0.55:
                        mat[i, r] = 0
                    elif u < 0.75:
                        mat[i, r] = 0.5 * (1 if np.random.random() < 0.5 else -1)
                    elif u < 0.90:
                        mat[i, r] = 1.0 * (1 if np.random.random() < 0.5 else -1)
                    else:
                        mat[i, r] = 1.5 * (1 if np.random.random() < 0.5 else -1)
            return mat
        
        U = rand_sparse(nU, target_rank)
        V = rand_sparse(nV, target_rank)
        W = rand_sparse(nW, target_rank)
        
        # Local search: flip individual entries to reduce residual
        # Residual: for each (i,j,k), R[i,j,k] = 1 - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        best_res = _max_residual(n, m, p, U, V, W, target_rank)
        
        if best_res < 1e-9:
            return (U.tolist(), V.tolist(), W.tolist())
        
        # Hill climbing: try flipping entries to reduce max residual
        lr_schedule = [0.05, 0.02, 0.01]
        for phase in range(3):
            if time.time() > deadline - 10:
                break
            max_iters = 2000
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                
                # Compute residual tensor
                R = np.ones((n, m, p))
                for r in range(target_rank):
                    Ur = U[:, r].reshape(n, m)
                    Vr = V[:, r].reshape(m, p)
                    Wr = W[:, r].reshape(p, n)
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                R[i, j, k] -= Ur[i, j] * Vr[j, k] * Wr[k, i]
                
                max_res = np.max(np.abs(R))
                if max_res < 1e-9:
                    return (U.tolist(), V.tolist(), W.tolist())
                
                # Find worst entry and try to fix it
                idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
                ii, jj, kk = idx
                res_ijk = R[ii, jj, kk]
                
                # Try flipping a random U entry
                if np.random.random() < 0.33:
                    ri = np.random.randint(nU)
                    rr = np.random.randint(target_rank)
                    old = U[ri, rr]
                    # Try a few new values
                    for new_val in [0, 0.5, -0.5, 1, -1, 1.5, -1.5]:
                        if new_val == old:
                            continue
                        U[ri, rr] = new_val
                        # Quick check: recompute residual at (ii,jj,kk)
                        # Only if this entry affects it
                        if ri == ii*m + jj:
                            delta = (new_val - old) * V[jj*p+kk, rr] * W[kk*n+ii, rr]
                            new_res = res_ijk - delta
                            if abs(new_res) < abs(res_ijk):
                                break
                            else:
                                U[ri, rr] = old
                                new_val = old
                        else:
                            U[ri, rr] = old
                            new_val = old
                
                elif np.random.random() < 0.66:
                    ri = np.random.randint(nV)
                    rr = np.random.randint(target_rank)
                    old = V[ri, rr]
                    for new_val in [0, 0.5, -0.5, 1, -1, 1.5, -1.5]:
                        if new_val == old:
                            continue
                        V[ri, rr] = new_val
                        if ri == jj*p + kk:
                            delta = (new_val - old) * U[ii*m+jj, rr] * W[kk*n+ii, rr]
                            new_res = res_ijk - delta
                            if abs(new_res) < abs(res_ijk):
                                break
                            else:
                                V[ri, rr] = old
                                new_val = old
                        else:
                            V[ri, rr] = old
                            new_val = old
                
                else:
                    ri = np.random.randint(nW)
                    rr = np.random.randint(target_rank)
                    old = W[ri, rr]
                    for new_val in [0, 0.5, -0.5, 1, -1, 1.5, -1.5]:
                        if new_val == old:
                            continue
                        W[ri, rr] = new_val
                        if ri == kk*n + ii:
                            delta = (new_val - old) * U[ii*m+jj, rr] * V[jj*p+kk, rr]
                            new_res = res_ijk - delta
                            if abs(new_res) < abs(res_ijk):
                                break
                            else:
                                W[ri, rr] = old
                                new_val = old
                        else:
                            W[ri, rr] = old
                            new_val = old
        
        # Final check
        final_res = _max_residual(n, m, p, U, V, W, target_rank)
        if final_res < 1e-9:
            return (U.tolist(), V.tolist(), W.tolist())
    
    return None


def _max_residual(n, m, p, U, V, W, rank):
    """Compute the max absolute residual of the tensor decomposition."""
    R = np.ones((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    R[i, j, k] -= Ur[i, j] * Vr[j, k] * Wr[k, i]
    return np.max(np.abs(R))


def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin for 150s limit
    
    # Build the known rank-54 Strassen decomposition
    U, V, W, rank = _build_strassen_54(n, m, p)
    
    # Verify it's correct
    if not _verify_decomp(U, V, W, n, m, p, rank):
        # Should not happen, but fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
    
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    best_rank = rank
    
    # Try to find a lower-rank decomposition
    # Theoretical lower bound: for 2x5x6, rank >= max(2*5, 5*6, 6*2) = 30
    # But Strassen gives 54. Try ranks below 54.
    # Be conservative: try 53, 52, 51, 50 in that order
    target_ranks = [53, 52, 51, 50]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 15:
            break
        
        result = _try_lower_rank(n, m, p, target_rank, deadline)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_decomp(U_new, V_new, W_new, n, m, p, target_rank):
                best_rank = target_rank
                best_U, best_V, best_W = U_new, V_new, W_new
                break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
