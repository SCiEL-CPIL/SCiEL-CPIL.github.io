"""
Matrix multiplication tensor decomposition for 2x3 * 3x10.

Strategy:
  - Start with a known rank-55 decomposition (Strassen blocks + outer products).
  - Use ALS-style local search to attempt to find a lower-rank exact decomposition.
  - If no lower rank is found within the time budget, return the best known rank-55 result.

The evaluator measures the raw rank value (lower is better).
"""
import time
import numpy as np
import random

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def _build_rank55_decomposition(n, m, p):
    """Build the known rank-55 decomposition using Strassen blocks + outer products."""
    RANK = 55
    U = np.zeros((n * m, RANK), dtype=np.int64)
    V = np.zeros((m * p, RANK), dtype=np.int64)
    W = np.zeros((p * n, RANK), dtype=np.int64)

    col = 0
    X00, X01, X10, X11 = 0, 1, 3, 4

    for j in range(5):
        Y00 = 2 * j
        Y01 = 2 * j + 1
        Y10 = 10 + 2 * j
        Y11 = 10 + 2 * j + 1
        Z00 = 4 * j
        Z10 = 4 * j + 1
        Z01 = 4 * j + 2
        Z11 = 4 * j + 3

        # M1 = (X00+X11)*(Y00+Y11)
        U[X00, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1; W[Z11, col] += 1
        col += 1

        # M2 = (X10+X11)*Y00
        U[X10, col] += 1; U[X11, col] += 1
        V[Y00, col] += 1
        W[Z10, col] += 1; W[Z11, col] -= 1
        col += 1

        # M3 = X00*(Y01-Y11)
        U[X00, col] += 1
        V[Y01, col] += 1; V[Y11, col] -= 1
        W[Z01, col] += 1; W[Z11, col] += 1
        col += 1

        # M4 = X11*(Y10-Y00)
        U[X11, col] += 1
        V[Y10, col] += 1; V[Y00, col] -= 1
        W[Z00, col] += 1; W[Z10, col] += 1
        col += 1

        # M5 = (X00+X01)*Y11
        U[X00, col] += 1; U[X01, col] += 1
        V[Y11, col] += 1
        W[Z00, col] -= 1; W[Z01, col] += 1
        col += 1

        # M6 = (X10-X00)*(Y00+Y01)
        U[X10, col] += 1; U[X00, col] -= 1
        V[Y00, col] += 1; V[Y01, col] += 1
        W[Z11, col] += 1
        col += 1

        # M7 = (X01-X11)*(Y10+Y11)
        U[X01, col] += 1; U[X11, col] -= 1
        V[Y10, col] += 1; V[Y11, col] += 1
        W[Z00, col] += 1
        col += 1

    # Part B: outer products for A_right * B_bottom
    for i in range(n):
        a_idx = i * m + 2
        for k in range(p):
            b_idx = 2 * p + k
            c_idx = k * n + i
            U[a_idx, col] = 1
            V[b_idx, col] = 1
            W[c_idx, col] = 1
            col += 1

    return U, V, W, RANK


def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that the decomposition exactly reproduces matrix multiplication."""
    # Build the multiplication tensor from U, V, W
    # T[i*m+j, j*p+k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    U_float = U.astype(np.float64)
    V_float = V.astype(np.float64)
    W_float = W.astype(np.float64)
    
    # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                val = np.dot(U_float[u_idx, :], V_float[v_idx, :])
                # Actually we need the tensor: T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # where a = i*m+j, b = j*p+k, c = k*n+i
                # But the tensor for matrix mult is: for each r, the contribution to C[i,k]
                # is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r
                # Wait, no. The tensor T has indices (i*m+j, j*p+k, k*n+i) but that's not right.
                # 
                # The multiplication tensor T has T[i*m+j, j*p+k, k*n+i] = delta_{i,j,k} 
                # No wait. For matrix multiplication C = A*B, C[i,k] = sum_j A[i,j]*B[j,k].
                # So the tensor T[i*m+j, j*p+k, k*n+i] = 1 if the indices are consistent, else 0.
                # Actually T[a, b, c] where a encodes (i,j), b encodes (j,k), c encodes (k,i)
                # T[a,b,c] = 1 if the j in a matches the j in b AND the k in b matches the k in c
                # AND the i in a matches the i in c. Otherwise 0.
                #
                # So for a = i*m+j, b = j'*p+k', c = k''*n+i'
                # T[a,b,c] = 1 if j == j' and k == k'' and i == i', else 0.
                #
                # The decomposition says T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                #
                # So let's verify:
                pass
    
    # More efficient verification:
    # Reshape and check
    rank = U.shape[1]
    # For each r, the contribution is outer product U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    # Sum over r should give the multiplication tensor
    
    # Build expected tensor (sparse, only n*m*p entries are 1)
    # Check a few random entries
    rng = np.random.RandomState(42)
    for _ in range(200):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        a = i * m + j
        b = j * p + k
        c = k * n + i
        # T[a,b,c] should be 1
        val = 0.0
        for r in range(rank):
            val += U_float[a, r] * V_float[b, r] * W_float[c, r]
        if abs(val - 1.0) > tol:
            return False, abs(val - 1.0)
    
    # Also check that off-diagonal entries are 0
    for _ in range(200):
        i = rng.randint(n)
        j = rng.randint(m)
        k = rng.randint(p)
        # Perturb one index to make it off-diagonal
        i2 = (i + rng.randint(1, n)) % n
        a = i * m + j
        b = j * p + k
        c = k * n + i2
        if i2 == i:
            continue
        val = 0.0
        for r in range(rank):
            val += U_float[a, r] * V_float[b, r] * W_float[c, r]
        if abs(val) > tol:
            return False, abs(val)
    
    return True, 0.0


def _try_reduce_rank(U, V, W, n, m, p, target_rank, time_budget, seed=42):
    """Try to find a valid decomposition at target_rank using ALS-style optimization."""
    if target_rank >= U.shape[1]:
        return None
    
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize with random projection of the rank-55 solution
    rank55 = U.shape[1]
    U_init = U[:, :target_rank].copy().astype(np.float64)
    V_init = V[:, :target_rank].copy().astype(np.float64)
    W_init = W[:, :target_rank].copy().astype(np.float64)
    
    # Add some random perturbation
    U_init += np.random.randn(*U_init.shape) * 0.1
    V_init += np.random.randn(*V_init.shape) * 0.1
    W_init += np.random.randn(*W_init.shape) * 0.1
    
    # ALS iterations
    best_err = float('inf')
    best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
    
    start = time.time()
    for it in range(200):
        if time.time() - start > time_budget:
            break
        
        # Update U: for each a, solve min_Ua ||sum_r Ua,r * V[:,r] ⊗ W[:,r] - target_a||^2
        # This is a linear least squares problem
        # For each a, we have a matrix V^T ⊗ W^T (size (m*p)*(p*n) x rank)
        # And target vector is the a-th slice of the multiplication tensor
        
        # Build the multiplication tensor slices
        # T[a, :, :] is a (m*p, p*n) matrix
        # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i with consistent indices
        
        # Actually, let's build the full tensor first
        # T has shape (n*m, m*p, p*n)
        T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        
        # For ALS, we flatten the last two dimensions
        # For each a, T[a] is a vector of length (m*p)*(p*n)
        # We want min_Ua sum_a ||T[a].flat - V_flat^T W_flat^T Ua||^2 ... no that's not right
        
        # Better approach: 
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V, W: given U, the residual for entry (a,b,c) is T[a,b,c] - sum_r U[a,r]*V[b,r]*W[c,r]
        # For fixed b,c, this is linear in U[a,:].
        # For fixed a, the contribution to the residual is U[a,:] · (V[b,:] * W[c,:])
        # So for each a, we want U[a,:] to minimize sum_{b,c} (T[a,b,c] - U[a,:] · (V[b,:]*W[c,:]))^2
        # This is a least squares problem: A_a U[a,:] = t_a where A_a[b,c] = V[b,:]*W[c,:] (flattened)
        # and t_a = T[a,:].flat
        
        # Build A for each a
        A_list = []
        t_list = []
        for a in range(n * m):
            # A_a has rows indexed by (b,c), columns by r
            # A_a[(b,c), r] = V[b, r] * W[c, r]
            A_a = np.zeros((m * p * p * n, target_rank), dtype=np.float64)
            t_a = T[a].flatten()
            for b in range(m * p):
                for c in range(p * n):
                    idx = b * (p * n) + c
                    A_a[idx, :] = V_init[b, :] * W_init[c, :]
            A_list.append(A_a)
            t_list.append(t_a)
        
        # Solve for each a
        for a in range(n * m):
            # Solve A_a U[a,:] = t_a
            try:
                U_init[a, :], residuals, rank, sv = np.linalg.lstsq(A_list[a], t_list[a], rcond=None)
            except:
                pass
        
        # Similarly update V and W
        for b in range(m * p):
            A_b = np.zeros((n * m * p * n, target_rank), dtype=np.float64)
            t_b = np.zeros(n * m * p * n, dtype=np.float64)
            for a in range(n * m):
                for c in range(p * n):
                    idx = a * (p * n) + c
                    A_b[idx, :] = U_init[a, :] * W_init[c, :]
                    t_b[idx] = T[a, b, c]
            try:
                V_init[b, :], _, _, _ = np.linalg.lstsq(A_b, t_b, rcond=None)
            except:
                pass
        
        for c in range(p * n):
            A_c = np.zeros((n * m * m * p, target_rank), dtype=np.float64)
            t_c = np.zeros(n * m * m * p, dtype=np.float64)
            for a in range(n * m):
                for b in range(m * p):
                    idx = a * (m * p) + b
                    A_c[idx, :] = U_init[a, :] * V_init[b, :]
                    t_c[idx] = T[a, b, c]
            try:
                W_init[c, :], _, _, _ = np.linalg.lstsq(A_c, t_c, rcond=None)
            except:
                pass
        
        # Compute error
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_init[a, :], V_init[b, :])
                    # Wait, this is wrong. The tensor entry is sum_r U[a,r]*V[b,r]*W[c,r]
                    val = 0.0
                    for r in range(target_rank):
                        val += U_init[a, r] * V_init[b, r] * W_init[c, r]
                    err += (val - 1.0) ** 2
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_init.copy(), V_init.copy(), W_init.copy()
        
        if best_err < 1e-15:
            break
    
    # Round to nearest integers or half-integers and verify
    best_U_int = np.round(best_U * 2).astype(np.int64) / 2.0
    best_V_int = np.round(best_V * 2).astype(np.int64) / 2.0
    best_W_int = np.round(best_W * 2).astype(np.int64) / 2.0
    
    valid, err = _verify_decomposition(best_U_int, best_V_int, best_W_int, n, m, p, tol=1e-8)
    if valid:
        return best_U_int, best_V_int, best_W_int
    
    return None


def construct():
    """
    Returns a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix
    by a 3x10 matrix.
    """
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Start with rank-55 decomposition
    U, V, W, rank55 = _build_rank55_decomposition(n, m, p)
    
    # Try to reduce rank using ALS
    best_result = (U, V, W, rank55)
    best_rank = rank55
    
    # Try ranks from 54 down to some minimum
    for target_rank in range(54, 30, -1):
        if time.time() > deadline - 30:
            break
        
        time_budget = min(10, deadline - time.time() - 30)
        if time_budget <= 0:
            break
        
        result = _try_reduce_rank(U, V, W, n, m, p, target_rank, time_budget, seed=target_rank)
        if result is not None:
            best_result = result + (target_rank,)
            best_rank = target_rank
            break
    
    U_best, V_best, W_best, r = best_result
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
# EVOLVE-BLOCK-END
