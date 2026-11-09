"""Matrix multiplication 2x3x6 - rank-minimal tensor decomposition search."""
import numpy as np
import time
import itertools
import random

DIMENSIONS = [2, 3, 6]

def _build_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n * m, m * p, p * n), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _verify_decomposition(U, V, W, T, tol=1e-10):
    """Verify that sum_c U_c ⊗ V_c ⊗ W_c == T exactly (within tol)."""
    r = U.shape[1]
    reconstructed = np.zeros(T.shape, dtype=np.float64)
    for c in range(r):
        reconstructed += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
    diff = np.abs(reconstructed - T)
    return diff.max() < tol, diff.max()

def _is_half_integer(arr, tol=1e-8):
    """Check if all entries are within tol of integers or half-integers."""
    scaled = 2.0 * arr
    rounded = np.round(scaled)
    return np.abs(scaled - rounded).max() < tol

def _round_to_half_int(arr):
    """Round entries to nearest half-integer (0.5 increments)."""
    return np.round(2.0 * arr) / 2.0

def _als_step(U, V, W, T, iters=50, lr=1.0):
    """Alternating least squares to refine decomposition."""
    r = U.shape[1]
    nA, nB, nC = T.shape
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for _ in range(iters):
        # Update U: for each column c, minimize ||T - sum U_c ⊗ V_c ⊗ W_c||^2 w.r.t U_c
        for c in range(r):
            # T reshaped: T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # For fixed c, the residual contribution from U_c is U_c[i] * (V_c ⊗ W_c)[i, k]
            # Actually: T[i, j, k] = sum_c U[i,c] * V[j,c] * W[k,c]
            # Fix all except U[:,c]: we want to fit T[i,j,k] = U[i,c]*V[j,c]*W[k,c] + other_terms
            # This is a linear least squares: for each (j,k), T[i,j,k] = U[i,c] * V[j,c] * W[k,c] + R[i,j,k]
            # R = T - sum_{c'≠c} U[:,c'] ⊗ V[:,c'] ⊗ W[:,c']
            # For each (j,k): T[:,j,k] - R[:,j,k] = U[:,c] * V[j,c] * W[k,c]
            # This is: T[:,j,k] = U[:,c] * (V[j,c]*W[k,c]) + sum_{c'≠c} U[i,c']*V[j,c']*W[k,c]
            # So T[:,j,k] - R[:,j,k] = U[:,c] * V[j,c] * W[k,c]
            # Let's do it properly with matmul
            
            # R[i,j,k] = sum_{c'≠c} U[i,c'] * V[j,c'] * W[k,c']
            # T[i,j,k] = U[i,c] * V[j,c] * W[k,c] + R[i,j,k]
            # For each (j,k): T[:,j,k] = U[:,c] * V[j,c]*W[k,c] + R[:,j,k]
            # If V[j,c]*W[k,c] != 0: U[:,c] = (T[:,j,k] - R[:,j,k]) / (V[j,c]*W[k,c])
            # But this only works if V[j,c]*W[k,c] is the same for all... no.
            
            # Better: treat as linear system. For fixed c, define:
            # M[i, (j,k)] = V[j,c] * W[k,c]
            # Then T[:, :, :].reshape(nA, -1) = U[:,c] @ M + R.reshape(nA, -1)
            # We want U[:,c] such that U[:,c] @ M ≈ T.reshape(nA, -1) - R.reshape(nA, -1)
            
            # Compute R
            R = np.zeros_like(T)
            for c2 in range(r):
                if c2 == c:
                    continue
                R += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
            
            target = T - R
            # M has shape (nA, nB*nC) where M[i, j*p+k] = V[j,c] * W[k,c]
            M = np.outer(V[:, c], W[:, c]).reshape(nB, nC)
            # M_flat has shape (nA, nB*nC)? No, M depends on i through U[i,c]
            # Actually T[i,j,k] = U[i,c] * V[j,c] * W[k,c] + R[i,j,k]
            # So for each i: target[i, :, :] = U[i,c] * np.outer(V[:,c], W[:,c]) + R[i, :, :]
            # This means: target[i] - R[i] = U[i,c] * outer(V[:,c], W[:,c])
            # So U[i,c] = (target[i] - R[i]) / outer(V[:,c], W[:,c]) elementwise, but they should be proportional
            # Use least squares: minimize ||target - U[:,c] * M_broadcast - R||^2
            # where M_broadcast[i,j,k] = M[j,k]
            # This is: for each (j,k), target[i,j,k] - R[i,j,k] = U[i,c] * M[j,k]
            # So we have nB*nC equations: for each (j,k), the vector (target[:,j,k] - R[:,j,k]) = U[:,c] * M[j,k]
            # If M[j,k] != 0: U[:,c] = (target[:,j,k] - R[:,j,k]) / M[j,k]
            # Use all (j,k) with M[j,k] != 0, average
            
            M_flat = M.flatten()  # shape (nB*nC,)
            residual_flat = (target - R).reshape(nA, -1)  # shape (nA, nB*nC)
            mask = np.abs(M_flat) > 1e-12
            if mask.sum() == 0:
                continue
            # For each (j,k) in mask: U[:,c] should equal residual_flat[:, jk] / M_flat[jk]
            # Take the average
            U_new = np.zeros(nA)
            count = 0
            for jk in range(len(M_flat)):
                if mask[jk]:
                    U_new += residual_flat[:, jk] / M_flat[jk]
                    count += 1
            if count > 0:
                U_new /= count
                # Damping
                U[:, c] = (1 - lr) * U[:, c] + lr * U_new
        
        # Similar updates for V and W
        for c in range(r):
            R = np.zeros_like(T)
            for c2 in range(r):
                if c2 == c:
                    continue
                R += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
            target = T - R
            # target[i,j,k] = V[j,c] * U[i,c] * W[k,c] + R[i,j,k]
            M = np.outer(U[:, c], W[:, c])  # shape (nA, nC)
            residual_flat = (target - R).reshape(-1, nB)  # shape (nA*nC, nB)
            M_flat = M.flatten()  # shape (nA*nC,)
            mask = np.abs(M_flat) > 1e-12
            if mask.sum() == 0:
                continue
            V_new = np.zeros(nB)
            count = 0
            for idx in range(len(M_flat)):
                if mask[idx]:
                    V_new += residual_flat[idx, :] / M_flat[idx]
                    count += 1
            if count > 0:
                V_new /= count
                V[:, c] = (1 - lr) * V[:, c] + lr * V_new
        
        for c in range(r):
            R = np.zeros_like(T)
            for c2 in range(r):
                if c2 == c:
                    continue
                R += np.einsum('i,j,k->ijk', U[:, c2], V[:, c2], W[:, c2])
            target = T - R
            # target[i,j,k] = W[k,c] * U[i,c] * V[j,c] + R[i,j,k]
            M = np.outer(U[:, c], V[:, c])  # shape (nA, nB)
            residual_flat = (target - R).reshape(-1, nC)  # shape (nA*nB, nC)
            M_flat = M.flatten()  # shape (nA*nB,)
            mask = np.abs(M_flat) > 1e-12
            if mask.sum() == 0:
                continue
            W_new = np.zeros(nC)
            count = 0
            for idx in range(len(M_flat)):
                if mask[idx]:
                    W_new += residual_flat[idx, :] / M_flat[idx]
                    count += 1
            if count > 0:
                W_new /= count
                W[:, c] = (1 - lr) * W[:, c] + lr * W_new
    
    return U, V, W

def _try_rank_decomposition(T, rank, n_iters=50, max_random_tries=50, time_limit=None):
    """Try to find a rank-decomposition with integer/half-integer entries."""
    nA, nB, nC = T.shape
    best_result = None
    best_max_err = float('inf')
    
    for attempt in range(max_random_tries):
        if time_limit is not None and time.time() > time_limit:
            break
        
        # Random initialization
        U = np.random.randn(nA, rank)
        V = np.random.randn(nB, rank)
        W = np.random.randn(nC, rank)
        
        # Scale to reasonable magnitude
        U /= np.linalg.norm(U) + 1e-10
        V /= np.linalg.norm(V) + 1e-10
        W /= np.linalg.norm(W) + 1e-10
        
        # ALS refinement
        U, V, W = _als_step(U, V, W, T, iters=n_iters, lr=0.5)
        
        # Check if close to solution
        reconstructed = np.zeros(T.shape)
        for c in range(rank):
            reconstructed += np.einsum('i,j,k->ijk', U[:, c], V[:, c], W[:, c])
        max_err = np.abs(reconstructed - T).max()
        
        if max_err < 1e-6:
            # Check if entries are close to half-integers
            if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                # Round and verify
                Ur = _round_to_half_int(U)
                Vr = _round_to_half_int(V)
                Wr = _round_to_half_int(W)
                ok, err = _verify_decomposition(Ur, Vr, Wr, T)
                if ok:
                    return Ur.tolist(), Vr.tolist(), Wr.tolist()
                best_max_err = min(best_max_err, err)
            else:
                # Try rounding anyway
                Ur = _round_to_half_int(U)
                Vr = _round_to_half_int(V)
                Wr = _round_to_half_int(W)
                ok, err = _verify_decomposition(Ur, Vr, Wr, T)
                if ok:
                    return Ur.tolist(), Vr.tolist(), Wr.tolist()
        
        best_max_err = min(best_max_err, max_err)
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
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

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    T = _build_tensor(n, m, p)
    
    start_time = time.time()
    # Hard deadline: must return before 120s, use 115s as safety margin
    deadline = start_time + 115.0
    
    # Try ranks from low to high
    for rank in [24, 27, 28, 29, 30]:
        if time.time() > deadline - 10:
            break
        
        # Use a time limit for this rank attempt
        rank_deadline = min(deadline - 5.0, time.time() + 30.0)
        
        result = _try_rank_decomposition(T, rank, n_iters=80, max_random_tries=100, time_limit=rank_deadline)
        
        if result is not None:
            U_list, V_list, W_list = result
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
