import time
import numpy as np
from itertools import product

DIMENSIONS = [4, 4, 8]
_START_TIME = time.time()
_TIME_LIMIT = 145.0

def _strassen_222():
    """Rank-7 decomposition for 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=float)
    V = np.zeros((4, 7), dtype=float)
    W = np.zeros((4, 7), dtype=float)
    
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _kron_decomposition(U1, V1, W1, U2, V2, W2):
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, _ = V1.shape
    m2, _ = V2.shape
    p1, _ = W1.shape
    p2, _ = W2.shape
    
    R = r1 * r2
    U = np.zeros((n1 * n2, R), dtype=float)
    V = np.zeros((m1 * m2, R), dtype=float)
    W = np.zeros((p1 * p2, R), dtype=float)
    
    for r1_idx in range(r1):
        for r2_idx in range(r2):
            r_idx = r1_idx * r2 + r2_idx
            U[:, r_idx] = np.kron(U1[:, r1_idx], U2[:, r2_idx])
            V[:, r_idx] = np.kron(V1[:, r1_idx], V2[:, r2_idx])
            W[:, r_idx] = np.kron(W1[:, r1_idx], W2[:, r2_idx])
            
    return U, V, W

def _direct_sum_decomposition(U1, V1, W1):
    n, r = U1.shape
    m, _ = V1.shape
    p, _ = W1.shape
    
    R = 2 * r
    U = np.zeros((n, R), dtype=float)
    V = np.zeros((2 * m, R), dtype=float)
    W = np.zeros((2 * p, R), dtype=float)
    
    U[:, :r] = U1
    V[:m, :r] = V1
    W[:p, :r] = W1
    
    U[:, r:] = U1
    V[m:, r:] = V1
    W[p:, r:] = W1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, tol=1e-8):
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i*m + j, :], V[j*p + k, :] * W[k*n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    R = n * m * p
    U = np.zeros((n*m, R))
    V = np.zeros((m*p, R))
    W = np.zeros((p*n, R))
    col = 0
    for ii in range(n):
        for jj in range(m):
            for kk in range(p):
                U[ii*m + jj, col] = 1
                V[jj*p + kk, col] = 1
                W[kk*n + ii, col] = 1
                col += 1
    return U, V, W

def _greedy_column_elimination(U, V, W, n, m, p, deadline):
    """
    Try to eliminate columns by checking if a column can be expressed
    as a linear combination of others in all three matrices simultaneously.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    R = U.shape[1]
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        R = U.shape[1]
        if R <= 1:
            break
        
        # Try each column from the end
        for k in range(R - 1, -1, -1):
            if time.time() > deadline:
                return U, V, W
            
            U_rest = np.delete(U, k, axis=1)
            V_rest = np.delete(V, k, axis=1)
            W_rest = np.delete(W, k, axis=1)
            
            # Check if column k is in the span of the rest for all three
            # Use least squares to find coefficients
            try:
                # For U
                coeffs_U, _, _, _ = np.linalg.lstsq(U_rest, U[:, k], rcond=None)
                U_resid = np.linalg.norm(U[:, k] - U_rest @ coeffs_U)
                
                # For V  
                coeffs_V, _, _, _ = np.linalg.lstsq(V_rest, V[:, k], rcond=None)
                V_resid = np.linalg.norm(V[:, k] - V_rest @ coeffs_V)
                
                # For W
                coeffs_W, _, _, _ = np.linalg.lstsq(W_rest, W[:, k], rcond=None)
                W_resid = np.linalg.norm(W[:, k] - W_rest @ coeffs_W)
                
                # All three must be simultaneously eliminable
                # This is a stronger condition: the same linear combination must work
                # Actually, we need: there exist coeffs c such that
                # U[:,k] = U_rest @ c, V[:,k] = V_rest @ c, W[:,k] = W_rest @ c
                # This is a joint condition.
                
                # Build the joint system
                # [U_rest; V_rest; W_rest] @ c = [U[:,k]; V[:,k]; W[:,k]]
                stacked_A = np.vstack([U_rest, V_rest, W_rest])
                stacked_b = np.concatenate([U[:, k], V[:, k], W[:, k]])
                
                coeffs, residuals, rank, _ = np.linalg.lstsq(stacked_A, stacked_b, rcond=None)
                residual = np.linalg.norm(stacked_A @ coeffs - stacked_b)
                
                if residual < 1e-7:
                    U = np.delete(U, k, axis=1)
                    V = np.delete(V, k, axis=1)
                    W = np.delete(W, k, axis=1)
                    changed = True
                    break  # Restart the loop
            except:
                continue
    
    return U, V, W

def _randomized_merge_search(U, V, W, n, m, p, deadline, n_trials=50):
    """
    Try to merge pairs of columns by finding a linear combination
    that preserves the tensor decomposition.
    """
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for trial in range(n_trials):
        if time.time() > deadline:
            break
        
        R = U.shape[1]
        if R <= 1:
            break
        
        # Pick two random columns to try to merge
        i, j = np.random.choice(R, 2, replace=False)
        
        # Try to find alpha, beta such that we can replace columns i and j
        # with a single column, and adjust the rest.
        # This is complex; instead, try: can column i be expressed in terms of others?
        # We already do this in greedy. Instead, try a perturbation approach.
        
        # Try: perturb column i slightly, then project back
        # This is heuristic and may not help much.
        # Skip for now.
        pass
    
    return U, V, W

def _structured_448_search(deadline):
    """
    Try a structured search for 4x4x8 decomposition.
    Uses the fact that 4x4x8 can be decomposed as:
    - Two 4x4x4 blocks (direct sum, rank 2*49=98)
    - Or a more clever factorization.
    
    Try: 4x4x8 = (2x2) kron (2x4x4)
    where 2x4x4 has some rank r, and 2x2 has rank 7 (Strassen).
    Total rank = 7 * r.
    
    What's rank(2x4x4)? U: 8×r, V: 16×r, W: 8×r.
    Schoolbook: 2*4*4 = 32.
    Can we do better? 2x4x4 = (2x2x2) + (2x2x2) + (2x2x2) + (2x2x2) in some grouping.
    Actually, 2x4x4: n=2, m=4, p=4.
    Split m=4 into two 2's: two 2x2x4 problems? No, p=4.
    Split p=4 into two 2's: two 2x4x2 problems.
    2x4x2: n=2, m=4, p=2. Schoolbook = 16.
    2x4x2 can be seen as: for each of the 2 p-values, we have a 2x4x1 problem.
    2x4x1: U is 8×r, V is 4×r, W is 2×r. This is just a matrix-vector product.
    rank(2x4x1) = 8 (schoolbook), but can we do better? 
    Actually 2x4x1: C[0,:] = A[0,:] * B[:,0] + A[0,1] * B[1,0] + ... 
    This is 4 independent products for each output, so rank = 4*2 = 8? No.
    
    Let me just try various constructions and verify.
    """
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # Construction 1: Kron of Strassen 2x2x2 (rank 7) with a 2x2x4 decomposition
    # For 2x2x4: U2: 4×r2, V2: 8×r2, W2: 8×r2
    # Try r2 = 7 (optimistic) to 14 (pessimistic)
    
    # First, get a valid 2x2x4 decomposition (schoolbook, rank 16)
    # Then try to reduce.
    
    # Actually, let's try: 2x2x4 as direct sum of two 2x2x2 (rank 7+7=14)
    U2, V2, W2 = _strassen_222()  # 4×7, 4×7, 4×7 for 2x2x2
    # For 2x2x4: n=2, m=2, p=4
    # U: 4×r, V: 8×r, W: 8×r
    # Direct sum of two 2x2x2:
    # First 2x2x2: U1: 4×7, V1: 4×7, W1: 4×7
    # Second 2x2x2: U2: 4×7, V2: 4×7, W2: 4×7
    # Combined: U: 4×14, V: 8×14, W: 8×14
    U_224 = np.zeros((4, 14))
    V_224 = np.zeros((8, 14))
    W_224 = np.zeros((8, 14))
    
    U_224[:, :7] = U2
    V_224[:4, :7] = V2
    W_224[:4, :7] = W2
    
    U_224[:, 7:] = U2
    V_224[4:, 7:] = V2
    W_224[4:, 7:] = W2
    
    # Verify 2x2x4
    if _verify_decomposition(U_224, V_224, W_224, 2, 2, 4):
        # Now Kron with Strassen 2x2x2
        U1, V1, W1 = _strassen_222()  # for the 2x2 part of n and m
        U_448, V_448, W_448 = _kron_decomposition(U1, V1, W1, U_224, V_224, W_224)
        # U_448: 16×98, V_448: 32×98, W_448: 32×98
        # That's rank 98, same as before.
        
        if U_448.shape[1] < best_rank:
            if _verify_decomposition(U_448, V_448, W_448, 4, 4, 8):
                best_rank = U_448.shape[1]
                best_U, best_V, best_W = U_448, V_448, W_448
    
    # Try a different split: 4x4x8 = (4x2) kron (1x2x4)?
    # 4x2x? doesn't make sense for matrix mult.
    
    # Try: 4x4x8 with a 4x4x2 (rank r_a) and combine
    # 4x4x2: U: 16×r, V: 8×r, W: 8×r
    # Schoolbook: 32. Strassen-based: ?
    
    # For now, return the best found
    if best_U is not None:
        return best_U, best_V, best_W
    
    return None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + _TIME_LIMIT
    
    # Start with the known rank-98 decomposition
    U2, V2, W2 = _strassen_222()
    U444, V444, W444 = _kron_decomposition(U2, V2, W2, U2, V2, W2)
    U, V, W = _direct_sum_decomposition(U444, V444, W444)
    
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try structured search
    if time.time() < deadline:
        sU, sV, sW = _structured_448_search(deadline)
        if sU is not None and sU.shape[1] < best_rank:
            if _verify_decomposition(sU, sV, sW, n, m, p):
                best_U, best_V, best_W = sU, sV, sW
                best_rank = sU.shape[1]
    
    # Try greedy column elimination
    if time.time() < deadline:
        U2, V2, W2 = _greedy_column_elimination(best_U, best_V, best_W, n, m, p, deadline)
        if U2.shape[1] < best_rank and _verify_decomposition(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Try randomized merge search
    if time.time() < deadline:
        U2, V2, W2 = _randomized_merge_search(best_U, best_V, best_W, n, m, p, deadline)
        if U2.shape[1] < best_rank and _verify_decomposition(U2, V2, W2, n, m, p):
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Final verification
    if not _verify_decomposition(best_U, best_V, best_W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
