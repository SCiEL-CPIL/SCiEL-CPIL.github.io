"""Find rank-minimal tensor decomposition for 3x5 * 5x7 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    """Build schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using numpy."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _try_rank_algebraic(n, m, p, r, max_attempts=200, deadline=None):
    """
    Try to find a rank-r decomposition by fixing V, W randomly and solving for U.
    Returns (U, V, W) if found, else None.
    """
    if deadline is not None and time.time() > deadline:
        return None
    
    for attempt in range(max_attempts):
        if deadline is not None and time.time() > deadline:
            return None
        
        # Generate random V and W with small integer entries
        # Use entries from {-2, -1, 0, 1, 2} but ensure enough non-zeros
        V = np.random.randint(-2, 3, size=(m*p, r))
        W = np.random.randint(-2, 3, size=(p*n, r))
        
        # Ensure V and W have sufficient rank/non-zero structure
        if np.linalg.matrix_rank(V) < min(m*p, r) * 0.5:
            continue
        if np.linalg.matrix_rank(W) < min(p*n, r) * 0.5:
            continue
        
        U = np.zeros((n*m, r))
        valid = True
        
        for i in range(n):
            # Build linear system for U rows corresponding to this i
            # Unknowns: U[i*m+j, c] for j in range(m), c in range(r) -> m*r unknowns
            # Equations: for (j,k) in range(m*p):
            #   sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            A_mat = np.zeros((m*p, m*r))
            b_vec = np.ones(m*p)
            
            for j in range(m):
                for k in range(p):
                    eq_idx = j*p + k
                    for c in range(r):
                        A_mat[eq_idx, j*r + c] = V[j*p+k, c] * W[k*n+i, c]
            
            # Solve least squares
            try:
                sol, residuals, rank_A, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            except:
                valid = False
                break
            
            U[i*m:(i+1)*m, :] = sol.reshape(m, r)
            
            # Check if solution is close to integer or half-integer
            # Scale by 2 and check if close to integers
            scaled = sol.reshape(m, r) * 2
            if not np.all(np.abs(scaled - np.round(scaled)) < 0.01):
                valid = False
                break
        
        if valid:
            # Verify the full decomposition
            if _verify_fast(U, V, W, n, m, p, tol=1e-6):
                # Convert to half-integers if needed
                U_half = np.round(U * 2) / 2
                V_half = np.round(V * 2) / 2
                W_half = np.round(W * 2) / 2
                if _verify_fast(U_half, V_half, W_half, n, m, p, tol=1e-6):
                    return U_half, V_half, W_half
                else:
                    return U, V, W
    
    return None

def _build_strassen_2x2x2():
    """Build rank-7 Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # M1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4 = (b-c)h
    U[1, 3] = 1; U[2, 3] = -1
    V[3, 3] = 1
    W[1, 3] = 1
    
    # M5 = (a+b)g
    U[0, 4] = 1; U[1, 4] = 1
    V[2, 4] = 1
    W[3, 4] = 1
    
    # M6 = (c-d)(f+h)
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[2, 5] = 1
    
    # M7 = (a-c)(e+g)
    U[0, 6] = 1; U[2, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _combine_decompositions(decomps):
    """Combine a list of (U, V, W) decompositions by horizontal concatenation."""
    if not decomps:
        return None
    total_rank = sum(d[0].shape[1] for d in decomps)
    U = np.zeros((decomps[0][0].shape[0], total_rank))
    V = np.zeros((decomps[0][1].shape[0], total_rank))
    W = np.zeros((decomps[0][2].shape[0], total_rank))
    col = 0
    for d in decomps:
        r = d[0].shape[1]
        U[:, col:col+r] = d[0]
        V[:, col:col+r] = d[1]
        W[:, col:col+r] = d[2]
        col += r
    return U, V, W

def _build_block_3x5x7():
    """Block decomposition: (2x5*5x7) + (1x5*5x7) using schoolbook."""
    n, m, p = 3, 5, 7
    n1, m1, p1 = 2, 5, 7
    U1, V1, W1 = _schoolbook(n1, m1, p1)
    U1_full = np.zeros((n*m, U1.shape[1]))
    U1_full[:n1*m1, :] = U1
    V1_full = V1.copy()
    W1_full = np.zeros((p*n, W1.shape[1]))
    for k in range(p1):
        for i in range(n1):
            W1_full[k*n+i, :] = W1[k*n1+i, :]
    
    n2, m2, p2 = 1, 5, 7
    U2, V2, W2 = _schoolbook(n2, m2, p2)
    U2_full = np.zeros((n*m, U2.shape[1]))
    U2_full[n1*m1:, :] = U2
    V2_full = V2.copy()
    W2_full = np.zeros((p*n, W2.shape[1]))
    for k in range(p2):
        W2_full[k*n+2, :] = W2[k*n2+0, :]
    
    return _combine_decompositions([(U1_full, V1_full, W1_full), (U2_full, V2_full, W2_full)])

def _build_strassen_based_3x5x7():
    """
    Try to use Strassen for 2x2 sub-blocks within the 3x5x7 multiplication.
    Split A (3x5) into A_top (2x5) and A_bottom (1x5).
    For A_top * B (2x5 * 5x7): split B into B1 (5x4) and B2 (5x3).
    For A_top * B1 (2x5 * 5x4): split A_top into A1 (2x2) and A2 (2x3).
      A1 * B1 (2x2 * 5x4): use Strassen-like? Not directly.
    
    Alternative: Just use the algebraic search approach.
    """
    n, m, p = 3, 5, 7
    # This is a placeholder; the main improvement comes from algebraic search
    return _schoolbook(n, m, p)

def construct():
    """Find rank-minimal exact tensor decomposition for 3x5 * 5x7 matrix multiplication."""
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Schoolbook decomposition as fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_result = {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
    best_rank = n * m * p
    
    # Try block decomposition (should be same rank as schoolbook for this split)
    U_block, V_block, W_block = _build_block_3x5x7()
    block_rank = U_block.shape[1]
    if _verify_fast(U_block, V_block, W_block, n, m, p, tol=1e-6):
        if block_rank < best_rank:
            best_rank = block_rank
            best_result = {"U": U_block.tolist(), "V": V_block.tolist(), "W": W_block.tolist()}
    
    # Try algebraic search for lower ranks
    # Start from a reasonable lower bound and go up
    # Lower bound considerations:
    # - Must be at least max(n*m, m*p, p*n) = max(15, 35, 21) = 35? No, that's not right.
    # - The rank must be at least the dimension of the tensor's support in some sense.
    # - Known: M(3,5,7) is expected to be less than 105 but the exact value is unknown.
    # - Let's try ranks from 70 to 104 (below schoolbook)
    
    # First, try a few specific ranks that might work
    # The algebraic search is expensive, so be selective
    
    ranks_to_try = []
    # Try ranks that are multiples of small numbers, and some specific values
    for r in range(70, 105):
        ranks_to_try.append(r)
    
    # Sort by likelihood (try lower ranks first, but also consider structure)
    # Actually, let's try a subset first to save time
    # Try ranks: 100, 95, 90, 85, 80, 75, 70 and a few in between
    
    priority_ranks = [100, 95, 90, 85, 80, 75, 72, 70]
    for r in ranks_to_try:
        if r not in priority_ranks:
            priority_ranks.append(r)
    
    random.seed(42)
    np.random.seed(42)
    
    for r in priority_ranks:
        if time.time() > deadline - 5:  # Leave 5s margin
            break
        if r >= best_rank:
            continue
        
        result = _try_rank_algebraic(n, m, p, r, max_attempts=50, deadline=deadline - 5)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify_fast(U_new, V_new, W_new, n, m, p, tol=1e-6):
                if r < best_rank:
                    best_rank = r
                    best_result = {"U": U_new.tolist(), "V": V_new.tolist(), "W": W_new.tolist()}
                    # If we found a good one, try to go lower
                    # But don't spend too much time
                    if r <= 80:
                        # Found a significantly better rank, try a few more lower ranks
                        lower_ranks = [x for x in range(50, r) if x not in priority_ranks]
                        for lr in lower_ranks[:5]:
                            if time.time() > deadline - 3:
                                break
                            result2 = _try_rank_algebraic(n, m, p, lr, max_attempts=20, deadline=deadline - 3)
                            if result2 is not None:
                                U2, V2, W2 = result2
                                if _verify_fast(U2, V2, W2, n, m, p, tol=1e-6):
                                    if lr < best_rank:
                                        best_rank = lr
                                        best_result = {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}
                        break
    
    return best_result
