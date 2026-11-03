"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations, product

def _verify_decomp(U, V, W, n, m, p, rank, tol=1e-9):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > tol:
                    return False
    return True

def _strassen_block(U, V, W, i0, i1, j0, j1, k0, k1, m, p, n, col):
    """Apply Strassen's 7-product scheme to a 2x2x2 block.
    i0,i1: the two i indices
    j0,j1: the two j (l) indices
    k0,k1: the two k indices
    Returns new col index."""
    # P1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
    U[i0*m + j0][col] += 1
    U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
    U[i1*m + j0][col] += 1
    U[i1*m + j1][col] += 1
    V[j0*p + k0][col] += 1
    W[k0*n + i1][col] += 1
    W[k1*n + i1][col] -= 1
    col += 1
    
    # P3 = A[i0,j0] * (B[j0,k1]-B[j1,k1])
    U[i0*m + j0][col] += 1
    V[j0*p + k1][col] += 1
    V[j1*p + k1][col] -= 1
    W[k1*n + i0][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P4 = A[i1,j1] * (B[j1,k0]-B[j0,k0])
    U[i1*m + j1][col] += 1
    V[j1*p + k0][col] += 1
    V[j0*p + k0][col] -= 1
    W[k0*n + i0][col] += 1
    W[k0*n + i1][col] += 1
    col += 1
    
    # P5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
    U[i0*m + j0][col] += 1
    U[i0*m + j1][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] -= 1
    W[k1*n + i0][col] += 1
    col += 1
    
    # P6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
    U[i1*m + j0][col] += 1
    U[i0*m + j0][col] -= 1
    V[j0*p + k0][col] += 1
    V[j0*p + k1][col] += 1
    W[k1*n + i1][col] += 1
    col += 1
    
    # P7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
    U[i0*m + j1][col] += 1
    U[i1*m + j1][col] -= 1
    V[j1*p + k0][col] += 1
    V[j1*p + k1][col] += 1
    W[k0*n + i0][col] += 1
    col += 1
    
    return col

def _build_rank_54():
    """Build the rank-54 Strassen-based decomposition."""
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = _strassen_block(U, V, W, 0, 1, j0, j1, k0, k1, m, p, n, col)
    
    # Remaining j=4: standard products
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

def _build_rank_53():
    """Try to build a rank-53 decomposition by combining the j=4 terms more cleverly.
    
    For j=4, we have 12 scalar products. Can we save 1 rank?
    The 2x1x6 sub-tensor has rank 12 in general, but let's try to see if we can
    combine with the Strassen blocks.
    
    Actually, let's try a different approach: use a 2x2x2 Strassen block for
    j=(3,4) and k=(4,5) instead of separate blocks.
    """
    n, m, p = DIMENSIONS
    # Try: j_pairs = [(0,1), (2,3)], k_pairs = [(0,1), (2,3), (4,5)] gives 6 blocks = 42 rank
    # Plus j=4 alone: 12 rank -> total 54
    
    # Alternative: j_pairs = [(0,1), (2,4)], k_pairs = [(0,1), (2,3), (4,5)]
    # But j=3 and j=5 are left over... this gets complicated.
    
    # Let's try: use Strassen for all 6 j-k pairs where we pair up j indices
    # j: 0,1,2,3,4 (5 indices)
    # k: 0,1,2,3,4,5 (6 indices)
    
    # Option: pair j as (0,1), (2,3), and handle j=4 separately
    # Pair k as (0,1), (2,3), (4,5)
    # 2 j-pairs * 3 k-pairs = 6 Strassen blocks = 42 rank
    # j=4 with all 6 k's = 12 scalar products
    # Total: 54
    
    # Can we do better by using a 2x2x3 or other non-cubic blocks?
    # For 2x2x3: minimum rank is 12 (vs 12 scalar products) - no savings
    # For 2x3x3: minimum rank is 18 (vs 18 scalar products) - no savings
    
    # What about overlapping blocks? E.g., use j=(1,2) in one block and j=(2,3) in another?
    # This would create interference. Let's think...
    
    # Actually, the key insight: we can use a 2x2x2 Strassen block for ANY pair of j's and ANY pair of k's.
    # The question is whether we can cover all 5*6=30 (j,k) pairs with fewer than 6 blocks + 12 scalars.
    
    # Let's try: 7 Strassen blocks (49 rank) + 1 scalar = 50? No, that doesn't work.
    # 5 Strassen blocks (35 rank) + remaining terms...
    
    # Actually, let's just try to eliminate one rank from the 54 by finding a linear dependency.
    
    return None  # Fall back to rank 54

def _try_reduce_rank(U, V, W, rank, n, m, p, max_attempts=100):
    """Try to reduce rank by finding linear dependencies among the terms."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check if any two terms are linearly dependent
    # Term r is a vector in R^(n*m) x R^(m*p) x R^(p*n)
    # Two terms r1, r2 are dependent if U[r1] = a*U[r2], V[r1] = b*V[r2], W[r1] = c*W[r2]
    # for some scalars a, b, c
    
    # This is complex. Let's try a simpler approach: check if removing one term
    # and adjusting others still gives a valid decomposition.
    
    # For each term r, check if we can express it as a linear combination of others
    for r_remove in range(rank):
        # Try to find coefficients for all other terms
        # This is a system of equations, but it's underdetermined in general
        
        # Simpler: just check if term r is zero
        if (np.all(U_arr[r_remove] == 0) and 
            np.all(V_arr[r_remove] == 0) and 
            np.all(W_arr[r_remove] == 0)):
            # Remove this term
            new_rank = rank - 1
            new_U = [row[:] for row in U]
            new_V = [row[:] for row in V]
            new_W = [row[:] for row in W]
            
            # Remove column r_remove
            for i in range(n*m):
                new_U[i].pop(r_remove)
            for i in range(m*p):
                new_V[i].pop(r_remove)
            for i in range(p*n):
                new_W[i].pop(r_remove)
            
            if _verify_decomp(new_U, new_V, new_W, n, m, p, new_rank):
                return new_U, new_V, new_W, new_rank
    
    return U, V, W, rank

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 140  # 10s safety margin before 150s kill
    
    # Build rank-54 decomposition
    U, V, W, rank = _build_rank_54()
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Verify
    if not _verify_decomp(best_U, best_V, best_W, n, m, p, best_rank):
        # Fallback to schoolbook if verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] = 1
                    V[l*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Try to reduce rank
    if time.time() < _deadline - 5:
        U, V, W, rank = _try_reduce_rank(U, V, W, rank, n, m, p)
        if rank < best_rank:
            best_rank = rank
            best_U = U
            best_V = V
            best_W = W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
