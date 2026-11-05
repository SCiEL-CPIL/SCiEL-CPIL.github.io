"""Matrix multiplication 3x4x7 rank-minimal decomposition.

Strategy:
  1. Build a deterministic Strassen-based decomposition of rank 82
     (two non-overlapping 2x2x2 Strassen blocks + schoolbook for the rest).
  2. Verify it exactly.
  3. Use the spare time to search for a lower-rank exact decomposition
     via structured random search over small integer/half-integer entries.
"""
import time
import random
import numpy as np
from fractions import Fraction

DEADLINE = 145.0  # safety margin before 150s kill


def _verify(U, V, W, n=3, m=4, p=7, tol=1e-9):
    """Check sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all (i,j,k)."""
    rank = U.shape[1]
    # Build via einsum for speed
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    Ur = U.reshape(n, m, rank)
    Vr = V.reshape(m, p, rank)
    Wr = W.reshape(p, n, rank)
    # Sum over c: Ur[i,j,c] * Vr[j,k,c] * Wr[k,i,c]
    # Use einsum: 'ijc,jkc,kic->ijk'
    result = np.einsum('ijc,jkc,kic->ijk', Ur, Vr, Wr, optimize=True)
    return np.allclose(result, np.ones((n, m, p)), atol=tol)


def _is_half_integer(mat, tol=1e-9):
    """Check all entries are integer or half-integer."""
    flat = mat.flatten()
    doubled = flat * 2
    nearest = np.round(doubled)
    return np.allclose(doubled, nearest, atol=tol)


def _strassen_block(i0, j0, k0, U, V, W, col):
    """Apply Strassen to a 2x2x2 block starting at (i0,j0,k0).
    Returns new column index (col+7)."""
    n, m, p = 3, 4, 7
    # Strassen for C2x2 = A2x2 * B2x2:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Block indices:
    # A rows: i0, i0+1 ; A cols: j0, j0+1
    # B rows: j0, j0+1 ; B cols: k0, k0+1
    # C rows: i0, i0+1 ; C cols: k0, k0+1
    
    # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
    # U entries: (i0,j0)=1, (i0+1,j0+1)=1
    # V entries: (j0,k0)=1, (j0+1,k0+1)=1
    # W entries: (k0,i0)=1, (k0+1,i0+1)=1
    U[i0 * m + j0, col] = 1
    U[(i0 + 1) * m + (j0 + 1), col] = 1
    V[j0 * p + k0, col] = 1
    V[(j0 + 1) * p + (k0 + 1), col] = 1
    W[k0 * n + i0, col] = 1
    W[(k0 + 1) * n + (i0 + 1), col] = 1
    col += 1
    
    # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
    U[(i0 + 1) * m + j0, col] = 1
    U[(i0 + 1) * m + (j0 + 1), col] = 1
    V[j0 * p + k0, col] = 1
    W[k0 * n + (i0 + 1), col] = 1
    col += 1
    
    # M3: A[i0,j0] * (B[j0+1,k0]-B[j0+1,k0+1])
    U[i0 * m + j0, col] = 1
    V[(j0 + 1) * p + k0, col] = 1
    V[(j0 + 1) * p + (k0 + 1), col] = -1
    W[k0 * n + i0, col] = 1
    W[(k0 + 1) * n + i0, col] = -1
    col += 1
    
    # M4: A[i0+1,j0+1] * (B[j0,k0+1]-B[j0,k0])
    U[(i0 + 1) * m + (j0 + 1), col] = 1
    V[j0 * p + (k0 + 1), col] = 1
    V[j0 * p + k0, col] = -1
    W[(k0 + 1) * n + (i0 + 1), col] = 1
    W[k0 * n + (i0 + 1), col] = -1
    col += 1
    
    # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
    U[i0 * m + j0, col] = 1
    U[i0 * m + (j0 + 1), col] = 1
    V[(j0 + 1) * p + (k0 + 1), col] = 1
    W[(k0 + 1) * n + i0, col] = 1
    col += 1
    
    # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
    U[(i0 + 1) * m + j0, col] = 1
    U[i0 * m + j0, col] = -1
    V[j0 * p + k0, col] = 1
    V[j0 * p + (k0 + 1), col] = 1
    W[k0 * n + (i0 + 1), col] = 1
    W[(k0 + 1) * n + (i0 + 1), col] = 1
    col += 1
    
    # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
    U[i0 * m + (j0 + 1), col] = 1
    U[(i0 + 1) * m + (j0 + 1), col] = -1
    V[(j0 + 1) * p + k0, col] = 1
    V[(j0 + 1) * p + (k0 + 1), col] = 1
    W[k0 * n + i0, col] = 1
    W[(k0 + 1) * n + i0, col] = 1
    W[k0 * n + (i0 + 1), col] = -1
    W[(k0 + 1) * n + (i0 + 1), col] = -1
    col += 1
    
    return col


def _build_strassen_rank82():
    """Build the rank-82 Strassen-based decomposition.
    
    Two non-overlapping 2x2x2 Strassen blocks:
      Block 1: i in {0,1}, j in {0,1}, k in {0,1}
      Block 2: i in {1,2}, j in {2,3}, k in {5,6}
    
    Schoolbook for all other (i,j,k) triples.
    Total rank = 7 + 7 + (84 - 4 - 4) = 14 + 76 = 90? No...
    
    Wait: total triples = 3*4*7 = 84.
    Block 1 covers (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1}: 2*2*2 = 8 triples.
    Block 2 covers (i,j,k) where i in {1,2}, j in {2,3}, k in {5,6}: 2*2*2 = 8 triples.
    These two sets are disjoint (different j ranges).
    Schoolbook handles the remaining 84 - 8 - 8 = 68 triples.
    Total rank = 7 + 7 + 68 = 82.
    """
    n, m, p = 3, 4, 7
    total_rank = 82
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    col = 0
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    col = _strassen_block(0, 0, 0, U, V, W, col)
    # Block 2: i in {1,2}, j in {2,3}, k in {5,6}
    col = _strassen_block(1, 2, 5, U, V, W, col)
    
    # Schoolbook for remaining triples
    block1_set = set()
    for i in [0, 1]:
        for j in [0, 1]:
            for k in [0, 1]:
                block1_set.add((i, j, k))
    
    block2_set = set()
    for i in [1, 2]:
        for j in [2, 3]:
            for k in [5, 6]:
                block2_set.add((i, j, k))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in block1_set or (i, j, k) in block2_set:
                    continue
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    assert col == total_rank, f"Expected {total_rank} cols, got {col}"
    return U, V, W


def _build_schoolbook():
    """Standard schoolbook rank-84 decomposition."""
    n, m, p = 3, 4, 7
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W


def _try_random_search(U_best, V_best, W_best, best_rank, start_time, deadline):
    """Try to find a lower-rank decomposition via random search.
    
    Uses structured random patterns with small integer/half-integer values.
    """
    n, m, p = 3, 4, 7
    # Try ranks from best_rank-1 down to a reasonable lower bound
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28, but realistically higher
    # For 3x4x7, known results suggest rank around 80+
    for target_rank in range(best_rank - 1, max(25, best_rank - 15), -1):
        if time.time() - start_time > deadline:
            break
        
        max_attempts = 200
        for attempt in range(max_attempts):
            if time.time() - start_time > deadline:
                break
            
            # Random sparse initialization
            U = np.zeros((n * m, target_rank), dtype=float)
            V = np.zeros((m * p, target_rank), dtype=float)
            W = np.zeros((p * n, target_rank), dtype=float)
            
            for c in range(target_rank):
                # Random support sizes
                n_u = random.randint(1, 5)
                n_v = random.randint(1, 5)
                n_w = random.randint(1, 5)
                
                u_idx = random.sample(range(n * m), min(n_u, n * m))
                v_idx = random.sample(range(m * p), min(n_v, m * p))
                w_idx = random.sample(range(p * n), min(n_w, p * n))
                
                for idx in u_idx:
                    U[idx, c] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
                for idx in v_idx:
                    V[idx, c] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
                for idx in w_idx:
                    W[idx, c] = random.choice([-2, -1, -0.5, 0.5, 1, 2])
            
            # Quick check
            if _verify(U, V, W, n, m, p):
                if _is_half_integer(U) and _is_half_integer(V) and _is_half_integer(W):
                    return U, V, W, target_rank
    
    return None, None, None, best_rank


def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7
    
    # Start with the rank-82 Strassen decomposition
    U, V, W = _build_strassen_rank82()
    best_rank = 82
    
    # Verify
    if not _verify(U, V, W, n, m, p):
        # Fallback to schoolbook
        U, V, W = _build_schoolbook()
        best_rank = 84
    
    # Try to improve
    U2, V2, W2, rank2 = _try_random_search(U, V, W, best_rank, start_time, DEADLINE)
    if U2 is not None:
        U, V, W, best_rank = U2, V2, W2, rank2
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
