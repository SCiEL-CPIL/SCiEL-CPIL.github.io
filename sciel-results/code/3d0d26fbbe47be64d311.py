"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.time()
TIME_BUDGET = 125.0

def _time_left():
    return max(0.0, TIME_BUDGET - (time.time() - _start_time))

def _verify(U, V, W, n, m, p):
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > 1e-8:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
    r = n*m*p
    U = [[0]*r for _ in range(n*m)]
    V = [[0]*r for _ in range(m*p)]
    W = [[0]*r for _ in range(p*n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2_2():
    """Return U, V, W arrays for a single 2x2x2 Strassen block (rank 7)."""
    # Strassen's 7 products for 2x2 matrix multiplication
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    n, m, p = 2, 2, 2
    r = 7
    
    U = np.zeros((n*m, r))
    V = np.zeros((m*p, r))
    W = np.zeros((p*n, r))
    
    # A indices: 00, 01, 10, 11 -> row indices 0,1,2,3
    # B indices: 00, 01, 10, 11 -> row indices 0,1,2,3
    # C indices: 00, 01, 10, 11 -> row indices 0,1,2,3
    
    # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = A11(B21-B22) -> C12 += M3, C22 += M3
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A22(B12-B11) -> C11 += M4, C21 += M4
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (A21-A11)(B11+B12) -> C22 += M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (A12-A22)(B21+B22) -> C11 += M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_strassen_embedding():
    """
    Build a decomposition for 3x3x8 using Strassen blocks.
    
    Strategy: Embed 2x2x2 Strassen blocks into the 3x3x8 problem.
    We can place a 2x2x2 block in the top-left 2x2 of A, first 2 columns of B,
    and first 2 rows of C. The remaining parts use schoolbook.
    
    For 3x3x8:
    - A is 3x3, B is 3x8, C is 3x8
    - Place one 2x2x2 Strassen block: uses rows 0,1 of A, rows 0,1 of B (first 2 cols), cols 0,1 of C
    - This covers C[0:2, 0:2] using 7 products instead of 4
    
    But we need to be careful about overlap. Let's try a different approach:
    Split the 8 columns of B into groups and use Strassen where possible.
    
    Actually, let's try: use Strassen for a 2x2x2 subproblem in the top-left,
    and schoolbook for the rest. This gives rank = 7 + (3*3*8 - 2*2*2) = 7 + 64 = 71.
    Not great.
    
    Better idea: Use multiple non-overlapping Strassen blocks.
    For 3x3x8, we can potentially fit one 2x2x2 block in the top-left.
    Remaining: rows 2 of A, all of B, all of C except the covered part.
    
    Let me think more carefully. The tensor is T[i,j,k] for i in [0,3), j in [0,3), k in [0,8).
    A 2x2x2 Strassen block at offset (i0, j0, k0) covers T[i0+i', j0+j', k0+k'] for i',j',k' in [0,2).
    
    If we place one block at (0,0,0): covers i in {0,1}, j in {0,1}, k in {0,1}
    Rank contribution: 7
    Remaining: all other (i,j,k) triples: 72 - 4 = 68, each needs rank 1
    Total: 7 + 68 = 75. Worse than schoolbook!
    
    The issue is that Strassen only helps when the 2x2x2 block is isolated.
    For matrix multiplication, the full 3x3x8 problem has dependencies that
    prevent simple block decomposition from helping.
    
    Let's try a different approach: look for linear dependencies in the tensor.
    """
    n, m, p = 3, 3, 8
    r_total = n * m * p  # 72
    
    # Try to find if we can reduce rank by exploiting structure
    # For now, return None to indicate no improvement found
    return None

def _try_rank_64():
    """
    Try to construct a rank-64 decomposition using a structured approach.
    """
    return None

def construct():
    n, m, p = DIMENSIONS
    best = _schoolbook()
    
    # Try lower rank decompositions
    candidates = [_build_strassen_embedding(), _try_rank_64()]
    for cand in candidates:
        if cand is not None and _verify(cand["U"], cand["V"], cand["W"], n, m, p):
            r = len(cand["U"][0]) if isinstance(cand["U"], list) else cand["U"].shape[1]
            best_r = len(best["U"][0]) if isinstance(best["U"], list) else best["U"].shape[1]
            if r < best_r:
                best = cand
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
# EVOLVE-BLOCK-END
