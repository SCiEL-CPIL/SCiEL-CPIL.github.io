"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly reproduces the tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that no spurious terms exist
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) not in valid:
                    val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                    if abs(val) > 1e-9:
                        return False
    return True

def _schoolbook(n, m, p):
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
    return U, V, W

def _strassen_block(n, m, p):
    """
    For multiplying (n x m) by (m x p) where n=m=2, p=16.
    The 2x16 matrix can be seen as 8 independent 2x2 blocks.
    Each 2x2x2 multiplication has rank 7 (Strassen).
    Total rank = 8 * 7 = 56.
    
    For block t (0..7), the columns in B are at indices 2t and 2t+1.
    Output rows in C are at indices 2t*n+0, 2t*n+1, (2t+1)*n+0, (2t+1)*n+1.
    """
    n, m, p = 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    rank = num_blocks * 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for t in range(num_blocks):
        # V indices for B entries in this block:
        # B[0, 2t] -> V[2t]
        # B[0, 2t+1] -> V[2t+1]
        # B[1, 2t] -> V[16+2t]
        # B[1, 2t+1] -> V[16+2t+1]
        
        # W indices for C entries in this block:
        # C[0, 2t] -> W[4t]
        # C[1, 2t] -> W[4t+1]
        # C[0, 2t+1] -> W[4t+2]
        # C[1, 2t+1] -> W[4t+3]
        
        # U indices: A00=0, A01=1, A10=2, A11=3
        
        intermediates = [
            # M1 = (A00 + A11) * (B00 + B11)
            {'U': [(0, 1), (3, 1)],
             'V': [(2*t, 1), (16+2*t+1, 1)],
             'W': [(4*t, 1), (4*t+3, 1)]},
            # M2 = (A10 + A11) * B00
            {'U': [(2, 1), (3, 1)],
             'V': [(2*t, 1)],
             'W': [(4*t+1, 1), (4*t+3, -1)]},
            # M3 = A00 * (B01 - B11)
            {'U': [(0, 1)],
             'V': [(2*t+1, 1), (16+2*t+1, -1)],
             'W': [(4*t+2, 1), (4*t+3, 1)]},
            # M4 = A11 * (B10 - B00)
            {'U': [(3, 1)],
             'V': [(16+2*t, 1), (2*t, -1)],
             'W': [(4*t, 1), (4*t+1, 1)]},
            # M5 = (A00 + A01) * B11
            {'U': [(0, 1), (1, 1)],
             'V': [(16+2*t+1, 1)],
             'W': [(4*t, -1), (4*t+2, 1)]},
            # M6 = (A10 - A00) * (B00 + B01)
            {'U': [(2, 1), (0, -1)],
             'V': [(2*t, 1), (2*t+1, 1)],
             'W': [(4*t+3, 1)]},
            # M7 = (A01 - A11) * (B10 + B11)
            {'U': [(1, 1), (3, -1)],
             'V': [(16+2*t, 1), (16+2*t+1, 1)],
             'W': [(4*t, 1)]}
        ]
        
        for inter in intermediates:
            for idx, coeff in inter['U']:
                U[idx][col] += coeff
            for idx, coeff in inter['V']:
                V[idx][col] += coeff
            for idx, coeff in inter['W']:
                W[idx][col] += coeff
            col += 1
    
    return U, V, W

def _strassen_4x4x16(n, m, p):
    """
    Try to reduce rank below 56 by exploiting the shared A-structure.
    
    Key insight: The U matrix in the rank-56 Strassen decomposition has
    only 4 distinct row patterns (one for each A entry). This means the
    56 rank-1 tensors share structure. We can try to find a lower-rank
    decomposition by solving a linear system.
    
    Approach: Start with the rank-56 Strassen decomposition and try to
    eliminate redundant dimensions by finding linear dependencies.
    
    Alternative: Use the rank-19 4x4x4 Strassen algorithm by embedding
    the 2x2x2x16 problem into a 4x4x4x4 framework.
    
    For now, try a targeted search: check if any of the 56 Strassen
    rank-1 tensors can be expressed as a linear combination of the others
    (which would indicate the rank can be reduced).
    """
    n, m, p = 2, 2, 16
    # Return the standard Strassen (rank 56) as a safe baseline
    # The rank reduction search is complex and may not converge in time
    return _strassen_block(n, m, p)

def _try_rank_reduction(U, V, W, n, m, p, max_time=10.0):
    """
    Try to reduce the rank of a tensor decomposition by finding linear
    dependencies among the rank-1 components.
    
    Given U (nm x r), V (mp x r), W (pn x r), try to find a subset of
    columns that can be eliminated while preserving the tensor.
    
    This is done by checking if any column r can be expressed as a linear
    combination of other columns in a way that preserves the tensor identity.
    """
    start_time = time.time()
    U = [list(row) for row in U]
    V = [list(row) for row in V]
    W = [list(row) for row in W]
    rank = len(U[0])
    
    # Build the set of valid tensor entries
    valid = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                valid.add((i*m+j, j*p+k, k*n+i))
    
    all_entries = []
    for a in range(n*m):
        for b in range(m*p):
            for c in range(p*n):
                if (a, b, c) in valid:
                    all_entries.append((a, b, c))
    
    # Try to eliminate columns one at a time
    # For each column r, check if it can be removed
    # by adjusting other columns
    
    # This is a complex optimization problem. For simplicity, try a
    # greedy approach: for each column, see if setting it to zero and
    # adjusting others can still satisfy the tensor.
    
    # Actually, a simpler approach: check if the current decomposition
    # has any obviously redundant columns (e.g., two columns that are
    # proportional in all three matrices).
    
    for r1 in range(rank):
        if time.time() - start_time > max_time:
            break
        for r2 in range(r1 + 1, rank):
            # Check if column r1 and r2 are proportional
            # in all three matrices
            if U[0][r1] != 0 and U[0][r2] != 0:
                ratio = U[0][r2] / U[0][r1]
                proportional = True
                for a in range(1, n*m):
                    if abs(U[a][r1] * ratio - U[a][r2]) > 1e-9:
                        proportional = False
                        break
                if proportional:
                    for b in range(m*p):
                        if abs(V[b][r1] * ratio - V[b][r2]) > 1e-9:
                            proportional = False
                            break
                if proportional:
                    for c in range(p*n):
                        if abs(W[c][r1] * ratio - W[c][r2]) > 1e-9:
                            proportional = False
                            break
                if proportional:
                    # Remove column r2
                    U = [row[:r2] + row[r2+1:] for row in U]
                    V = [row[:r2] + row[r2+1:] for row in V]
                    W = [row[:r2] + row[r2+1:] for row in W]
                    rank = len(U[0])
                    break
        if time.time() - start_time > max_time:
            break
    
    if _verify(U, V, W, n, m, p):
        return U, V, W, rank
    
    return None, None, None, None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    # Try Strassen-based construction (rank 56 = 8 * 7)
    U, V, W = _strassen_block(n, m, p)
    if _verify(U, V, W, n, m, p):
        # Try to reduce rank
        if time.time() < deadline - 15:
            U_red, V_red, W_red, rank_red = _try_rank_reduction(
                U, V, W, n, m, p, max_time=10.0
            )
            if U_red is not None and rank_red < 56:
                return {"U": U_red, "V": V_red, "W": W_red}
        
        return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook (rank 64)
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
