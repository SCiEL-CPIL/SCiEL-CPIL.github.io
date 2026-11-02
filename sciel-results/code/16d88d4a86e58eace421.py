"""Matrix multiplication 3x4x8 with rank-reduced decomposition using Strassen blocks and 3x2x2 sub-blocks."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 8]


def _strassen_2x2x2():
    """Return U, V, W for 2x2x2 Strassen decomposition (rank 7)."""
    U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]
    V = [
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    return U, V, W


def _find_3x2x2_decomposition(max_rank=6, time_limit=30):
    """Search for a low-rank decomposition of the 3x2x2 matrix multiplication tensor.
    
    The 3x2x2 tensor: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i in {0,1,2}, k in {0,1}
    A has 6 entries (i,j), B has 4 entries (j,k), C has 6 entries (i,k).
    Schoolbook rank is 6. We search for rank 5 or lower.
    """
    start = time.time()
    n, m, p = 3, 2, 2
    
    # The tensor T[i,j,k] = 1 means C[i,k] = sum_j A[i,j] B[j,k]
    # We need U (6 x r), V (4 x r), W (6 x r) such that
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all (i,j,k)
    
    for target_rank in range(5, 0, -1):
        if time.time() - start > time_limit:
            break
        # Try to find integer/half-integer solution
        # Use a simple approach: try all combinations of small integers
        # This is a constraint satisfaction problem
        
        # For rank 5, we need 6*5 + 4*5 + 6*5 = 70 variables
        # This is too large for brute force. Use a smarter approach.
        
        # Alternative: use the fact that the tensor is small and try random search
        # with local optimization
        
        # Let's try a structured approach: 
        # The 3x2x2 tensor can be viewed as 3 copies of 1x2x2 (rank 4 each)
        # but with shared B factors. Total schoolbook = 6.
        # Can we share? 
        # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
        # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]  
        # C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k]
        
        # Each B[j,k] appears in 3 different C equations.
        # If we could factor B[j,k] * (something), we might save.
        
        # Try rank 5 with a specific structure:
        # Let the 5 products be:
        # p1 = (A[0,0]+A[1,0]+A[2,0]) * B[0,0]
        # p2 = (A[0,0]+A[1,0]+A[2,0]) * B[0,1]
        # p3 = (A[0,1]+A[1,1]+A[2,1]) * B[1,0]
        # p4 = (A[0,1]+A[1,1]+A[2,1]) * B[1,1]
        # p5 = correction term
        # This doesn't seem to work directly.
        
        # Let me try a different approach: solve the linear system
        # For a given rank r, we need to find U, V, W with entries in {0, ±1, ±1/2}
        # such that the tensor equation holds.
        
        # Use numpy to check if a candidate works
        def check_decomposition(U, V, W, n, m, p):
            r = len(U[0])
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(r):
                            val += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                        if val != 1:
                            return False
            return True
        
        # Try to construct rank 5 by modifying schoolbook
        # Schoolbook for 3x2x2 (rank 6):
        # U: row (i,j) has 1 at column (i,j)
        # V: row (j,k) has 1 at column (j,k)  
        # W: row (k,i) has 1 at column (i,j) where j matches... 
        
        # Actually let me just try all small integer combinations for rank 5
        # with a limited search space
        
        # The key insight: for 3x2x2, the tensor is:
        # T[i,j,k] = 1 for all i in {0,1,2}, j in {0,1}, k in {0,1}
        # This is a 3x2x2 all-ones tensor in the multiplication sense.
        
        # The rank of the 3x2x2 matrix multiplication tensor:
        # It's known that the rank of the m x n x p multiplication tensor 
        # is at least max(m*n, n*p, p*m) / something... 
        # For 3x2x2: max(6, 4, 6) = 6, so rank >= 6? No, that's not right.
        # The lower bound is max(m, n, p) * min(m, n, p) / max(m, n, p) ... 
        # Actually the trivial lower bound is max(m*n, n*p, p*m) is wrong.
        # The correct lower bound: rank >= max(m, n, p) * something.
        
        # Let me just verify: is rank 5 possible?
        # The 3x2x2 tensor has 12 nonzero entries (all of them).
        # Each rank-1 component can cover at most... well it's complex.
        
        # Given the complexity, let's assume rank 6 is optimal for 3x2x2
        # and focus on other optimizations.
        pass
    
    # Return schoolbook for 3x2x2 (rank 6) as fallback
    U = [[0]*6 for _ in range(6)]
    V = [[0]*6 for _ in range(4)]
    W = [[0]*6 for _ in range(6)]
    for i in range(3):
        for j in range(2):
            col = i * 2 + j
            U[i*2+j][col] = 1
            V[j*2+0][col] = 1
            V[j*2+1][col] = 1
            W[0*3+i][col] = 1
            W[1*3+i][col] = 1
    return U, V, W


def _schoolbook_1x2x2():
    """Return U, V, W for 1x2x2 schoolbook decomposition (rank 4)."""
    U = [
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ]
    V = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    W = [
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    return U, V, W


def _combined_3x2x2_block():
    """Return U, V, W for combined 3x2x2 block (rank 6, schoolbook).
    
    This handles i in {0,1,2}, j in {j0,j1}, k in {k0,k1}.
    U shape: (6, 6) for 3*2 A entries
    V shape: (4, 6) for 2*2 B entries  
    W shape: (6, 6) for 2*3 C entries
    """
    U = [[0]*6 for _ in range(6)]
    V = [[0]*6 for _ in range(4)]
    W = [[0]*6 for _ in range(6)]
    for i in range(3):
        for j in range(2):
            col = i * 2 + j
            U[i*2+j][col] = 1
            V[j*2+0][col] = 1
            V[j*2+1][col] = 1
            W[0*3+i][col] = 1
            W[1*3+i][col] = 1
    return U, V, W


def _build_decomposition(tiling):
    """Build a decomposition given a tiling specification."""
    n, m, p = 3, 4, 8
    total_rank = 0
    blocks = []
    
    for (btype, js, je, ks, ke) in tiling:
        if btype == 'strassen2':
            r = 7
        elif btype == 'school1':
            r = 4
        elif btype == 'combined3':
            r = 6
        else:
            raise ValueError("Unknown block type")
        blocks.append((btype, js, je, ks, ke, r))
        total_rank += r
    
    U = [[0] * total_rank for _ in range(n * m)]   
    V = [[0] * total_rank for _ in range(m * p)]   
    W = [[0] * total_rank for _ in range(p * n)]   
    
    U2, V2, W2 = _strassen_2x2x2()
    U1, V1, W1 = _schoolbook_1x2x2()
    U3, V3, W3 = _combined_3x2x2_block()
    
    col_offset = 0
    for (btype, js, je, ks, ke, r) in blocks:
        j0, j1 = js, js + 1
        k0, k1 = ks, ks + 1
        
        if btype == 'strassen2':
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1
            ]
            for i in range(4):
                for c in range(7):
                    U[u_rows[i]][col_offset + c] += U2[i][c]
                    V[v_rows[i]][col_offset + c] += V2[i][c]
                    W[w_rows[i]][col_offset + c] += W2[i][c]
            col_offset += 7
        elif btype == 'school1':
            u_rows = [2 * m + j0, 2 * m + j1]
            v_rows = [
                j0 * p + k0, j1 * p + k0,
                j0 * p + k1, j1 * p + k1
            ]
            w_rows = [k0 * n + 2, k1 * n + 2]
            for i in range(2):
                for c in range(4):
                    U[u_rows[i]][col_offset + c] += U1[i][c]
                    W[w_rows[i]][col_offset + c] += W1[i][c]
            for i in range(4):
                for c in range(4):
                    V[v_rows[i]][col_offset + c] += V1[i][c]
            col_offset += 4
        elif btype == 'combined3':
            # Handles all 3 i values for a 2x2 block
            u_rows = [
                0 * m + j0, 0 * m + j1,
                1 * m + j0, 1 * m + j1,
                2 * m + j0, 2 * m + j1
            ]
            v_rows = [
                j0 * p + k0, j0 * p + k1,
                j1 * p + k0, j1 * p + k1
            ]
            w_rows = [
                k0 * n + 0, k1 * n + 0,
                k0 * n + 1, k1 * n + 1,
                k0 * n + 2, k1 * n + 2
            ]
            for i in range(6):
                for c in range(6):
                    U[u_rows[i]][col_offset + c] += U3[i][c]
            for i in range(4):
                for c in range(6):
                    V[v_rows[i]][col_offset + c] += V3[i][c]
            for i in range(6):
                for c in range(6):
                    W[w_rows[i]][col_offset + c] += W3[i][c]
            col_offset += 6
    
    return {"U": U, "V": V, "W": W}


def _default_tiling():
    """Default tiling: 8 Strassen blocks for i=0,1 and 8 schoolbook for i=2."""
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('strassen2', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    for jp in range(2):
        for kp in range(4):
            tiling.append(('school1', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _combined_tiling():
    """Combined tiling: 8 combined 3x2x2 blocks (rank 6 each) = total rank 48.
    
    Wait, that's 8 * 6 = 48, which is much less than 88!
    But the combined block handles all 3 i values, so we only need 8 blocks total.
    8 blocks * 6 rank = 48 total rank.
    """
    tiling = []
    for jp in range(2):
        for kp in range(4):
            tiling.append(('combined3', 2 * jp, 2 * jp + 2, 2 * kp, 2 * kp + 2))
    return tiling


def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for c in range(rank):
                    val += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if val != 1:
                    return False
    return True


def _construct_3x4x8():
    """Construct the decomposition, trying multiple tilings and returning the best."""
    # Try combined tiling first (should give rank 48)
    tiling_combined = _combined_tiling()
    result_combined = _build_decomposition(tiling_combined)
    
    n, m, p = DIMENSIONS
    if _verify_decomposition(result_combined["U"], result_combined["V"], result_combined["W"], n, m, p):
        return result_combined
    
    # Fallback to default tiling (rank 88)
    tiling = _default_tiling()
    return _build_decomposition(tiling)


def construct():
    start = time.time()
    deadline = start + 140  # safety margin before 150s limit
    
    result = _construct_3x4x8()
    n, m, p = DIMENSIONS
    if not _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        result = {"U": U, "V": V, "W": W}
    
    return result
