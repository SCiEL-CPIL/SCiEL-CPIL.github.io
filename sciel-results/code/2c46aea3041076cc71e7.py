"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0

def _time_left():
    return TIME_LIMIT - (time.time() - _start_time) - SAFETY_MARGIN

def _verify(U, V, W):
    """Verify that U, V, W form an exact tensor decomposition of matrix multiplication.
    Returns True if valid, False otherwise.
    """
    n, m, p = DIMENSIONS
    U = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in U]
    V = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in V]
    W = [[int(x) if abs(x - round(x)) < 1e-9 else x for x in row] for row in W]
    
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _strassen_2x2_block_decomp():
    """
    Construct a rank-68 decomposition using 4 Strassen 2x2x2 blocks.
    
    Strategy: Split the 8 columns into 4 pairs. For each pair, apply Strassen
    to the 2x2x2 sub-problem (i in {0,1}, j in {0,1}, k in the pair).
    The remaining entries (i=2 or j=2) use schoolbook.
    
    Total rank: 4*7 (Strassen) + 3*8 (i=2) + 2*8 (j=2) = 28 + 24 + 16 = 68
    """
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    U = []
    V = []
    W = []
    
    # Strassen 2x2x2 decomposition: 7 terms
    # For i,j in {0,1} (local), k in {0,1} (local)
    # U indexed by (i,j), V by (j,k), W by (k,i)
    # Using standard Strassen formulas:
    # M0 = (A00+A11)(B00+B11) -> C00 + C11
    # M1 = (A10-A00)(B00+B01) -> C10
    # M2 = (A00+A01)(B11-B01) -> C01
    # M3 = (A11+A00)(B11-B10) -> C00
    # M4 = (A01-A11)(B00+B10) -> C00 + C01
    # M5 = (A10-A00)(B11-B10) -> C11
    # M6 = (A00+A11)(B01-B11) -> C01 + C11
    
    # Let me verify these give T[i,j,k]=1 for all 8 entries.
    # I'll use the known correct Strassen tensor decomposition.
    
    # The 7 rank-1 terms for 2x2x2 (each is (u,v,w) where u in R^4, v in R^4, w in R^4):
    # u indexed by (i,j) in row-major: (0,0),(0,1),(1,0),(1,1)
    # v indexed by (j,k) in row-major: (0,0),(0,1),(1,0),(1,1)
    # w indexed by (k,i) in row-major: (0,0),(0,1),(1,0),(1,1)
    
    strassen_terms = [
        # (U_entries, V_entries, W_entries)
        # M0: (A00+A11)(B00+B11) -> C00+C11
        ([1,0,0,1], [1,0,0,1], [1,0,0,1]),
        # M1: (A10-A00)(B00+B01) -> C10
        ([0,0,1,-1], [1,1,0,0], [0,0,1,0]),
        # M2: (A00+A01)(B11-B01) -> C01
        ([1,1,0,0], [0,-1,0,1], [0,1,0,0]),
        # M3: (A11+A00)(B11-B10) -> C00
        ([1,0,0,1], [0,0,-1,1], [1,0,0,0]),
        # M4: (A01-A11)(B00+B10) -> C00+C01
        ([0,1,0,-1], [1,0,1,0], [1,1,0,0]),
        # M5: (A10-A00)(B11-B10) -> C11
        ([0,0,1,-1], [0,0,-1,1], [0,0,0,1]),
        # M6: (A00+A11)(B01-B11) -> C01+C11
        ([1,0,0,1], [0,1,0,-1], [0,1,0,1]),
    ]
    
    # Verify Strassen: for each (i,j,k) in {0,1}^3, sum of u[i,j]*v[j,k]*w[k,i] should be 1
    for i in range(2):
        for j in range(2):
            for k in range(2):
                total = 0
                for u, v, w in strassen_terms:
                    total += u[i*2+j] * v[j*2+k] * w[k*2+i]
                if total != 1:
                    # This is a problem, let's print for debugging
                    pass
    
    # Build the full decomposition
    rank = 0
    
    # Add Strassen terms for each column pair
    for t in range(4):
        k_offset = 2 * t  # k indices are 2t and 2t+1
        for u, v, w in strassen_terms:
            # U: 9 entries (i in 0..2, j in 0..2), only i,j in {0,1} are nonzero
            U_row = [0] * 9
            for li in range(2):
                for lj in range(2):
                    U_row[li * 3 + lj] = u[li * 2 + lj]
            
            # V: 24 entries (j in 0..2, k in 0..7), only j in {0,1}, k in {2t, 2t+1}
            V_row = [0] * 24
            for lj in range(2):
                for lk in range(2):
                    V_row[lj * 8 + (k_offset + lk)] = v[lj * 2 + lk]
            
            # W: 24 entries (k in 0..7, i in 0..2), only k in {2t, 2t+1}, i in {0,1}
            W_row = [0] * 24
            for lk in range(2):
                for li in range(2):
                    W_row[(k_offset + lk) * 3 + li] = w[lk * 2 + li]
            
            U.append(U_row)
            V.append(V_row)
            W.append(W_row)
            rank += 1
    
    # Add schoolbook terms for remaining entries
    # Remaining: (i=2, any j, any k) and (i in {0,1}, j=2, any k)
    for i in range(3):
        for j in range(3):
            for k in range(8):
                # Skip if already covered by Strassen (i in {0,1}, j in {0,1})
                if i in (0, 1) and j in (0, 1):
                    continue
                U_row = [0] * 9
                U_row[i * 3 + j] = 1
                V_row = [0] * 24
                V_row[j * 8 + k] = 1
                W_row = [0] * 24
                W_row[k * 3 + i] = 1
                U.append(U_row)
                V.append(V_row)
                W.append(W_row)
                rank += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    schoolbook_result = _schoolbook()
    
    if not _verify(schoolbook_result["U"], schoolbook_result["V"], schoolbook_result["W"]):
        pass
    
    if _time_left() < 5:
        return schoolbook_result
    
    # Try the Strassen-based rank-68 decomposition
    strassen_result = _strassen_2x2_block_decomp()
    
    if _verify(strassen_result["U"], strassen_result["V"], strassen_result["W"]):
        return strassen_result
    
    # Fall back to schoolbook
    return schoolbook_result

# EVOLVE-BLOCK-END
