import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify that U, V, W form a valid decomposition."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                u_idx = i * m + k
                v_idx = k * p + j
                w_idx = j * n + i
                val = 0
                for c in range(rank):
                    val += U[u_idx][c] * V[v_idx][c] * W[w_idx][c]
                if abs(val - 1) > tol:
                    return False
    return True

def _kron_decomp_dims(U1, V1, W1, n1, m1, p1, U2, V2, W2, n2, m2, p2):
    """Kronecker product of two decompositions with explicit dimensions."""
    r1 = len(U1[0])
    r2 = len(U2[0])
    n, m, p = n1 * n2, m1 * m2, p1 * p2
    r = r1 * r2
    
    U = [[0] * r for _ in range(n * m)]
    V = [[0] * r for _ in range(m * p)]
    W = [[0] * r for _ in range(p * n)]
    
    for i1 in range(n1):
        for k1 in range(m1):
            for i2 in range(n2):
                for k2 in range(m2):
                    u_idx = (i1 * n2 + i2) * (m1 * m2) + (k1 * m2 + k2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            U[u_idx][c] = U1[i1 * m1 + k1][c1] * U2[i2 * m2 + k2][c2]
    
    for k1 in range(m1):
        for j1 in range(p1):
            for k2 in range(m2):
                for j2 in range(p2):
                    v_idx = (k1 * m2 + k2) * (p1 * p2) + (j1 * p2 + j2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            V[v_idx][c] = V1[k1 * p1 + j1][c1] * V2[k2 * p2 + j2][c2]
    
    for j1 in range(p1):
        for i1 in range(n1):
            for j2 in range(p2):
                for i2 in range(n2):
                    w_idx = (j1 * p2 + j2) * (n1 * n2) + (i1 * n2 + i2)
                    for c1 in range(r1):
                        for c2 in range(r2):
                            c = c1 * r2 + c2
                            W[w_idx][c] = W1[j1 * n1 + i1][c1] * W2[j2 * n2 + i2][c2]
    
    return {"U": U, "V": V, "W": W}

def _strassen_2x2x2():
    """Strassen's rank-7 decomposition for 2x2x2 matrix multiplication."""
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U indices: (i,k) -> i*m+k: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # V indices: (k,j) -> k*p+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # W indices: (j,i) -> j*n+i: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    
    # m1: (a00+a11)(b00+b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2: (a10+a11)b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # m3: a00(b01-b11)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # m4: a11(b10-b00)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1
    
    # m5: (a00+a10)b11
    U[0][4] = 1; U[2][4] = 1
    V[3][4] = 1
    W[2][4] = 1; W[0][4] = -1
    
    # m6: (a01-a10)(b00+b10)
    U[1][5] = 1; U[2][5] = -1
    V[0][5] = 1; V[2][5] = 1
    W[3][5] = 1
    
    # m7: (a01-a11)(b01+b11)
    U[1][6] = 1; U[3][6] = -1
    V[1][6] = 1; V[3][6] = 1
    W[1][6] = 1; W[0][6] = -1
    
    return {"U": U, "V": V, "W": W}

def _schoolbook(n, m, p):
    """Schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                U[i * m + k][col] = 1
                V[k * p + j][col] = 1
                W[j * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_4x4x4():
    """Strassen-based rank-49 decomposition for 4x4x4 matrix multiplication.
    Uses Kronecker product of two 2x2x2 Strassen decompositions."""
    d1 = _strassen_2x2x2()
    d2 = _strassen_2x2x2()
    return _kron_decomp_dims(d1["U"], d1["V"], d1["W"], 2, 2, 2,
                             d2["U"], d2["V"], d2["W"], 2, 2, 2)

def _strassen_4x4x9():
    """Decomposition for 4x4x9 using Strassen for the 4x4 part and schoolbook for the 3-column part.
    
    Strategy: 
    - Split the 4x4x9 problem using the structure 4=2*2, 9=3*3
    - Use Kronecker product of (Strassen 2x2x2) x (schoolbook 2x2x3)
    - This gives rank 7 * 12 = 84
    
    Alternatively, try to use a better 2x2x3 decomposition.
    """
    # Option 1: Kronecker of Strassen 2x2x2 (rank 7) and schoolbook 2x2x3 (rank 12)
    d1 = _strassen_2x2x2()
    d2 = _schoolbook(2, 2, 3)
    result = _kron_decomp_dims(d1["U"], d1["V"], d1["W"], 2, 2, 2,
                               d2["U"], d2["V"], d2["W"], 2, 2, 3)
    return result

def _slice_packing_4x4x9():
    """Slice-packing strategy: partition the 9-column dimension into three 3-column blocks.
    
    For each 3-column block, we need a 4x4x3 decomposition.
    We use Strassen on the 4x4 part (rank 49 for 4x4x4) but adapted for 4x4x3.
    
    Actually, let's think about this differently:
    4x4x9 = three independent 4x4x3 subproblems (one for each group of 3 columns).
    
    For 4x4x3: split 4=2*2, 3=1*3 or 3=3*1.
    Using Kronecker: 4x4x3 = (2x2x1) x (2x2x3)
    - 2x2x1 schoolbook: rank 4
    - 2x2x3 schoolbook: rank 12
    - Total: 4*12 = 48 per slice, 3 slices = 144 (same as schoolbook)
    
    Better: 4x4x3 = (2x2x3) x (2x2x1)
    Same result.
    
    Or: use Strassen 2x2x2 (rank 7) for the 4x4 part, and handle the 3 columns separately.
    4x4x3: split as (2x2x2) x (2x2x1) doesn't work because 2*1 != 3.
    
    Let me try: 4x4x9 directly via Kronecker (2x2x3) x (2x2x3)
    With Strassen for 2x2x2 part and schoolbook for the extra column:
    Actually, 2x2x3 can be seen as 2x2x2 + 2x2x1
    Rank of 2x2x2 with Strassen = 7, rank of 2x2x1 schoolbook = 4
    But these don't combine via Kronecker easily.
    
    Best simple approach: Kronecker of Strassen 2x2x2 (rank 7) and schoolbook 2x2x3 (rank 12)
    = rank 84 for 4x4x9.
    """
    d1 = _strassen_2x2x2()
    d2 = _schoolbook(2, 2, 3)
    return _kron_decomp_dims(d1["U"], d1["V"], d1["W"], 2, 2, 2,
                             d2["U"], d2["V"], d2["W"], 2, 2, 3)

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    # Try multiple strategies and pick the best valid one
    
    # Strategy 1: Kronecker of Strassen 2x2x2 (rank 7) and schoolbook 2x2x3 (rank 12)
    # Total rank: 84
    try:
        result1 = _strassen_4x4x9()
        if _verify(result1["U"], result1["V"], result1["W"], n, m, p):
            if time.time() > deadline:
                return result1
            rank1 = len(result1["U"][0])
        else:
            rank1 = float('inf')
    except Exception:
        rank1 = float('inf')
    
    # Strategy 2: Schoolbook (rank 144)
    result2 = _schoolbook(n, m, p)
    if _verify(result2["U"], result2["V"], result2["W"], n, m, p):
        rank2 = len(result2["U"][0])
    else:
        rank2 = float('inf')
    
    # Strategy 3: Kronecker of schoolbook 2x2x3 and schoolbook 2x2x3 (rank 144)
    d1 = _schoolbook(2, 2, 3)
    d2 = _schoolbook(2, 2, 3)
    result3 = _kron_decomp_dims(d1["U"], d1["V"], d1["W"], 2, 2, 3,
                                d2["U"], d2["V"], d2["W"], 2, 2, 3)
    if _verify(result3["U"], result3["V"], result3["W"], n, m, p):
        rank3 = len(result3["U"][0])
    else:
        rank3 = float('inf')
    
    # Pick the best
    candidates = []
    if rank1 != float('inf'):
        candidates.append((rank1, result1))
    if rank2 != float('inf'):
        candidates.append((rank2, result2))
    if rank3 != float('inf'):
        candidates.append((rank3, result3))
    
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
    else:
        # Fallback: return schoolbook even if verification failed (shouldn't happen)
        return result2
# EVOLVE-BLOCK-END
