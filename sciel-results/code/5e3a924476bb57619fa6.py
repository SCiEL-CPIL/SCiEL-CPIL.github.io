"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

def _strassen_222():
    """Return Strassen rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U (4x7), V (4x7), W (4x7) for the 2x2x2 tensor.
    Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    """
    # Strassen's 7 products for 2x2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # U indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # V indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # W indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # M1 = (A11+A22)(B11+B22)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # M2 = (A21+A22)B11
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = 1
    
    # M3 = A11(B12-B22)
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # M4 = A22(B21-B11)
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[2][3] = 1; W[3][3] = 1
    
    # M5 = (A11+A12)B22
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[3][4] = 1
    
    # M6 = (A21-A11)(B11+B12)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # M7 = (A12-A22)(B21+B22)
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1; W[3][6] = 1
    
    return U, V, W

def _verify_decomposition(U, V, W, n, m, p, num_trials=5):
    """Verify decomposition by checking random matrix products."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = len(U[0]) if U else 0
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for _ in range(num_trials):
        A = np.random.randint(-3, 4, size=(n, m)).astype(float)
        B = np.random.randint(-3, 4, size=(m, p)).astype(float)
        C_expected = A @ B
        
        C_computed = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                total = 0.0
                for j in range(m):
                    total += np.dot(U_arr[i*m+j], V_arr[j*p+k] * W_arr[k*n+i])
                C_computed[i, k] = total
        
        if not np.allclose(C_computed, C_expected, atol=1e-9):
            return False
    return True

def _build_recursive(n, m, p, deadline):
    """Build decomposition for n x m x p matrix multiplication recursively.
    Returns (U, V, W, rank) where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    """
    if time.time() > deadline:
        return None
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Base cases
    if n == 1 and m == 1 and p == 1:
        U = [[1]]
        V = [[1]]
        W = [[1]]
        return U, V, W, 1
    
    if n == 1:
        # 1 x m x p: schoolbook rank = m*p
        rank = m * p
        U = [[0]*rank for _ in range(m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p)]
        col = 0
        for j in range(m):
            for k in range(p):
                U[j][col] = 1
                V[j*p+k][col] = 1
                W[k][col] = 1
                col += 1
        return U, V, W, rank
    
    if m == 1:
        # n x 1 x p: schoolbook rank = n*p
        rank = n * p
        U = [[0]*rank for _ in range(n)]
        V = [[0]*rank for _ in range(p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k*n+i][col] = 1
                col += 1
        return U, V, W, rank
    
    if p == 1:
        # n x m x 1: schoolbook rank = n*m
        rank = n * m
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m)]
        W = [[0]*rank for _ in range(n)]
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W, rank
    
    # Check if we can use Strassen for 2x2x2
    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_222()
        return U, V, W, 7
    
    # Try to split into sub-problems
    # Strategy: split the middle dimension m into m1 + m2
    # C = A1*B1 + A2*B2 where A1 is n x m1, B1 is m1 x p, A2 is n x m2, B2 is m2 x p
    
    best_result = None
    best_rank = nm * p  # schoolbook
    
    # Try all splits of m into m1 + m2
    for m1 in range(1, m):
        m2 = m - m1
        if time.time() > deadline:
            break
        
        # Get decompositions for sub-problems
        res1 = _build_recursive(n, m1, p, deadline)
        if res1 is None:
            continue
        U1, V1, W1, r1 = res1
        
        res2 = _build_recursive(n, m2, p, deadline)
        if res2 is None:
            continue
        U2, V2, W2, r2 = res2
        
        total_rank = r1 + r2
        if total_rank >= best_rank:
            continue
        
        # Combine: U has rows for A1 (n*m1) then A2 (n*m2)
        # V has rows for B1 (m1*p) then B2 (m2*p)
        # W is same for both (p*n)
        
        # U: (n*m1 + n*m2) x total_rank
        U = [row[:r1] + [0]*r2 for row in U1]
        U += [[0]*r1 + row for row in U2]
        
        # V: (m1*p + m2*p) x total_rank
        V = [row[:r1] + [0]*r2 for row in V1]
        V += [[0]*r1 + row for row in V2]
        
        # W: (p*n) x total_rank
        W = [row[:r1] + [0]*r2 for row in W1]
        for i in range(len(W)):
            W[i] = W[i][:r1] + W2[i]
        
        best_result = (U, V, W)
        best_rank = total_rank
    
    # Also try splitting n into n1 + n2
    for n1 in range(1, n):
        n2 = n - n1
        if time.time() > deadline:
            break
        
        res1 = _build_recursive(n1, m, p, deadline)
        if res1 is None:
            continue
        U1, V1, W1, r1 = res1
        
        res2 = _build_recursive(n2, m, p, deadline)
        if res2 is None:
            continue
        U2, V2, W2, r2 = res2
        
        total_rank = r1 + r2
        if total_rank >= best_rank:
            continue
        
        # U: (n1*m + n2*m) x total_rank
        U = [row[:r1] + [0]*r2 for row in U1]
        U += [[0]*r1 + row for row in U2]
        
        # V: (m*p) x total_rank
        V = [row[:r1] + [0]*r2 for row in V1]
        for i in range(len(V)):
            V[i] = V[i][:r1] + V2[i]
        
        # W: (p*n1 + p*n2) x total_rank
        W = [row[:r1] + [0]*r2 for row in W1]
        W += [[0]*r1 + row for row in W2]
        
        best_result = (U, V, W)
        best_rank = total_rank
    
    # Also try splitting p into p1 + p2
    for p1 in range(1, p):
        p2 = p - p1
        if time.time() > deadline:
            break
        
        res1 = _build_recursive(n, m, p1, deadline)
        if res1 is None:
            continue
        U1, V1, W1, r1 = res1
        
        res2 = _build_recursive(n, m, p2, deadline)
        if res2 is None:
            continue
        U2, V2, W2, r2 = res2
        
        total_rank = r1 + r2
        if total_rank >= best_rank:
            continue
        
        # U: (n*m) x total_rank
        U = [row[:r1] + [0]*r2 for row in U1]
        for i in range(len(U)):
            U[i] = U[i][:r1] + U2[i]
        
        # V: (m*p1 + m*p2) x total_rank
        V = [row[:r1] + [0]*r2 for row in V1]
        V += [[0]*r1 + row for row in V2]
        
        # W: (p1*n + p2*n) x total_rank
        W = [row[:r1] + [0]*r2 for row in W1]
        W += [[0]*r1 + row for row in W2]
        
        best_result = (U, V, W)
        best_rank = total_rank
    
    if best_result is not None:
        return best_result[0], best_result[1], best_result[2], best_rank
    
    # Fallback: schoolbook
    rank = nm * p
    U = [[0]*rank for _ in range(nm)]
    V = [[0]*rank for _ in range(mp)]
    W = [[0]*rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W, rank

def _schoolbook():
    n, m, p = DIMENSIONS
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
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s kill
    
    schoolbook = _schoolbook()
    
    try:
        result = _build_recursive(n, m, p, deadline)
        if result is not None:
            U, V, W, rank = result
            # Verify before returning
            if _verify_decomposition(U, V, W, n, m, p, num_trials=3):
                return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    return schoolbook
# EVOLVE-BLOCK-END
