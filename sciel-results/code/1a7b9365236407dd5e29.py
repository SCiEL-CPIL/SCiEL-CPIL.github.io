"""Matrix multiplication 2,4,6 rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import sys
import random

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # 10s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _strassen_2x2x2_terms():
    """
    Return the 7 Strassen terms for 2x2 matrix multiplication.
    Each term is (U, V, W) where:
    U is a 4-vector for A = [[a,b],[c,d]] -> [a,b,c,d]
    V is a 4-vector for B = [[e,f],[g,h]] -> [e,f,g,h]
    W is a 4-vector for C = [[C00,C01],[C10,C11]] -> [C00,C01,C10,C11]
    
    Standard Strassen:
    P1 = (a+d)(e+h)
    P2 = (c+d)(e)
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)(h)
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C00 = P1 - P4 + P6 - P7
    C01 = P3 + P5
    C10 = P2 + P4
    C11 = P1 - P3 + P5 - P6
    """
    terms = []
    # P1 = (a+d)(e+h) -> contributes to C00(+1), C11(+1)
    terms.append(([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]))
    # P2 = (c+d)(e) -> contributes to C10(+1)
    terms.append(([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]))
    # P3 = a(f-h) -> contributes to C01(+1), C11(-1)
    terms.append(([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, -1]))
    # P4 = d(g-e) -> contributes to C00(-1), C10(+1)
    terms.append(([0, 0, 0, 1], [0, 0, 1, -1], [-1, 0, 1, 0]))
    # P5 = (a+b)(h) -> contributes to C01(+1), C11(+1)
    terms.append(([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1]))
    # P6 = (c-a)(e+f) -> contributes to C00(+1), C11(-1)
    terms.append(([-1, 0, 1, 0], [1, 1, 0, 0], [1, 0, 0, -1]))
    # P7 = (b-d)(g+h) -> contributes to C00(-1)
    terms.append(([0, 1, 0, -1], [0, 0, 1, 1], [-1, 0, 0, 0]))
    return terms

def _verify_strassen():
    """Verify the Strassen decomposition numerically."""
    terms = _strassen_2x2x2_terms()
    for trial in range(50):
        a, b, c, d = [random.randint(-10, 10) for _ in range(4)]
        e, f, g, h = [random.randint(-10, 10) for _ in range(4)]
        A_flat = [a, b, c, d]
        B_flat = [e, f, g, h]
        
        C00 = a*e + b*f
        C01 = a*g + b*h
        C10 = c*e + d*f
        C11 = c*g + d*h
        expected = [C00, C01, C10, C11]
        
        computed = [0, 0, 0, 0]
        for U, V, W in terms:
            u_val = sum(U[i] * A_flat[i] for i in range(4))
            v_val = sum(V[i] * B_flat[i] for i in range(4))
            for j in range(4):
                computed[j] += W[j] * u_val * v_val
        
        if computed != expected:
            return False
    return True

def _schoolbook_decomposition():
    """Fallback schoolbook decomposition of rank n*m*p."""
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

def _construct_rank42():
    """
    Construct a rank-42 decomposition for 2x4x6 matrix multiplication using
    Strassen-based blocks.
    
    We view A (2x4) as two 1x2 blocks stacked vertically: A = [A1; A2] where A1, A2 are 1x2.
    We view B (4x6) as two 2x3 blocks stacked vertically: B = [B1; B2] where B1, B2 are 2x3.
    Wait, that's not right. A is 2x4, B is 4x6.
    Let's split the middle dimension (4) into 2+2.
    A = [A_left | A_right] where A_left is 2x2, A_right is 2x2.
    B = [B_top; B_bottom] where B_top is 2x6, B_bottom is 2x6.
    C = A_left * B_top + A_right * B_bottom.
    
    Now, A_left (2x2) * B_top (2x6).
    We can split B_top into three 2x2 blocks: B_top = [B1 | B2 | B3].
    A_left * B_top = A_left * B1 + A_left * B2 + A_left * B3.
    Each A_left * Bi (2x2 * 2x2) can be done with rank 7 (Strassen).
    So A_left * B_top has rank 3 * 7 = 21.
    Similarly, A_right * B_bottom has rank 3 * 7 = 21.
    Total rank = 21 + 21 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    strassen_terms = _strassen_2x2x2_terms()
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # A is 2x4, flattened row-major: [a00, a01, a02, a03, a10, a11, a12, a13]
    # U indices: 0..7
    # A_left = [[a00, a01], [a10, a11]] -> U indices [0, 1, 4, 5]
    # A_right = [[a02, a03], [a12, a13]] -> U indices [2, 3, 6, 7]
    
    # B is 4x6, flattened row-major: [b00..b05, b10..b15, b20..b25, b30..b35]
    # V indices: 0..23
    # B_top = B[0:2, :] = [[b00..b05], [b10..b15]]
    # B_bottom = B[2:4, :] = [[b20..b25], [b30..b35]]
    
    # B_top split into 2x2 blocks:
    # B1 = B[0:2, 0:2] -> V indices [0, 1, 6, 7]
    # B2 = B[0:2, 2:4] -> V indices [2, 3, 8, 9]
    # B3 = B[0:2, 4:6] -> V indices [4, 5, 10, 11]
    
    # B_bottom split into 2x2 blocks:
    # B4 = B[2:4, 0:2] -> V indices [12, 13, 18, 19]
    # B5 = B[2:4, 2:4] -> V indices [14, 15, 20, 21]
    # B6 = B[2:4, 4:6] -> V indices [16, 17, 22, 23]
    
    # C is 2x6, flattened row-major: [c00..c05, c10..c15]
    # W indices: 0..11
    # W convention: W[k*n + i, r] where k is column, i is row.
    # C[i][k] -> W[k*2 + i]
    # C00 -> W[0], C01 -> W[1], C02 -> W[2], C03 -> W[3], C04 -> W[4], C05 -> W[5]
    # C10 -> W[6], C11 -> W[7], C12 -> W[8], C13 -> W[9], C14 -> W[10], C15 -> W[11]
    
    # For A_left * B1 (2x2 * 2x2 -> 2x2):
    # Result contributes to C[0:2, 0:2] -> W indices [0, 1, 6, 7]
    # For A_left * B2:
    # Result contributes to C[0:2, 2:4] -> W indices [2, 3, 8, 9]
    # For A_left * B3:
    # Result contributes to C[0:2, 4:6] -> W indices [4, 5, 10, 11]
    
    # For A_right * B4:
    # Result contributes to C[0:2, 0:2] -> W indices [0, 1, 6, 7]
    # For A_right * B5:
    # Result contributes to C[0:2, 2:4] -> W indices [2, 3, 8, 9]
    # For A_right * B6:
    # Result contributes to C[0:2, 4:6] -> W indices [4, 5, 10, 11]
    
    a_left_indices = [0, 1, 4, 5]
    a_right_indices = [2, 3, 6, 7]
    
    b_top_indices = [
        [0, 1, 6, 7],   # B1
        [2, 3, 8, 9],   # B2
        [4, 5, 10, 11]  # B3
    ]
    
    b_bottom_indices = [
        [12, 13, 18, 19],   # B4
        [14, 15, 20, 21],   # B5
        [16, 17, 22, 23]    # B6
    ]
    
    w_indices = [
        [0, 1, 6, 7],   # cols 0,1
        [2, 3, 8, 9],   # cols 2,3
        [4, 5, 10, 11]  # cols 4,5
    ]
    
    # Block A_left * B_top (3 blocks of 2x2)
    for b_block in range(3):
        for t in range(7):
            col = b_block * 7 + t
            U_t, V_t, W_t = strassen_terms[t]
            
            for i in range(4):
                U[a_left_indices[i]][col] = U_t[i]
            
            for i in range(4):
                V[b_top_indices[b_block][i]][col] = V_t[i]
            
            for i in range(4):
                W[w_indices[b_block][i]][col] = W_t[i]
    
    # Block A_right * B_bottom (3 blocks of 2x2)
    for b_block in range(3):
        for t in range(7):
            col = 21 + b_block * 7 + t
            U_t, V_t, W_t = strassen_terms[t]
            
            for i in range(4):
                U[a_right_indices[i]][col] = U_t[i]
            
            for i in range(4):
                V[b_bottom_indices[b_block][i]][col] = V_t[i]
            
            for i in range(4):
                W[w_indices[b_block][i]][col] = W_t[i]
    
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n, m, p, num_trials=50):
    """Verify a decomposition numerically."""
    rank = len(W[0]) if W else 0
    for trial in range(num_trials):
        A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
        
        C_exp = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        C_comp = [[0] * p for _ in range(n)]
        for col in range(rank):
            u_val = sum(U[i * m + j][col] * A[i][j] for i in range(n) for j in range(m))
            v_val = sum(V[j * p + k][col] * B[j][k] for j in range(m) for k in range(p))
            for i in range(n):
                for k in range(p):
                    C_comp[i][k] += W[k * n + i][col] * u_val * v_val
        
        if C_comp != C_exp:
            return False
    return True

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x4x6 matrix multiplication.
    Uses Strassen-based blocks for rank 42, with verification and fallback.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Verify Strassen first
    if not _verify_strassen():
        return _schoolbook_decomposition()
    
    # Build rank-42 decomposition
    result = _construct_rank42()
    
    # Verify with time check
    if _time_left() > 5.0:
        if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p, num_trials=30):
            return result
    
    # Fallback to schoolbook if verification fails or time is tight
    return _schoolbook_decomposition()
# EVOLVE-BLOCK-END
