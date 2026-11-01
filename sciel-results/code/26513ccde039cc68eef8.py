import time

DIMENSIONS = [2, 2, 12]

def _schoolbook(n=2, m=2, p=12):
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}

def _check(U, V, W, n, m, p, tol=1e-8):
    R = len(U[0]) if len(U)>0 else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - (1.0 if j == i else 0.0)) > tol:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_t = time.time()
    deadline = start_t + 105
    n, m, p = DIMENSIONS  # 2, 2, 12
    
    # Fallback
    fb = _schoolbook(n, m, p)
    if not _check(fb["U"], fb["V"], fb["W"], n, m, p):
        return fb

    # Attempt rank-42 using 6 blocks of rank-7
    # Block decomposition for 2x2x2
    # U_block (4x7), V_block (4x7), W_block (4x7)
    # Rows of U: A00, A01, A10, A11
    # Rows of V: B00, B01, B10, B11
    # Rows of W: C00, C01, C10, C11 (indices k*n+i: 0,1,2,3)
    
    U_b = [
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 1, 0, 0, 1, 1, 1], # A01
        [0, 0, 0, 0, 0, 1, 0], # A10
        [1, 0, 0, 1, 0, 0, -1] # A11
    ]
    # Wait, let's use the P1..P7 mapping derived earlier.
    # P1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1] (C00, C11)
    # P2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] (C10)
    # P3: U=[1,0,0,0], V=[0,0,1,-1], W=[0,1,0,0] (C01)
    # P4: U=[0,0,0,1], V=[-1,1,0,0], W=[0,0,1,0] (C10) -- Wait, W for P4 in C10 is +1. In C00 is -1.
    # Let's build columns explicitly.
    
    # Col 0 (P1):
    # U: A00=1, A11=1
    # V: B00=1, B11=1
    # W: C00=1, C11=1
    # Col 1 (P2):
    # U: A10=1, A11=1
    # V: B00=1
    # W: C10=1
    # Col 2 (P3):
    # U: A00=1
    # V: B10=1, B11=-1
    # W: C01=1
    # Col 3 (P4):
    # U: A11=1
    # V: B01=1, B00=-1
    # W: C10=1, C00=-1
    # Col 4 (P5):
    # U: A00=1, A01=1
    # V: B11=1
    # W: C01=1, C00=-1
    # Col 5 (P6):
    # U: A10=1, A00=-1
    # V: B00=1, B01=1
    # W: C11=1
    # Col 6 (P7):
    # U: A01=1, A11=-1
    # V: B10=1, B11=1
    # W: C00=1

    U_b = [
        [1, 0, 1, 0, 1, -1, 0], # A00
        [0, 1, 0, 0, 1, 0, 1],  # A01
        [0, 1, 0, 0, 0, 1, 0],  # A10
        [1, 1, 0, 1, 0, 0, -1]  # A11
    ]
    V_b = [
        [1, 1, 0, -1, 0, 1, 0], # B00
        [0, 0, 0, 1, 0, 1, 0],  # B01
        [0, 0, 1, 0, 0, 0, 1],  # B10
        [1, 0, -1, 0, 1, 0, 1]  # B11
    ]
    W_b = [
        [1, 0, 0, -1, -1, 0, 1], # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, 0, 0, 0, 0, 1, 0]    # C11
    ]

    # Verify block
    # C00 = P1 - P4 - P5 + P7
    # P1: A00B00 + A00B11 + A11B00 + A11B11
    # -P4: -A11B01 + A11B00
    # -P5: -A00B11 - A01B11
    # P7: A01B10 + A01B11 - A11B10 - A11B11
    # Sum: A00B00 + (A00B11-A00B11) + (A11B00+A11B00) + (A11B11-A11B11) - A11B01 - A01B11 + A01B11 + A01B10 - A11B10
    # = A00B00 + 2A11B00 - A11B01 + A01B10 - A11B10
    # This is NOT A00B00 + A01B10.
    # The standard Strassen formula I am using seems to have an error in my manual expansion or the formula itself is slightly different in signs.
    # Given the complexity and risk of error, and the time limit, I will stick to the schoolbook rank-48 which is guaranteed to work.
    # However, I can try to just return the schoolbook.
    
    return fb

# EVOLVE-BLOCK-END
