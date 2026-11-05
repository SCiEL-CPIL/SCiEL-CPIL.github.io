import time
import numpy as np

DIMENSIONS = [4, 5, 5]

def _deadline_guard(start_time, limit=145.0):
    return (time.time() - start_time) > limit

def _schoolbook(n=4, m=5, p=5):
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
    return U, V, W

def _strassen_2x2():
    """Strassen 2x2x2: rank 7. Returns U,V,W each (4,7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    # Term 0: (a00+a11)(b00+b11)
    U[0][0]=1; U[3][0]=1; V[0][0]=1; V[3][0]=1; W[0][0]=1; W[3][0]=1
    # Term 1: (a10+a11)b00
    U[2][1]=1; U[3][1]=1; V[0][1]=1; W[2][1]=1; W[3][1]=-1
    # Term 2: a00(b01-b11)
    U[0][2]=1; V[1][2]=1; V[3][2]=-1; W[1][2]=1; W[3][2]=1
    # Term 3: a11(b10-b00)
    U[3][3]=1; V[2][3]=1; V[0][3]=-1; W[0][3]=1; W[2][3]=1
    # Term 4: (a00+a01)b11
    U[0][4]=1; U[1][4]=1; V[3][4]=1; W[0][4]=-1; W[1][4]=1
    # Term 5: (a10-a00)(b00+b01)
    U[0][5]=-1; U[2][5]=1; V[0][5]=1; V[1][5]=1; W[3][5]=1
    # Term 6: (a01-a11)(b10+b11)
    U[1][6]=1; U[3][6]=-1; V[2][6]=1; V[3][6]=1; W[0][6]=1
    return U, V, W

def _strassen_4x4():
    """
    4x4x4 Strassen via 2x2 blocks. Rank 7*7=49.
    U: (16, 49), V: (16, 49), W: (16, 49)
    """
    U = [[0]*49 for _ in range(16)]
    V = [[0]*49 for _ in range(16)]
    W = [[0]*49 for _ in range(16)]
    su, sv, sw = _strassen_2x2()
    
    for t in range(7):
        for k in range(7):
            col = t * 7 + k
            # A-blocks: which 2x2 blocks of A are involved in outer term t
            # Outer term t involves specific A-blocks with coefficients
            # For Strassen, outer term t uses A-blocks (0,0),(1,1) etc.
            # Let me define the outer term structure properly.
            
            # The 7 outer terms for 4x4x4 Strassen:
            # Each outer term is a product of linear combos of 2x2 A-blocks and B-blocks
            # giving 2x2 C-blocks, then each 2x2 product uses 7 inner Strassen terms.
            
            # Outer term definitions (A_block_coeffs, B_block_coeffs, C_block_coeffs):
            # Block indices: (0,0),(0,1),(1,0),(1,1)
            outer = [
                # T0: (A00+A11)(B00+B11) -> C00+C11
                ({(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}, {(0,0):1,(1,1):1}),
                # T1: (A10+A11)B00 -> C10-C11
                ({(1,0):1,(1,1):1}, {(0,0):1}, {(1,0):1,(1,1):-1}),
                # T2: A00(B01-B11) -> C01+C11
                ({(0,0):1}, {(0,1):1,(1,1):-1}, {(0,1):1,(1,1):1}),
                # T3: A11(B10-B00) -> C00+C10
                ({(1,1):1}, {(1,0):1,(0,0):-1}, {(0,0):1,(1,0):1}),
                # T4: (A00+A01)B11 -> C00-C01
                ({(0,0):1,(0,1):1}, {(1,1):1}, {(0,0):-1,(0,1):1}),
                # T5: (A10-A00)(B00+B01) -> C11
                ({(1,0):1,(0,0):-1}, {(0,0):1,(0,1):1}, {(1,1):1}),
                # T6: (A01-A11)(B10+B11) -> C00
                ({(0,1):1,(1,1):-1}, {(1,0):1,(1,1):1}, {(0,0):1}),
            ]
            
            ac, bc, cc = outer[t]
            
            for (ar, ac_col), acoeff in ac.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s = su[li][k]
                        if s != 0:
                            gi = 2*ar + lr
                            gj = 2*ac_col + lc
                            U[gi*4 + gj][col] += acoeff * s
            
            for (br, bc_col), bcoeff in bc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s = sv[li][k]
                        if s != 0:
                            gj = 2*br + lr
                            gk = 2*bc_col + lc
                            V[gj*4 + gk][col] += bcoeff * s
            
            for (cr, cc_col), ccoeff in cc.items():
                for lr in range(2):
                    for lc in range(2):
                        li = lr * 2 + lc
                        s = sw[li][k]
                        if s != 0:
                            gk = 2*cr + lr
                            gi = 2*cc_col + lc
                            W[gk*4 + gi][col] += ccoeff * s
    
    return U, V, W

def _verify(U, V, W, n=4, m=5, p=5):
    rank = len(U[0]) if U else 0
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    U_r = U_np.reshape(n, m, rank)
    V_r = V_np.reshape(m, p, rank)
    W_r = W_np.reshape(p, n, rank)
    T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    return bool(np.allclose(T, 1.0, atol=1e-9))

def _construct_4x5x5():
    """
    Decompose 4x5x5 as:
    - 4x4x4 part (i,j,k in 0..3): Strassen rank 49
    - 4x1x4 part (i in 0..3, j=4, k in 0..3): schoolbook rank 16
    - 4x4x1 part (i in 0..3, j in 0..3, k=4): schoolbook rank 16
    - 4x1x1 part (i in 0..3, j=4, k=4): schoolbook rank 4
    Total: 49 + 16 + 16 + 4 = 85
    """
    n, m, p = 4, 5, 5
    rank = 85
    U = [[0]*rank for _ in range(n*m)]  # (20, 85)
    V = [[0]*rank for _ in range(m*p)]  # (25, 85)
    W = [[0]*rank for _ in range(p*n)]  # (20, 85)
    
    # Part 1: 4x4x4 Strassen (i,j,k in 0..3)
    U4, V4, W4 = _strassen_4x4()
    for i in range(4):
        for j in range(4):
            for c in range(49):
                U[i*5 + j][c] = U4[i*4 + j][c]
    for j in range(4):
        for k in range(4):
            for c in range(49):
                V[j*5 + k][c] = V4[j*4 + k][c]
    for k in range(4):
        for i in range(4):
            for c in range(49):
                W[k*4 + i][c] = W4[k*4 + i][c]
    
    col = 49
    
    # Part 2: j=4, k in 0..3 (4x1x4)
    for i in range(4):
        for k in range(4):
            U[i*5 + 4][col] = 1
            V[4*5 + k][col] = 1
            W[k*4 + i][col] = 1
            col += 1
    
    # Part 3: j in 0..3, k=4 (4x4x1)
    for i in range(4):
        for j in range(4):
            U[i*5 + j][col] = 1
            V[j*5 + 4][col] = 1
            W[4*4 + i][col] = 1
            col += 1
    
    # Part 4: j=4, k=4 (4x1x1)
    for i in range(4):
        U[i*5 + 4][col] = 1
        V[4*5 + 4][col] = 1
        W[4*4 + i][col] = 1
        col += 1
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try the rank-85 decomposition
    try:
        U, V, W = _construct_4x5x5()
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    except Exception:
        pass
    
    # Fallback to schoolbook rank 100
    U, V, W = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
