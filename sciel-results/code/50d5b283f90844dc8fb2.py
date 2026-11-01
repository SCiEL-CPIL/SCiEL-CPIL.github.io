"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _build_multiplication_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    rank = len(U[0])
    T = _build_multiplication_tensor(n, m, p)
    for i in range(n*m):
        for j in range(m*p):
            for k in range(p*n):
                s = 0.0
                for r in range(rank):
                    s += U[i][r] * V[j][r] * W[k][r]
                if abs(s - T[i,j,k]) > tol:
                    return False
    return True

def _strassen_2x2_block(n, m, p, block_offset):
    """
    Build Strassen decomposition for 2x2 * 2x2 block.
    block_offset: which 2-column block of the 2x4 matrix (0 or 1).
    Returns (U_block, V_block, W_block) each of shape (n*m, 7), (m*p, 7), (p*n, 7)
    but only the relevant columns are non-zero.
    """
    # For 2x2 * 2x2, Strassen uses 7 multiplications:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A10)B11
    # M6 = (A01-A00)(B00+B10)
    # M7 = (A10-A11)(B01+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    #
    # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    # n=2, m=2, p=2 for the block
    # i in {0,1}, j in {0,1}, k in {0,1}
    # U indices: i*2+j -> 0,1,2,3
    # V indices: j*2+k -> 0,1,2,3
    # W indices: k*2+i -> 0,1,2,3
    
    # For the full 2x2*2x4 problem:
    # n=2, m=2, p=4
    # U indices: i*2+j -> 0,1,2,3
    # V indices: j*4+k -> 0,1,2,3,4,5,6,7
    # W indices: k*2+i -> 0,1,2,3,4,5,6,7
    
    # Block 0: k in {0,1}, V indices 0-3, W indices 0-3
    # Block 1: k in {2,3}, V indices 4-7, W indices 4-7
    
    U_block = np.zeros((4, 7))
    V_block = np.zeros((8, 7))
    W_block = np.zeros((8, 7))
    
    if block_offset == 0:
        # V rows: j*4+k where k=0,1 -> rows 0,1 (j=0), 4,5 (j=1)
        # W rows: k*2+i where k=0,1 -> rows 0,1 (k=0), 2,3 (k=1)
        v_rows = [0, 1, 4, 5]  # j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
        w_rows = [0, 2, 1, 3]  # k=0,i=0; k=1,i=0; k=0,i=1; k=1,i=1
        # Actually: W index = k*n+i = k*2+i
        # k=0,i=0 -> 0; k=0,i=1 -> 1; k=1,i=0 -> 2; k=1,i=1 -> 3
        w_rows = [0, 1, 2, 3]  # (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
    else:
        # Block 1: k in {2,3}
        # V rows: j*4+k where k=2,3 -> rows 2,3 (j=0), 6,7 (j=1)
        v_rows = [2, 3, 6, 7]
        # W rows: k*2+i where k=2,3
        # k=2,i=0 -> 4; k=2,i=1 -> 5; k=3,i=0 -> 6; k=3,i=1 -> 7
        w_rows = [4, 5, 6, 7]
    
    # Strassen factors for 2x2*2x2:
    # M1 = (A00+A11)(B00+B11): U=(1,0,0,1), V=(1,0,0,1), W=(1,0,0,1) for C00
    # Let me define the 7 terms properly.
    
    # Term 0: M1 = (A00+A11)(B00+B11)
    # U: A00=1, A11=1 -> [1,0,0,1]
    # V: B00=1, B11=1 -> [1,0,0,1]
    # W: contributes to C00(+1), C11(+1)
    U_block[0,0] = 1; U_block[3,0] = 1
    V_block[v_rows[0],0] = 1; V_block[v_rows[3],0] = 1
    W_block[w_rows[0],0] = 1  # C00
    W_block[w_rows[3],0] = 1  # C11
    
    # Term 1: M2 = (A10+A11)B00
    # U: A10=1, A11=1 -> [0,0,1,1]
    # V: B00=1 -> [1,0,0,0]
    # W: contributes to C10(+1), C11(-1)
    U_block[2,1] = 1; U_block[3,1] = 1
    V_block[v_rows[0],1] = 1
    W_block[w_rows[2],1] = 1  # C10
    W_block[w_rows[3],1] = -1  # C11
    
    # Term 2: M3 = A00(B01-B11)
    # U: A00=1 -> [1,0,0,0]
    # V: B01=1, B11=-1 -> [0,1,0,-1]
    # W: contributes to C01(+1), C11(+1)
    U_block[0,2] = 1
    V_block[v_rows[1],2] = 1; V_block[v_rows[3],2] = -1
    W_block[w_rows[1],2] = 1  # C01
    W_block[w_rows[3],2] = 1  # C11
    
    # Term 3: M4 = A11(B10-B00)
    # U: A11=1 -> [0,0,0,1]
    # V: B10=1, B00=-1 -> [0,0,1,-1]
    # W: contributes to C00(+1), C10(+1)
    U_block[3,3] = 1
    V_block[v_rows[2],3] = 1; V_block[v_rows[0],3] = -1
    W_block[w_rows[0],3] = 1  # C00
    W_block[w_rows[2],3] = 1  # C10
    
    # Term 4: M5 = (A00+A10)B11
    # U: A00=1, A10=1 -> [1,0,1,0]
    # V: B11=1 -> [0,0,0,1]
    # W: contributes to C01(+1), C00(-1)
    U_block[0,4] = 1; U_block[2,4] = 1
    V_block[v_rows[3],4] = 1
    W_block[w_rows[1],4] = 1  # C01
    W_block[w_rows[0],4] = -1  # C00
    
    # Term 5: M6 = (A01-A00)(B00+B10)
    # U: A01=1, A00=-1 -> [-1,1,0,0]
    # V: B00=1, B10=1 -> [1,0,1,0]
    # W: contributes to C11(+1)
    U_block[0,5] = -1; U_block[1,5] = 1
    V_block[v_rows[0],5] = 1; V_block[v_rows[2],5] = 1
    W_block[w_rows[3],5] = 1  # C11
    
    # Term 6: M7 = (A10-A11)(B01+B11)
    # U: A10=1, A11=-1 -> [0,0,1,-1]
    # V: B01=1, B11=1 -> [0,1,0,1]
    # W: contributes to C00(+1)
    U_block[2,6] = 1; U_block[3,6] = -1
    V_block[v_rows[1],6] = 1; V_block[v_rows[3],6] = 1
    W_block[w_rows[0],6] = 1  # C00
    
    return U_block, V_block, W_block

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 130
    
    # Use Strassen for each 2x2 block: total rank 14
    U0, V0, W0 = _strassen_2x2_block(n, m, p, 0)
    U1, V1, W1 = _strassen_2x2_block(n, m, p, 1)
    
    U = U0 + U1
    V = V0 + V1
    W = W0 + W1
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    # Verify
    if _verify_decomposition(U_list, V_list, W_list, n, m, p, tol=1e-6):
        return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
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
# EVOLVE-BLOCK-END
