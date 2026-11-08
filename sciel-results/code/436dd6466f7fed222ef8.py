import numpy as np
import time

_START_TIME = time.time()
DEADLINE_SECONDS = 130

def _elapsed():
    return time.time() - _START_TIME

def _time_remaining():
    return DEADLINE_SECONDS - _elapsed()

def construct():
    """Find a rank-minimal exact tensor decomposition for 2x4 * 4x7 matrix multiplication."""
    n, m, p = 2, 4, 7
    nA = n * m   # 8
    nB = m * p   # 28
    nW = p * n   # 14
    
    # Decomposition strategy:
    # Split j into 2 blocks of size 2: J0={0,1}, J1={2,3}
    # Split k into 4 blocks: K0={0,1}, K1={2,3}, K2={4,5}, K3={6}
    # For each (Jb, Kc) with |Kc|=2: Strassen 2x2x2 (rank 7)
    # For each (Jb, K3) with |K3|=1: schoolbook 2x2x1 (rank 2)
    # Total rank: 2*(7+7+7+2) = 46
    
    rank = 46
    U = [[0] * rank for _ in range(nA)]
    V = [[0] * rank for _ in range(nB)]
    W = [[0] * rank for _ in range(nW)]
    col = 0
    
    def add_strassen_222(i0, j0, k0):
        """Add a Strassen 2x2x2 block for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2] -> C[i0:i0+2, k0:k0+2]"""
        nonlocal col
        # A indices: (i0,j0), (i0,j0+1), (i0+1,j0), (i0+1,j0+1)
        a00 = i0 * m + j0
        a01 = i0 * m + (j0 + 1)
        a10 = (i0 + 1) * m + j0
        a11 = (i0 + 1) * m + (j0 + 1)
        # B indices: (j0,k0), (j0,k0+1), (j0+1,k0), (j0+1,k0+1)
        b00 = j0 * p + k0
        b01 = j0 * p + (k0 + 1)
        b10 = (j0 + 1) * p + k0
        b11 = (j0 + 1) * p + (k0 + 1)
        # W indices: C[k*n+i], so (k0,i0), (k0,i0+1), (k0+1,i0), (k0+1,i0+1)
        w00 = k0 * n + i0
        w01 = k0 * n + (i0 + 1)
        w10 = (k0 + 1) * n + i0
        w11 = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[a00][col] += 1; U[a11][col] += 1
        V[b00][col] += 1; V[b11][col] += 1
        W[w00][col] += 1; W[w11][col] += 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[a10][col] += 1; U[a11][col] += 1
        V[b00][col] += 1
        W[w10][col] += 1; W[w11][col] -= 1
        col += 1
        
        # M3 = A00(B10-B00)
        U[a00][col] += 1
        V[b10][col] += 1; V[b00][col] -= 1
        W[w01][col] += 1; W[w11][col] += 1
        col += 1
        
        # M4 = A11(B01-B10)
        U[a11][col] += 1
        V[b01][col] += 1; V[b10][col] -= 1
        W[w00][col] += 1; W[w10][col] += 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[a00][col] += 1; U[a01][col] += 1
        V[b11][col] += 1
        W[w00][col] -= 1; W[w01][col] += 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[a10][col] += 1; U[a00][col] -= 1
        V[b00][col] += 1; V[b01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[a01][col] += 1; U[a11][col] -= 1
        V[b10][col] += 1; V[b11][col] += 1
        W[w00][col] += 1
        col += 1
    
    def add_schoolbook_221(i0, j0, k0):
        """Add schoolbook for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1] -> C[i0:i0+2, k0:k0+1]"""
        nonlocal col
        for i in range(2):
            for j in range(2):
                ai = (i0 + i) * m + (j0 + j)
                bi = (j0 + j) * p + k0
                wi = k0 * n + (i0 + i)
                U[ai][col] = 1
                V[bi][col] = 1
                W[wi][col] = 1
                col += 1
    
    # Process each j-block and k-block
    j_blocks = [0, 2]  # j0 values for blocks {0,1} and {2,3}
    k_blocks_2 = [0, 2, 4]  # k0 values for blocks of size 2: {0,1}, {2,3}, {4,5}
    k_block_1 = 6  # k0 value for block of size 1: {6}
    
    for i0 in [0]:  # Only one i-block since n=2
        for j0 in j_blocks:
            for k0 in k_blocks_2:
                add_strassen_222(i0, j0, k0)
            # Handle the size-1 k block
            add_schoolbook_221(i0, j0, k_block_1)
    
    # Trim to actual used rank
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    # Verify
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    r = U_np.shape[1]
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                ai = i * m + j
                bi = j * p + k
                wi = k * n + i
                for l in range(r):
                    val += U_np[ai][l] * V_np[bi][l] * W_np[wi][l]
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if valid:
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook
    U = [[0] * (nA * nW) for _ in range(nA)]
    V = [[0] * (nA * nW) for _ in range(nB)]
    W = [[0] * (nA * nW) for _ in range(nW)]
    col = 0
    for j in range(m):
        for i in range(n):
            for k in range(p):
                ai = i * m + j
                bi = j * p + k
                wi = k * n + i
                U[ai][col] = 1
                V[bi][col] = 1
                W[wi][col] = 1
                col += 1
    U = [row[:col] for row in U]
    V = [row[:col] for row in V]
    W = [row[:col] for row in W]
    
    return {"U": U, "V": V, "W": W}
