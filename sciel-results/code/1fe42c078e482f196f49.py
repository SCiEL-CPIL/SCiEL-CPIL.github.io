"""Matrix multiplication 2,3,5 — rank-29 decomposition via one Strassen 2x2x2 block."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import time

def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-29 ansatz.
    
    We use one Strassen 2x2x2 block for i in {0,1}, j in {0,1}, k in {0,1}:
    - 7 Strassen terms replace 8 schoolbook terms
    - Remaining schoolbook terms: 30 - 8 = 22
    Total rank: 7 + 22 = 29
    
    Strassen identities for C = A*B where A is 2x2, B is 2x2:
    m1 = (a00+a11)*(b00+b11)
    m2 = (a10+a11)*b00
    m3 = a00*(b01-b11)
    m4 = a11*(b10-b00)
    m5 = (a00+a01)*b11
    m6 = (a10-a00)*(b00+b01)
    m7 = (a01-a11)*(b10+b11)
    c00 = m1 + m4 - m5 + m7
    c01 = m3 + m5
    c10 = m2 + m4
    c11 = m1 - m2 + m3 + m6
    """
    n, m, p = 2, 3, 5
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # === Strassen 2x2x2 block: i in {0,1}, j in {0,1}, k in {0,1} ===
    # m1 = (a00+a11)*(b00+b11)
    U[0*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m2 = (a10+a11)*b00
    U[1*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1
    W[0*2+1][c] = 1; W[1*2+1][c] = -1
    c += 1
    
    # m3 = a00*(b01-b11)
    U[0*3+0][c] = 1
    V[0*5+1][c] = 1; V[1*5+1][c] = -1
    W[1*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m4 = a11*(b10-b00)
    U[1*3+1][c] = 1
    V[1*5+0][c] = 1; V[0*5+0][c] = -1
    W[0*2+0][c] = 1; W[0*2+1][c] = 1
    c += 1
    
    # m5 = (a00+a01)*b11
    U[0*3+0][c] = 1; U[0*3+1][c] = 1
    V[1*5+1][c] = 1
    W[0*2+0][c] = -1; W[1*2+0][c] = 1
    c += 1
    
    # m6 = (a10-a00)*(b00+b01)
    U[1*3+0][c] = 1; U[0*3+0][c] = -1
    V[0*5+0][c] = 1; V[0*5+1][c] = 1
    W[1*2+1][c] = 1
    c += 1
    
    # m7 = (a01-a11)*(b10+b11)
    U[0*3+1][c] = 1; U[1*3+1][c] = -1
    V[1*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1
    c += 1
    
    # === Remaining 22 schoolbook terms ===
    # Skip terms where i in {0,1}, j in {0,1}, k in {0,1} (Strassen block)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i in (0, 1) and j in (0, 1) and k in (0, 1):
                    continue  # Handled by Strassen block
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
