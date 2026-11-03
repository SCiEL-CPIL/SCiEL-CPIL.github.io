"""Matrix multiplication 2,3,5 — rank-28 decomposition via two Strassen 2x2x2 blocks."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3 * 3x5 matrix multiplication using rank-28 ansatz.
    
    We use two disjoint Strassen 2x2x2 blocks:
    - Block A: i in {0,1}, j in {0,1}, k in {0,1} -> 7 Strassen terms
    - Block B: i in {0,1}, j in {0,1}, k in {3,4} -> 7 Strassen terms
    - Remaining schoolbook terms: 30 - 8 - 8 = 14 terms
    Total rank: 7 + 7 + 14 = 28
    """
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    
    # === Block A: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1} ===
    # m1 = (a00+a11)*(b00+b11), contributes to c00(+1), c11(+1)
    U[0*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m2 = (a10+a11)*b00, contributes to c10(+1), c11(-1)
    U[1*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+0][c] = 1
    W[0*2+1][c] = 1; W[1*2+1][c] = -1
    c += 1
    
    # m3 = a00*(b01-b11), contributes to c01(+1), c11(+1)
    U[0*3+0][c] = 1
    V[0*5+1][c] = 1; V[1*5+1][c] = -1
    W[1*2+0][c] = 1; W[1*2+1][c] = 1
    c += 1
    
    # m4 = a11*(b10-b00), contributes to c00(+1), c10(+1)
    U[1*3+1][c] = 1
    V[1*5+0][c] = 1; V[0*5+0][c] = -1
    W[0*2+0][c] = 1; W[0*2+1][c] = 1
    c += 1
    
    # m5 = (a00+a01)*b11, contributes to c00(-1), c01(+1)
    U[0*3+0][c] = 1; U[0*3+1][c] = 1
    V[1*5+1][c] = 1
    W[0*2+0][c] = -1; W[1*2+0][c] = 1
    c += 1
    
    # m6 = (a10-a00)*(b00+b01), contributes to c11(+1)
    U[1*3+0][c] = 1; U[0*3+0][c] = -1
    V[0*5+0][c] = 1; V[0*5+1][c] = 1
    W[1*2+1][c] = 1
    c += 1
    
    # m7 = (a01-a11)*(b10+b11), contributes to c00(+1)
    U[0*3+1][c] = 1; U[1*3+1][c] = -1
    V[1*5+0][c] = 1; V[1*5+1][c] = 1
    W[0*2+0][c] = 1
    c += 1
    
    # === Block B: Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {3,4} ===
    # m1' = (a00+a11)*(b03+b14), contributes to c03(+1), c14(+1)
    U[0*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+3][c] = 1; V[1*5+4][c] = 1
    W[3*2+0][c] = 1; W[4*2+1][c] = 1
    c += 1
    
    # m2' = (a10+a11)*b03, contributes to c13(+1), c14(-1)
    U[1*3+0][c] = 1; U[1*3+1][c] = 1
    V[0*5+3][c] = 1
    W[3*2+1][c] = 1; W[4*2+1][c] = -1
    c += 1
    
    # m3' = a00*(b04-b14), contributes to c04(+1), c14(+1)
    U[0*3+0][c] = 1
    V[0*5+4][c] = 1; V[1*5+4][c] = -1
    W[4*2+0][c] = 1; W[4*2+1][c] = 1
    c += 1
    
    # m4' = a11*(b13-b03), contributes to c03(+1), c13(+1)
    U[1*3+1][c] = 1
    V[1*5+3][c] = 1; V[0*5+3][c] = -1
    W[3*2+0][c] = 1; W[3*2+1][c] = 1
    c += 1
    
    # m5' = (a00+a01)*b14, contributes to c03(-1), c04(+1)
    U[0*3+0][c] = 1; U[0*3+1][c] = 1
    V[1*5+4][c] = 1
    W[3*2+0][c] = -1; W[4*2+0][c] = 1
    c += 1
    
    # m6' = (a10-a00)*(b03+b04), contributes to c14(+1)
    U[1*3+0][c] = 1; U[0*3+0][c] = -1
    V[0*5+3][c] = 1; V[0*5+4][c] = 1
    W[4*2+1][c] = 1
    c += 1
    
    # m7' = (a01-a11)*(b13+b14), contributes to c03(+1)
    U[0*3+1][c] = 1; U[1*3+1][c] = -1
    V[1*5+3][c] = 1; V[1*5+4][c] = 1
    W[3*2+0][c] = 1
    c += 1
    
    # === Remaining 14 schoolbook terms ===
    # Skip terms where j in {0,1} AND k in {0,1} (Block A)
    # Skip terms where j in {0,1} AND k in {3,4} (Block B)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if j in (0, 1) and k in (0, 1):
                    continue  # Handled by Block A
                if j in (0, 1) and k in (3, 4):
                    continue  # Handled by Block B
                U[i*m+j][c] = 1
                V[j*p+k][c] = 1
                W[k*n+i][c] = 1
                c += 1
    
    assert c == rank, f"Expected rank {rank}, got {c}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
