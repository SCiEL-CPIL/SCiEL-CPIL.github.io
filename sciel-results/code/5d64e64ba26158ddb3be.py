import time
import numpy as np

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    
    T_true = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for l in range(p):
                a_idx = i * m + j
                b_idx = j * p + l
                c_idx = l * n + i
                T_true[a_idx, b_idx, c_idx] = 1.0
    
    return np.allclose(T_decomp, T_true, atol=1e-10)

def _build_strassen_2x2x2():
    """
    Build a rank-7 decomposition for 2x2x2 matrix multiplication.
    Returns U, V, W as lists of lists.
    """
    n, m, p = 2, 2, 2
    rank = 7
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Standard Strassen decomposition for 2x2 matrices
    # A = [[a00, a01], [a10, a11]]
    # B = [[b00, b01], [b10, b11]]
    # C = A*B
    
    # m1 = (a00+a11)(b00+b11)
    U[0*2+0][0] = 1
    U[1*2+1][0] = 1
    V[0*2+0][0] = 1
    V[1*2+1][0] = 1
    W[0*2+0][0] = 1
    W[1*2+1][0] = 1
    
    # m2 = (a10+a11)b00
    U[1*2+0][1] = 1
    U[1*2+1][1] = 1
    V[0*2+0][1] = 1
    W[0*2+1][1] = 1
    W[1*2+1][1] = -1
    
    # m3 = a00(b01-b11)
    U[0*2+0][2] = 1
    V[0*2+1][2] = 1
    V[1*2+1][2] = -1
    W[1*2+0][2] = 1
    W[1*2+1][2] = 1
    
    # m4 = a11(b10-b00)
    U[1*2+1][3] = 1
    V[1*2+0][3] = 1
    V[0*2+0][3] = -1
    W[0*2+0][3] = 1
    W[0*2+1][3] = 1
    
    # m5 = (a00+a01)b11
    U[0*2+0][4] = 1
    U[0*2+1][4] = 1
    V[1*2+1][4] = 1
    W[0*2+0][4] = -1
    W[1*2+0][4] = 1
    
    # m6 = (a10-a00)(b00+b01)
    U[1*2+0][5] = 1
    U[0*2+0][5] = -1
    V[0*2+0][5] = 1
    V[0*2+1][5] = 1
    W[1*2+1][5] = 1
    
    # m7 = (a01-a11)(b10+b11)
    U[0*2+1][6] = 1
    U[1*2+1][6] = -1
    V[1*2+0][6] = 1
    V[1*2+1][6] = 1
    W[0*2+0][6] = 1
    
    return U, V, W

def _build_rank_44(n, m, p):
    """
    Build a rank-44 decomposition for 2x3x8 matrix multiplication.
    
    Strategy:
    - Treat as 2x3x2 tensor repeated 4 times (since p=8 = 4*2)
    - Use Strassen (rank 7) for each 2x2x2 block
    - Total rank: 4 * 7 = 28 for the 2x2x8 part
    - Wait, that's not right. Let's think again.
    
    Actually, let's use a different approach:
    - Decompose 2x3x8 as block multiplication
    - j=0: 2x8 multiplication (rank 16 via schoolbook)
    - j=1,2: 2x2x8 multiplication, split into 4 blocks of 2x2x2
      Each 2x2x2 block uses Strassen (rank 7)
      So 4 blocks * 2 j-values * 7 = 56? No, that's too high.
    
    Let me reconsider. The current best is 44.
    
    Alternative: 
    - 2x3x8 = (2x3x2) x (1x1x4) in some sense
    - Rank(2x3x2) = 11 (known)
    - If we can share factors, maybe we can do better
    
    Actually, let's just use the known rank-44 construction:
    - 16 schoolbook terms for j=0
    - 4 blocks of 2x2x2 Strassen for j=1,2 (each block covers 2 k-values)
    - Wait, that gives 16 + 4*2*7 = 16 + 56 = 72. That's wrong.
    
    Let me re-examine the original construction.
    """
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # Part 1: j=0, schoolbook for 2x8
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    # Part 2: j=1,2 in 4 blocks of 2x2x2 using Strassen
    # Each block covers 2 k-values (k0, k1)
    # For each block, we have 2 j-values (j=1, j=2)
    # But Strassen is for 2x2x2, so it handles 2 i-values, 2 j-values, 2 k-values
    # Here n=2, so i-values are 0,1
    # j-values in a block are 1,2 (but wait, m=3, so j=0,1,2)
    # k-values in a block are 2 consecutive values
    
    # Actually, I think the original construction treats each (i,j,k) triple
    # and uses Strassen-like combinations
    
    # Let me just replicate the original rank-44 construction
    for block in range(4):
        k0 = 2 * block
        k1 = 2 * block + 1
        
        # Strassen for 2x2x2 blocks where j=1,2
        # m1 = (a00+a11)(b00+b11)
        U[0 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m2 = (a10+a11)b00
        U[1 * m + 1][col] = 1
        U[1 * m + 2][col] = 1
        V[1 * p + k0][col] = 1
        W[k0 * n + 1][col] = 1
        W[k1 * n + 1][col] = -1
        col += 1
        
        # m3 = a00(b01-b11)
        U[0 * m + 1][col] = 1
        V[1 * p + k1][col] = 1
        V[2 * p + k1][col] = -1
        W[k1 * n + 0][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m4 = a11(b10-b00)
        U[1 * m + 2][col] = 1
        V[2 * p + k0][col] = 1
        V[1 * p + k0][col] = -1
        W[k0 * n + 0][col] = 1
        W[k0 * n + 1][col] = 1
        col += 1
        
        # m5 = (a00+a01)b11
        U[0 * m + 1][col] = 1
        U[0 * m + 2][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = -1
        W[k1 * n + 0][col] = 1
        col += 1
        
        # m6 = (a10-a00)(b00+b01)
        U[1 * m + 1][col] = 1
        U[0 * m + 1][col] = -1
        V[1 * p + k0][col] = 1
        V[1 * p + k1][col] = 1
        W[k1 * n + 1][col] = 1
        col += 1
        
        # m7 = (a01-a11)(b10+b11)
        U[0 * m + 2][col] = 1
        U[1 * m + 2][col] = -1
        V[2 * p + k0][col] = 1
        V[2 * p + k1][col] = 1
        W[k0 * n + 0][col] = 1
        col += 1
    
    return U, V, W, rank

def construct():
    n, m, p = 2, 3, 8
    
    # Use the rank-44 Strassen-based construction
    U, V, W, rank = _build_rank_44(n, m, p)
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    n, m, p = 2, 3, 8
    U, V, W = result["U"], result["V"], result["W"]
    print(f"Rank: {len(U[0])}")
    print(f"Verified: {_verify_decomposition(U, V, W, n, m, p)}")
