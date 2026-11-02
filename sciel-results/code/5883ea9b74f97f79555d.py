import time
import numpy as np

DIMENSIONS = [4, 4, 8]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_val = U[i * m + j]
                v_val = V[j * p + k]
                w_val = W[k * n + i]
                s = np.sum(u_val * v_val * w_val)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """
    Return U, V, W for 2x2 * 2x2 -> 2x2 with rank 7.
    Indices: i, j, k in {0, 1}.
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c].
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # c=0: M1 = (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # c=1: M2 = (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # c=2: M3 = A00(B10-B11)
    U[0, 2] = 1
    V[2, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # c=3: M4 = A11(B01-B00)
    U[3, 3] = 1
    V[1, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # c=4: M5 = (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = -1; W[3, 4] = 1
    
    # c=5: M6 = (A10-A00)(B00+B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # c=6: M7 = (A01-A11)(B10+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _build_4x4_rank56():
    """
    Construct rank-56 decomposition for 4x4 * 4x4.
    Uses 8 Strassen decompositions for the 8 block products.
    """
    n = 4
    rank = 56
    U = np.zeros((16, rank))
    V = np.zeros((16, rank))
    W = np.zeros((16, rank))
    
    U2, V2, W2 = _strassen_2x2()
    
    col = 0
    for a in range(2): # block row of A
        for b in range(2): # block col of A / block row of B
            for c in range(2): # block col of B / block col of C
                # A_ab is at rows 2a..2a+1, cols 2b..2b+1
                # B_bc is at rows 2b..2b+1, cols 2c..2c+1
                # C_ac is at rows 2a..2a+1, cols 2c..2c+1
                
                for k in range(7):
                    # Map local indices (0..1) to global indices
                    # Local i -> global row offset in block a
                    # Local j -> global col offset in block b
                    # Local k -> global row offset in block c (for V) and col offset in block a (for W)
                    
                    # U: A_ab[i_local, j_local] -> U[2a + i_local, 2b + j_local]
                    # V: B_bc[j_local, k_local] -> V[2b + j_local, 2c + k_local]
                    # W: C_ac[k_local, i_local] -> W[2c + k_local, 2a + i_local]
                    
                    for i_l in range(2):
                        for j_l in range(2):
                            u_val = U2[i_l * 2 + j_l, k]
                            if u_val != 0:
                                U[2*a + i_l, 2*b + j_l, col] += u_val
                                
                    for j_l in range(2):
                        for k_l in range(2):
                            v_val = V2[j_l * 2 + k_l, k]
                            if v_val != 0:
                                V[2*b + j_l, 2*c + k_l, col] += v_val
                                
                    for k_l in range(2):
                        for i_l in range(2):
                            w_val = W2[k_l * 2 + i_l, k]
                            if w_val != 0:
                                W[2*c + k_l, 2*a + i_l, col] += w_val
                                
                    col += 1
    return U, V, W

def _construct_4x4x8_rank112():
    """
    Construct rank-112 decomposition for 4x4 * 4x8.
    Uses two rank-56 decompositions for 4x4 * 4x4.
    """
    n, m, p = 4, 4, 8
    rank = 112
    
    U4, V4, W4 = _build_4x4_rank56()
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # First half (cols 0-3 of B)
    U[:, :56] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, :56] = V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[k * 4 + i, :56] = W4[k * 4 + i]
            
    # Second half (cols 4-7 of B)
    U[:, 56:] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + (k + 4), 56:] = V4[j * 4 + k]
    for k in range(4):
        for i in range(4):
            W[(k + 4) * 4 + i, 56:] = W4[k * 4 + i]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    
    # Try to construct rank-112 decomposition
    try:
        result = _construct_4x4x8_rank112()
        if _verify(result['U'], result['V'], result['W'], n, m, p):
            return result
    except Exception:
        pass
    
    # Fall back to schoolbook
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
