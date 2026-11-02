"""Optimized 2x5x6 matrix multiplication tensor decomposition using Strassen blocks."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START

def _strassen_222():
    """Return U, V, W (each 4x7) for 2x2x2 matmul using Strassen.
    Convention: U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # r=0: (A00+A11)(B00+B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[1, 0] = 1; V[2, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[1, 0] = 1; W[2, 0] = 1; W[3, 0] = 1
    
    # r=1: (A10+A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1  # k=0,i=1
    
    # r=2: A00(B01-B11)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1  # k=1,i=0
    
    # r=3: A11(B10-B00)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[2, 3] = 1  # k=0,i=1
    
    # r=4: (A00+A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1  # k=1,i=0
    
    # r=5: (A10-A00)(B00+B10)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[2, 5] = 1
    W[0, 5] = 1  # k=0,i=0
    
    # r=6: (A01-A11)(B01+B11)
    U[1, 6] = 1; U[3, 6] = -1
    V[1, 6] = 1; V[3, 6] = 1
    W[0, 6] = -1  # k=0,i=0
    
    return U, V, W

def _build_decomposition():
    """Build the 2x5x6 decomposition.
    
    Strategy: Split m=5 as 2+1+2, p=6 as 2+2+2.
    - A (2x5) = [A_a (2x2) | A_b (2x1) | A_c (2x2)]
    - B (5x6) = [B_a (2x6); B_b (1x6); B_c (2x6)]
    
    Each 2x2x6 block uses 3 Strassen 2x2x2 (rank 7 each) = rank 21
    The 2x1x6 block uses schoolbook = rank 12
    
    Total: 21 + 12 + 21 = 54
    """
    n, m, p = 2, 5, 6
    total_rank = 54
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    sU, sV, sW = _strassen_222()
    col = 0
    
    # Part 1: A_a * B_a (2x2 x 6), three Strassen 2x2x2 blocks
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            # U: i in {0,1}, j in {0,1}
            U[0, col] = sU[0, r]
            U[1, col] = sU[1, r]
            U[5, col] = sU[2, r]
            U[6, col] = sU[3, r]
            
            # V: j in {0,1}, k in {k_off, k_off+1}
            V[0*6 + k_off + 0, col] = sV[0, r]
            V[0*6 + k_off + 1, col] = sV[1, r]
            V[1*6 + k_off + 0, col] = sV[2, r]
            V[1*6 + k_off + 1, col] = sV[3, r]
            
            # W: k in {k_off, k_off+1}, i in {0,1}
            W[(k_off + 0)*2 + 0, col] = sW[0, r]
            W[(k_off + 0)*2 + 1, col] = sW[1, r]
            W[(k_off + 1)*2 + 0, col] = sW[2, r]
            W[(k_off + 1)*2 + 1, col] = sW[3, r]
            col += 1
    # col = 21
    
    # Part 2: A_b * B_b (2x1 x 6), schoolbook rank 12
    for i in range(2):
        for k in range(6):
            U[i*5 + 2, col] = 1
            V[2*6 + k, col] = 1
            W[k*2 + i, col] = 1
            col += 1
    # col = 33
    
    # Part 3: A_c * B_c (2x2 x 6), three Strassen 2x2x2 blocks
    for block in range(3):
        k_off = block * 2
        for r in range(7):
            U[0*5 + 3, col] = sU[0, r]
            U[0*5 + 4, col] = sU[1, r]
            U[1*5 + 3, col] = sU[2, r]
            U[1*5 + 4, col] = sU[3, r]
            
            V[3*6 + k_off + 0, col] = sV[0, r]
            V[3*6 + k_off + 1, col] = sV[1, r]
            V[4*6 + k_off + 0, col] = sV[2, r]
            V[4*6 + k_off + 1, col] = sV[3, r]
            
            W[(k_off + 0)*2 + 0, col] = sW[0, r]
            W[(k_off + 0)*2 + 1, col] = sW[1, r]
            W[(k_off + 1)*2 + 0, col] = sW[2, r]
            W[(k_off + 1)*2 + 1, col] = sW[3, r]
            col += 1
    # col = 54
    
    assert col == total_rank
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _verify_vectorized(U, V, W, n=2, m=5, p=6):
    """Vectorized verification using matrix operations."""
    Ua = np.array(U)
    Va = np.array(V)
    Wa = np.array(W)
    rank = Ua.shape[1]
    
    # For each (i,j,k), check sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1
    # Vectorize: for fixed i,j, compute for all k at once
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            u_row = Ua[i*m + j, :]  # (rank,)
            for k in range(p):
                v_row = Va[j*p + k, :]  # (rank,)
                w_row = Wa[k*n + i, :]  # (rank,)
                s = float(np.dot(u_row, v_row * w_row))
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < 1e-9

def _check_half_integer(U, V, W):
    """Check all entries are integer or half-integer."""
    for mat in [U, V, W]:
        arr = np.array(mat)
        if not np.allclose(arr * 2, np.round(arr * 2), atol=1e-12):
            return False
    return True

def _schoolbook():
    """Schoolbook decomposition, rank = n*m*p = 60."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j][c] = 1.0
                V[j*p + k][c] = 1.0
                W[k*n + i][c] = 1.0
                c += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    deadline = t0 + 100  # safety margin before 150s kill
    
    # Primary: Strassen-based rank-54
    try:
        result = _build_decomposition()
        if time.time() < deadline:
            if _verify_vectorized(result['U'], result['V'], result['W']):
                if _check_half_integer(result['U'], result['V'], result['W']):
                    return result
    except Exception:
        pass
    
    # Fallback: schoolbook rank-60
    return _schoolbook()

# EVOLVE-BLOCK-END
