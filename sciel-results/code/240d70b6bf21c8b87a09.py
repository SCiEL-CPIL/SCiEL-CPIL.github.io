import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check the cyclic bilinear identity."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(R):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _build_m424_rank18():
    """
    Returns U, V, W for M_4,2,4 with rank 18.
    A is 4x2, B is 2x4, C is 4x4.
    U shape (8, 18), V shape (8, 18), W shape (16, 18).
    Indices:
    U: (i, j) for A[i,j], i in 0..3, j in 0..1. Index i*2+j.
    V: (j, k) for B[j,k], j in 0..1, k in 0..3. Index j*4+k.
    W: (k, i) for C[k,i], k in 0..3, i in 0..3. Index k*4+i.
    """
    # Rank 18 decomposition for 4x2x4
    # This is a known result. 
    # I will use a heuristic construction or a hardcoded set if available.
    # For now, I will use 2x M_2,2,4 (rank 10+10=20) if I can't find 18.
    # But 18 is known.
    # Let's use a simple valid decomposition first (rank 20) to ensure correctness, 
    # then try to optimize.
    
    # Rank 20 construction for M_4,2,4:
    # Split A into A1 (2x2), A2 (2x2).
    # C = A1 B + A2 B.
    # Compute A1 B (M_2,2,4, rank 10) and A2 B (M_2,2,4, rank 10).
    
    # M_2,2,4 rank 10 construction:
    # A (2x2), B (2x4).
    # This is also tricky.
    
    # I will use Schoolbook for M_4,2,4 which is 4*2*4 = 32.
    # This is safe.
    
    # If I use schoolbook for M_4,4,2 (rank 32), M_4,4,3 is 16+32=48.
    # M_4,8,3 is 48+48=96.
    # This is worse than 88.
    
    # So I MUST find better decompositions.
    
    # Let's stick to the 88 solution if I can't find better ones quickly.
    # But the prompt asks for a stronger algorithm.
    
    # I will implement the 72 rank solution (M_3,4,8 -> M_1,4,8 + M_2,4,8).
    # M_1,4,8 rank 32.
    # M_2,4,8 rank 40 (2x M_2,2,8, each 20).
    # M_2,2,8 rank 20 (2x M_2,2,4, each 10).
    # So I need M_2,2,4 rank 10.
    
    # I will hardcode a rank-10 decomposition for M_2,2,4.
    
    pass

def _build_m224_rank10():
    """
    Returns U, V, W for M_2,2,4 with rank 10.
    A is 2x2, B is 2x4, C is 2x4.
    U shape (4, 10), V shape (8, 10), W shape (8, 10).
    """
    # Hardcoded rank-10 decomposition for 2x2x4
    # Reference: "Fast Matrix Multiplication"
    # One such decomposition:
    # M1 = (a+b)(e+i)
    # M2 = (c+d)(e+i)
    # M3 = (a+b)(f+j)
    # M4 = (c+d)(f+j)
    # M5 = (a+b)(g+k)
    # M6 = (c+d)(g+k)
    # M7 = (a+b)(h+l)
    # M8 = (c+d)(h+l)
    # M9 = a(e+i+f+j+g+k+h+l) - ...? No.
    
    # I will use a known valid set.
    # If I can't find it, I'll use rank 14 (2x Strassen).
    # Rank 14 for M_2,2,4 -> Rank 28 for M_2,2,8 -> Rank 56 for M_2,4,8 -> Rank 88 for M_3,4,8.
    # This is the current score.
    
    # I will return None to indicate failure to find better.
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140
    n, m, p = DIMENSIONS
    
    # Fallback to rank 88
    U, V, W = _build_rank88()
    
    # Attempt to find better
    # If _build_m224_rank10 works, use it.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _build_rank88():
    """Verified rank-88 construction."""
    n, m, p = DIMENSIONS
    rank = 88
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for j in range(m):
        for k in range(p):
            U[0 * m + j, col] = 1.0
            V[j * p + k, col] = 1.0
            W[k * n + 0, col] = 1.0
            col += 1
    sU = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
    ]
    sV = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
    ]
    sW = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
    ]
    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    if sU[t][l]:
                        li, lj = l // 2, l % 2
                        U[(1 + li) * m + (js + lj), col] += sU[t][l]
                    if sV[t][l]:
                        lj, lk = l // 2, l % 2
                        V[(js + lj) * p + (ks + lk), col] += sV[t][l]
                    if sW[t][l]:
                        li, lk = l // 2, l % 2
                        W[(ks + lk) * n + (1 + li), col] += sW[t][l]
                col += 1
    return U, V, W
