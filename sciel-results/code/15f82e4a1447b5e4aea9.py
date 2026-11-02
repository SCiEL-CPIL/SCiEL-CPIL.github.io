"""Rewritten program: hybrid Strassen/schoolbook decomposition for 2x3x7 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = 150.0
_SAFETY = 5.0

def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _START)

def _verify(U, V, W, n, m, p, trials=20, seed=12345):
    rng = np.random.default_rng(seed)
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    U3 = Ua.reshape(n, m, r)
    V3 = Va.reshape(m, p, r)
    W3 = Wa.reshape(p, n, r)
    for _ in range(trials):
        A = rng.integers(-5, 5, size=(n, m))
        B = rng.integers(-5, 5, size=(m, p))
        C = A @ B
        C2 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    a_part = sum(U3[i, j, c] * A[i, j] for j in range(m))
                    b_part = sum(V3[j, k, c] * B[j, k] for j in range(m))
                    s += W3[k, i, c] * a_part * b_part
                C2[i, k] = s
        if not np.allclose(C, C2, atol=1e-6):
            return False
    return True

def construct():
    n, m, p = DIMENSIONS
    
    # We'll build a decomposition using Strassen for 2x2x2 blocks
    # Split the 3 inner dimension: j=0 handled separately (2x1x7), j=1,2 handled as 2x2x7
    # 2x1x7: rank 14 (schoolbook)
    # 2x2x7: split 7 into 2+2+2+1, use Strassen for 2x2x2 blocks (rank 7 each) and schoolbook for 2x2x1 (rank 4)
    # Total: 14 + 7*3 + 4 = 39
    
    # Actually, let's try to do better.
    # 2x2x3 can potentially be done in rank 8 (known result)
    # 2x2x7 = 2x2x3 + 2x2x2 + 2x2x2 = 8 + 7 + 7 = 22
    # Total: 14 + 22 = 36
    
    # Let me implement a rank-8 decomposition for 2x2x3 if I can find it.
    # Known rank-8 terms for M_{2,2,3}:
    # Using the approach from "A note on the rank of the 2x2x3 matrix multiplication tensor"
    
    # Let's use a simpler approach: just use the 39-rank decomposition which is known to work.
    rank = 39
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    
    # Block 1: j=0. A (2x1), B (1x7).
    for i in range(n):
        for k in range(p):
            U[i*m + 0][col] = 1
            V[0*p + k][col] = 1
            W[k*n + i][col] = 1
            col += 1
            
    # Block 2: j in {1,2}. A (2x2), B (2x7).
    # Split p into 2,2,2,1.
    
    def add_strassen(k_start):
        nonlocal col
        terms = [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 1]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 1, 0, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 1, 0]),
            ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
        ]
        
        for u_vec, v_vec, w_vec in terms:
            for il in range(2):
                for jl in range(2):
                    val = u_vec[il*2 + jl]
                    if val != 0:
                        U[il*m + (1 + jl)][col] += val
            for jl in range(2):
                for kl in range(2):
                    val = v_vec[jl*2 + kl]
                    if val != 0:
                        V[(1 + jl)*p + (k_start + kl)][col] += val
            for kl in range(2):
                for il in range(2):
                    val = w_vec[kl*2 + il]
                    if val != 0:
                        W[(k_start + kl)*n + il][col] += val
            col += 1

    add_strassen(0)
    add_strassen(2)
    add_strassen(4)
    
    # Sub-block: k=6. 2x2x1.
    U[0*m + 1][col] = 1
    V[1*p + 6][col] = 1
    W[6*n + 0][col] = 1
    col += 1
    
    U[0*m + 2][col] = 1
    V[2*p + 6][col] = 1
    W[6*n + 0][col] = 1
    col += 1
    
    U[1*m + 1][col] = 1
    V[1*p + 6][col] = 1
    W[6*n + 1][col] = 1
    col += 1
    
    U[1*m + 2][col] = 1
    V[2*p + 6][col] = 1
    W[6*n + 1][col] = 1
    col += 1
    
    assert col == rank, f"Rank mismatch: {col} vs {rank}"
    
    if _verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        # Fallback to schoolbook
        rank_school = n * m * p
        U_s = [[0]*rank_school for _ in range(n*m)]
        V_s = [[0]*rank_school for _ in range(m*p)]
        W_s = [[0]*rank_school for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][c] = 1
                    V_s[j*p+k][c] = 1
                    W_s[k*n+i][c] = 1
                    c += 1
        return {"U": U_s, "V": V_s, "W": W_s}

# EVOLVE-BLOCK-END
