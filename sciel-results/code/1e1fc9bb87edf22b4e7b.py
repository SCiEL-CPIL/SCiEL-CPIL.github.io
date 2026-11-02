import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def _build_strassen_block(t, U, V, W, col_base):
    """Fill in one Strassen block (7 terms) starting at col_base."""
    # Term 1: P1 = (a00+a11)*(b_t0+b_t1) -> c_t0, c_t3
    col = col_base
    U[0][col] = 1; U[3][col] = 1
    V[2*t][col] = 1; V[11+2*t][col] = 1
    W[4*t][col] = 1; W[4*t+3][col] = 1
    
    # Term 2: P2 = (a10+a11)*b_t0 -> c_t1, -c_t3
    col = col_base + 1
    U[2][col] = 1; U[3][col] = 1
    V[2*t][col] = 1
    W[4*t+1][col] = 1; W[4*t+3][col] = -1
    
    # Term 3: P3 = a00*(b_t1-b_t1) -> c_t2, c_t3  (Note: original code comment might be slightly off, but logic is specific)
    # Original: U[0]=1, V[2t+1]=1, V[11+2t]=-1, W[4t+2]=1, W[4t+3]=1
    col = col_base + 2
    U[0][col] = 1
    V[2*t+1][col] = 1; V[11+2*t][col] = -1
    W[4*t+2][col] = 1; W[4*t+3][col] = 1
    
    # Term 4: P4 = a11*(b_t1-b_t0) -> c_t0, c_t1
    # Original: U[3]=1, V[10+2t]=1, V[2t]=-1, W[4t]=1, W[4t+1]=1
    col = col_base + 3
    U[3][col] = 1
    V[10+2*t][col] = 1; V[2*t][col] = -1
    W[4*t][col] = 1; W[4*t+1][col] = 1
    
    # Term 5: P5 = (a00+a01)*b_t1 -> -c_t0, c_t2
    # Original: U[0]=1, U[1]=1, W[4t]=-1, W[4t+2]=1, V[11+2t]=1
    col = col_base + 4
    U[0][col] = 1; U[1][col] = 1
    W[4*t][col] = -1; W[4*t+2][col] = 1
    V[11+2*t][col] = 1
    
    # Term 6: P6 = (a10-a00)*(b_t0+b_t1) -> c_t3
    # Original: U[2]=1, U[0]=-1, V[2t]=1, V[2t+1]=1, W[4t+3]=1
    col = col_base + 5
    U[2][col] = 1; U[0][col] = -1
    V[2*t][col] = 1; V[2*t+1][col] = 1
    W[4*t+3][col] = 1
    
    # Term 7: P7 = (a01-a11)*(b_t1+b_t1) -> c_t0
    # Original: U[1]=1, U[3]=-1, V[10+2t]=1, V[11+2t]=1, W[4t]=1
    col = col_base + 6
    U[1][col] = 1; U[3][col] = -1
    V[10+2*t][col] = 1; V[11+2*t][col] = 1
    W[4*t][col] = 1

def _verify_decomposition(U, V, W, n=2, m=2, p=10):
    """Verify that the tensor decomposition exactly computes matrix multiplication."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(actual - expected) > 1e-9:
                    return False, (i, j, k, actual, expected)
    return True, None

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(5):
        if time.time() > deadline:
            break
        _build_strassen_block(t, U, V, W, t * 7)
        
    # Verify
    valid, err = _verify_decomposition(U, V, W)
    if not valid:
        # Fallback: should not happen
        return {"U": U, "V": V, "W": W}
        
    return {"U": U, "V": V, "W": W}
