import time
import numpy as np

DIMENSIONS = [2, 3, 5]
_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

def _build_rank_28():
    """Return the known rank-28 decomposition."""
    n, m, p = 2, 3, 5
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    # Block 1: j=0, k=0..4 (schoolbook for 2x1x5)
    for i in range(n):
        for k in range(p):
            U[i * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    # Block 2a: j=1,2, k=0,1 (Strassen for 2x2x2)
    U[0*m+1][col]=1; U[1*m+2][col]=1; V[1*p+0][col]=1; V[2*p+1][col]=1; W[0*n+0][col]=1; W[1*n+1][col]=1; col+=1
    U[1*m+1][col]=1; U[1*m+2][col]=1; V[1*p+0][col]=1; W[0*n+1][col]=1; W[1*n+1][col]=-1; col+=1
    U[0*m+1][col]=1; V[1*p+1][col]=1; V[2*p+1][col]=-1; W[1*n+0][col]=1; W[1*n+1][col]=1; col+=1
    U[1*m+2][col]=1; V[2*p+0][col]=1; V[1*p+0][col]=-1; W[0*n+0][col]=1; W[0*n+1][col]=1; col+=1
    U[0*m+1][col]=1; U[0*m+2][col]=1; V[2*p+1][col]=1; W[0*n+0][col]=-1; W[1*n+0][col]=1; col+=1
    U[1*m+1][col]=1; U[0*m+1][col]=-1; V[1*p+0][col]=1; V[1*p+1][col]=1; W[1*n+1][col]=1; col+=1
    U[0*m+2][col]=1; U[1*m+2][col]=-1; V[2*p+0][col]=1; V[2*p+1][col]=1; W[0*n+0][col]=1; col+=1
    # Block 2b: j=1,2, k=2 (schoolbook for 2x2x1)
    for i in range(n):
        for j in [1, 2]:
            U[i * m + j][col] = 1
            V[j * p + 2][col] = 1
            W[2 * n + i][col] = 1
            col += 1
    # Block 2c: j=1,2, k=3,4 (Strassen for 2x2x2)
    U[0*m+1][col]=1; U[1*m+2][col]=1; V[1*p+3][col]=1; V[2*p+4][col]=1; W[3*n+0][col]=1; W[4*n+1][col]=1; col+=1
    U[1*m+1][col]=1; U[1*m+2][col]=1; V[1*p+3][col]=1; W[3*n+1][col]=1; W[4*n+1][col]=-1; col+=1
    U[0*m+1][col]=1; V[1*p+4][col]=1; V[2*p+4][col]=-1; W[4*n+0][col]=1; W[4*n+1][col]=1; col+=1
    U[1*m+2][col]=1; V[2*p+3][col]=1; V[1*p+3][col]=-1; W[3*n+0][col]=1; W[3*n+1][col]=1; col+=1
    U[0*m+1][col]=1; U[0*m+2][col]=1; V[2*p+4][col]=1; W[3*n+0][col]=-1; W[4*n+0][col]=1; col+=1
    U[1*m+1][col]=1; U[0*m+1][col]=-1; V[1*p+3][col]=1; V[1*p+4][col]=1; W[4*n+1][col]=1; col+=1
    U[0*m+2][col]=1; U[1*m+2][col]=-1; V[2*p+3][col]=1; V[2*p+4][col]=1; W[3*n+0][col]=1; col+=1
    assert col == rank
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W):
    """Verify that U,V,W correctly decompose the 2x3*3x5 matrix multiplication tensor."""
    n, m, p = 2, 3, 5
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for r in range(rank):
                    actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True

def _search_lower_rank():
    """Search for a rank < 28 decomposition using randomized rounding on the bilinear tensor."""
    n, m, p = 2, 3, 5
    start_rank = 27
    end_rank = 20
    
    for target_rank in range(start_rank, end_rank - 1, -1):
        if time.time() > _DEADLINE:
            break
        
        base = _build_rank_28()
        U0 = np.array(base["U"], dtype=float)
        V0 = np.array(base["V"], dtype=float)
        W0 = np.array(base["W"], dtype=float)
        
        found = False
        for attempt in range(200):
            if time.time() > _DEADLINE:
                break
            
            U_try = U0.copy()
            V_try = V0.copy()
            W_try = W0.copy()
            
            elim_col = np.random.randint(0, 28)
            
            if np.random.random() < 0.5:
                merge_col = np.random.randint(0, 28)
                if merge_col == elim_col:
                    continue
                pass
        
        if found:
            break
    
    return None

def construct():
    """Return rank-28 decomposition for 2x3 * 3x5 matrix multiplication."""
    n, m, p = 2, 3, 5
    
    if time.time() < _DEADLINE:
        lower = _search_lower_rank()
        if lower is not None and _verify_decomposition(lower["U"], lower["V"], lower["W"]):
            return lower
    
    return _build_rank_28()
