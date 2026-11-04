import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

def get_strassen_2x2():
    """Return U, V, W for 2x2x2 matrix multiplication (Strassen, rank 7)."""
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # Term 0
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # Term 1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1
    
    # Term 2
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1
    
    # Term 3
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # Term 4
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # Term 5
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1
    
    # Term 6
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def construct_4x4x4_strassen():
    """Construct rank-49 decomposition for 4x4x4 via Kronecker of 2x2 Strassen."""
    U2, V2, W2 = get_strassen_2x2()
    r2 = 7
    r4 = r2 * r2  # 49
    
    U4 = [[0]*r4 for _ in range(16)]
    V4 = [[0]*r4 for _ in range(16)]
    W4 = [[0]*r4 for _ in range(16)]
    
    for i1 in range(2):
        for i2 in range(2):
            for j1 in range(2):
                for j2 in range(2):
                    I = (i1*2+i2)*4 + (j1*2+j2)
                    u_idx1 = i1*2+j1
                    u_idx2 = i2*2+j2
                    for t1 in range(r2):
                        for t2 in range(r2):
                            R = t1*r2+t2
                            U4[I][R] = U2[u_idx1][t1] * U2[u_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            J = (j1*2+j2)*4 + (k1*2+k2)
                            v_idx1 = j1*2+k1
                            v_idx2 = j2*2+k2
                            for t1 in range(r2):
                                for t2 in range(r2):
                                    R = t1*r2+t2
                                    V4[J][R] = V2[v_idx1][t1] * V2[v_idx2][t2]
                    
                    for k1 in range(2):
                        for k2 in range(2):
                            K = (k1*2+k2)*4 + (i1*2+i2)
                            w_idx1 = k1*2+i1
                            w_idx2 = k2*2+i2
                            for t1 in range(r2):
                                for t2 in range(r2):
                                    R = t1*r2+t2
                                    W4[K][R] = W2[w_idx1][t1] * W2[w_idx2][t2]
    
    return U4, V4, W4

def construct_4x4x3_schoolbook():
    """Construct schoolbook rank-48 decomposition for 4x4x3."""
    n, m, p = 4, 4, 3
    rank = n * m * p  # 48
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return U, V, W, rank

def search_4x4x3_low_rank(deadline, seed=42):
    """Search for a low-rank decomposition of the 4x4x3 matrix multiplication tensor.
    Uses vectorized residual computation and structured search.
    Returns (U, V, W, rank) or None if not found within deadline."""
    n, m, p = 4, 4, 3
    rng = np.random.RandomState(seed)
    
    # Precompute index arrays for fast tensor reconstruction
    idx_i = np.arange(n*m)
    idx_j = np.arange(m*p)
    idx_k = np.arange(p*n)
    
    def compute_residual(U, V, W, rank):
        """Compute residual tensor using vectorized ops."""
        # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        # Residual[i,j,k] = sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] - 1
        # Use einsum: for each t, outer product of reshaped matrices
        res = np.zeros((n, m, p))
        for t in range(rank):
            u_t = U[:, t].reshape(n, m)
            v_t = V[:, t].reshape(m, p)
            w_t = W[:, t].reshape(p, n)
            # res += u_t * v_t * w_t elementwise after broadcasting
            # u_t[i,j] * v_t[j,k] * w_t[k,i]
            # This is a tensor contraction
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        res[i,j,k] += u_t[i,j] * v_t[j,k] * w_t[k,i]
        res -= 1.0
        return res
    
    # Try to find a valid decomposition
    # Start with schoolbook and try to reduce rank
    for target_rank in range(30, 49):
        if time.time() > deadline:
            break
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
            
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            # Structured initialization based on schoolbook pattern
            for t in range(target_rank):
                U[:, t] = rng.choice([-1, 0, 1], size=n*m)
                V[:, t] = rng.choice([-1, 0, 1], size=m*p)
                W[:, t] = rng.choice([-1, 0, 1], size=p*n)
            
            # Quick check
            res = compute_residual(U, V, W, target_rank)
            if np.max(np.abs(res)) < 1e-9:
                return U.astype(int).tolist(), V.astype(int).tolist(), W.astype(int).tolist(), target_rank
    
    return None

def verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the n x m x p matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for t in range(rank):
                    s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Part 1: 4x4x4 Strassen, rank 49
    U4, V4, W4 = construct_4x4x4_strassen()
    rank1 = 49
    
    # Part 2: Try to find low-rank decomposition for 4x4x3
    U2, V2, W2, rank2 = None, None, None, 48
    
    if time.time() < deadline - 10:  # Leave time for assembly
        result = search_4x4x3_low_rank(deadline - 10, seed=42)
        if result is not None:
            U2, V2, W2, rank2 = result
    
    if U2 is None:
        # Fall back to schoolbook for 4x4x3
        U2, V2, W2, rank2 = construct_4x4x3_schoolbook()
    
    total_rank = rank1 + rank2
    
    # Assemble the full 4x4x7 decomposition
    U = [[0]*total_rank for _ in range(n*m)]       # 16 rows
    V = [[0]*total_rank for _ in range(m*p)]       # 28 rows
    W = [[0]*total_rank for _ in range(p*n)]       # 28 rows
    
    # Fill U: U4 rows 0..15, U2 rows 0..15
    for i in range(n*m):
        for t in range(rank1):
            U[i][t] = U4[i][t]
        for t in range(rank2):
            U[i][rank1+t] = U2[i][t]
    
    # Fill V: 
    # V4 row j*4+k (k=0..3) -> V row j*7+k
    # V2 row j*3+k' (k'=0..2) -> V row j*7+(4+k')
    for j in range(m):
        for k in range(4):
            v4_row = j*4+k
            v_row = j*p+k
            for t in range(rank1):
                V[v_row][t] = V4[v4_row][t]
        for kp in range(3):
            v2_row = j*3+kp
            v_row = j*p+(4+kp)
            for t in range(rank2):
                V[v_row][rank1+t] = V2[v2_row][t]
    
    # Fill W:
    # W4 row k*4+i (k=0..3) -> W row k*4+i
    # W2 row k'*4+i (k'=0..2) -> W row (4+k')*4+i
    for k in range(4):
        for i in range(n):
            w4_row = k*4+i
            w_row = k*n+i
            for t in range(rank1):
                W[w_row][t] = W4[w4_row][t]
    for kp in range(3):
        for i in range(n):
            w2_row = kp*4+i
            w_row = (4+kp)*n+i
            for t in range(rank2):
                W[w_row][rank1+t] = W2[w2_row][t]
    
    # Quick verification using numpy for speed
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U_np[i*m+j], V_np[j*p+k] * W_np[k*n+i])
                if abs(s - 1) > 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    
    if not ok:
        # Fallback to full schoolbook
        rank = n*m*p
        U_fb = [[0]*rank for _ in range(n*m)]
        V_fb = [[0]*rank for _ in range(m*p)]
        W_fb = [[0]*rank for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i*m+j][c] = 1
                    V_fb[j*p+k][c] = 1
                    W_fb[k*n+i][c] = 1
                    c += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    return {"U": U, "V": V, "W": W}
