import random
import time

DIMENSIONS = [3, 3, 8]
N, M, P = DIMENSIONS
SCHOOLBOOK = N * M * P  # 72

def _verify(U, V, W, tol=1e-9):
    """Check that sum_c U[a,c] V[b,c] W[c] == 1 iff a,b,c correspond to a product."""
    try:
        rank = len(U[0])
        for a in range(N * M):
            for b in range(M * P):
                acc = [0.0] * (P * N)
                for c in range(rank):
                    u = U[a][c]
                    if u == 0:
                        continue
                    v = V[b][c]
                    if v == 0:
                        continue
                    uv = u * v
                    if uv == 0:
                        continue
                    wc = W[c]
                    for k in range(P * N):
                        acc[k] += uv * wc[k]
                i, j = divmod(a, M)
                j2, k2 = divmod(b, P)
                for kk in range(P * N):
                    expected = 1.0 if (kk % N == i and kk // N == j2) else 0.0
                    if abs(acc[kk] - expected) > tol:
                        return False
        return True
    except Exception:
        return False

def _search_3x3_3x4(deadline, rng):
    """Search for a rank-16 decomposition for 3x3 * 3x4."""
    n, m, p = 3, 3, 4
    target_rank = 16
    
    # Try to find a sparse decomposition
    # Structure: 
    # U is 9 x 16
    # V is 12 x 16
    # W is 12 x 16
    
    for it in range(100000):
        if time.time() > deadline:
            return None
        
        U = [[0.0] * target_rank for _ in range(n * m)]
        V = [[0.0] * target_rank for _ in range(m * p)]
        W = [[0.0] * target_rank for _ in range(p * n)]
        
        # Random sparse initialization
        for a in range(n * m):
            nz = rng.randint(1, 4)
            cols = rng.sample(range(target_rank), nz)
            for c in cols:
                U[a][c] = rng.choice([-1, 1])
        
        for b in range(m * p):
            nz = rng.randint(1, 4)
            cols = rng.sample(range(target_rank), nz)
            for c in cols:
                V[b][c] = rng.choice([-1, 1])
        
        for k in range(p * n):
            nz = rng.randint(1, 4)
            cols = rng.sample(range(target_rank), nz)
            for c in cols:
                W[k][c] = rng.choice([-1, 1])
        
        if _verify(U, V, W):
            return U, V, W
            
    return None

def _construct_3x3_3x8(deadline, rng):
    """Construct 3x3 * 3x8 decomposition using two 3x3 * 3x4 blocks."""
    # Split P=8 into P1=4 and P2=4
    # Block 1: A(3x3) * B1(3x4) -> C1(4x3)
    # Block 2: A(3x3) * B2(3x4) -> C2(4x3)
    
    # We need a decomposition for 3x3 * 3x4
    # Let's try to find one. If we fail, we fall back to schoolbook for the sub-blocks (rank 36 each, total 72)
    # But we hope for rank 16 each (total 32)
    
    sub_decomp = _search_3x3_3x4(deadline, rng)
    
    if sub_decomp is None:
        # Fallback: Schoolbook for the whole thing
        rank = SCHOOLBOOK
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        column = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][column] = 1
                    V[j * P + k][column] = 1
                    W[k * N + i][column] = 1
                    column += 1
        return U, V, W

    U_sub, V_sub, W_sub = sub_decomp
    rank_sub = len(U_sub[0])
    total_rank = 2 * rank_sub
    
    U = [[0] * total_rank for _ in range(N * M)]
    V = [[0] * total_rank for _ in range(M * P)]
    W = [[0] * total_rank for _ in range(P * N)]
    
    # Copy Block 1 (k=0..3)
    for a in range(N * M):
        for c in range(rank_sub):
            U[a][c] = U_sub[a][c]
            
    for b in range(M * 4):
        j, k = divmod(b, 4)
        # V index in sub-problem is j*4 + k
        v_idx = j * 4 + k
        for c in range(rank_sub):
            V[b][c] = V_sub[v_idx][c]
            
    for k in range(4 * N):
        kk, i = divmod(k, N)
        # W index in sub-problem is kk*N + i
        w_idx = kk * N + i
        for c in range(rank_sub):
            W[k][c] = W_sub[w_idx][c]
            
    # Copy Block 2 (k=4..7)
    offset = rank_sub
    for a in range(N * M):
        for c in range(rank_sub):
            U[a][c + offset] = U_sub[a][c]
            
    for b in range(M * P):
        j, k = divmod(b, P)
        if k >= 4:
            k_sub = k - 4
            v_idx = j * 4 + k_sub
            for c in range(rank_sub):
                V[b][c + offset] = V_sub[v_idx][c]
                
    for k in range(P * N):
        kk, i = divmod(k, N)
        if kk >= 4:
            kk_sub = kk - 4
            w_idx = kk_sub * N + i
            for c in range(rank_sub):
                W[k][c + offset] = W_sub[w_idx][c]
                
    return U, V, W

def construct():
    start = time.time()
    deadline = start + 140.0
    rng = random.Random(12345)
    
    # Try the constructive strategy
    U, V, W = _construct_3x3_3x8(deadline, rng)
    
    # Verify the result
    if _verify(U, V, W):
        return {"U": U, "V": V, "W": W}
    else:
        # If verification fails (should not happen if logic is correct), fallback to schoolbook
        rank = SCHOOLBOOK
        U = [[0] * rank for _ in range(N * M)]
        V = [[0] * rank for _ in range(M * P)]
        W = [[0] * rank for _ in range(P * N)]
        column = 0
        for i in range(N):
            for j in range(M):
                for k in range(P):
                    U[i * M + j][column] = 1
                    V[j * P + k][column] = 1
                    W[k * N + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
