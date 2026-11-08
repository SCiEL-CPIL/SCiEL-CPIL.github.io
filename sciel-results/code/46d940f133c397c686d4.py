import numpy as np
import time

_START_TIME = time.time()
TIME_LIMIT = 140  # seconds, leave 10s safety margin for 150s evaluator kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

def _verify(U, V, W, n, m, p, num_trials=5):
    """Verify exact integer decomposition using small integer test matrices."""
    rank = len(U[0])
    np.random.seed(42)
    for _ in range(num_trials):
        A = np.random.randint(-3, 4, size=(n, m))
        B = np.random.randint(-3, 4, size=(m, p))
        C_true = A @ B
        C_decomp = np.zeros((n, p))
        for r in range(rank):
            x = np.dot(A.flatten(), [U[i*m+j][r] for i in range(n) for j in range(m)])
            y = np.dot(B.flatten(), [V[j*p+k][r] for j in range(m) for k in range(p)])
            for i in range(n):
                for k in range(p):
                    C_decomp[i,k] += x * y * W[k*n+i][r]
        if not np.array_equal(C_true, C_decomp):
            return False
    return True

def _build_kronecker_solution(n, m, p):
    """Simple Kronecker/naive solution: rank = n*m*p"""
    rank = n * m * p
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i*m*p + j*p + k
                U[i*m+j][r] = 1
                V[j*p+k][r] = 1
                W[k*n+i][r] = 1
    return {"U": U, "V": V, "W": W}

def _build_block_solution(n, m, p):
    """
    Structured block decomposition for 4x5x5 matrix multiplication.
    Splits into:
      - 4x4x4 part using Strassen (rank 49)
      - 4x4x1 part (rank 16)
      - 4x1x5 part (rank 20)
    Total rank: 85
    """
    rank = 85
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    # Micro Strassen Factors for 2x2x2 (Rank 7)
    u_micro = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, 0, 1, -1],# M6
        [0, 1, 0, -1] # M7
    ]
    v_micro = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    w_micro = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 1, 0, 0], # M2: C01
        [0, 1, 0, -1],# M3: C01, C11(-)
        [1, 0, 0, 1], # M4: C00, C11
        [-1, 0, 0, 1],# M5: C00(-), C11
        [0, 0, 1, 0], # M6: C10
        [-1, 0, 1, 0] # M7: C00(-), C10
    ]
    
    # Macro Strassen Factors (same structure, applied to 2x2 blocks)
    u_macro = u_micro
    v_macro = v_micro
    w_macro = w_micro
    
    col = 0
    # Part 1: 4x4x4 Strassen (Rank 49)
    for k in range(7):
        for l in range(7):
            # U term: u_macro[k] (4 blocks) x u_micro[l] (4 elems) -> 16 entries
            for b in range(4):
                if u_macro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if u_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gi = br * 2 + er
                    gj = bc * 2 + ec
                    ui = gi * 5 + gj
                    U[ui][col] += u_macro[k][b] * u_micro[l][o]
            
            # V term: v_macro[k] x v_micro[l]
            for b in range(4):
                if v_macro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if v_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gj = br * 2 + er
                    gk = bc * 2 + ec
                    vi = gj * 5 + gk
                    V[vi][col] += v_macro[k][b] * v_micro[l][o]
            
            # W term: w_macro[k] x w_micro[l]
            for b in range(4):
                if w_macro[k][b] == 0: continue
                br = b // 2
                bc = b % 2
                for o in range(4):
                    if w_micro[l][o] == 0: continue
                    er = o // 2
                    ec = o % 2
                    gi = br * 2 + er
                    gk = bc * 2 + ec
                    wi = gk * 4 + gi
                    W[wi][col] += w_macro[k][b] * w_micro[l][o]
            
            col += 1
    
    # Part 2: 4x4x1 (Rank 16)
    # A(4x4) * B(4x1) -> C(4x1)
    # A indices: i in 0..3, j in 0..3
    # B indices: j in 0..3, k=4
    # C indices: i in 0..3, k=4
    for i in range(4):
        for j in range(4):
            ui = i * 5 + j
            vi = j * 5 + 4
            wi = 4 * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
            
    # Part 3: 4x1x5 (Rank 20)
    # A(4x1) * B(1x5) -> C(4x5)
    # A indices: i in 0..3, j=4
    # B indices: j=4, k in 0..4
    # C indices: i in 0..3, k in 0..4
    for i in range(4):
        for k in range(5):
            ui = i * 5 + 4
            vi = 4 * 5 + k
            wi = k * 4 + i
            U[ui][col] = 1
            V[vi][col] = 1
            W[wi][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}

def _build_random_candidate(n, m, p, rank, rng):
    """Generate a random integer candidate and verify it."""
    U = [[rng.randint(-2,2) for _ in range(rank)] for _ in range(n*m)]
    V = [[rng.randint(-2,2) for _ in range(rank)] for _ in range(m*p)]
    W = [[rng.randint(-2,2) for _ in range(rank)] for _ in range(p*n)]
    if _verify(U, V, W, n, m, p, num_trials=2):
        return {"U": U, "V": V, "W": W}
    return None

def construct():
    n, m, p = 4, 5, 5
    best = None
    best_rank = float('inf')
    
    # Deadline guard
    deadline = _START_TIME + TIME_LIMIT - 5  # 5s buffer
    
    # Strategy 1: Known 85-rank solution
    sol = _build_block_solution(n, m, p)
    rank = len(sol['U'][0])
    if _verify(sol['U'], sol['V'], sol['W'], n, m, p):
        best = sol
        best_rank = rank
    
    # Strategy 2: Try lower ranks with structured candidates
    # The trivial lower bound is n*m*p = 100... no wait, rank can be less.
    # Actually for matrix multiplication C[4,5], the tensor has 4*5*5 = 100 entries.
    # The rank of the matmul tensor M_{n,p,m}... 
    # Known: r(4x4x4) = 49 (Strassen-like). For 4x5x5, lower bounds exist.
    # Let's just try a few ranks below 85 to see if anything works.
    
    rng = np.random.RandomState(12345)
    
    # Try ranks from best_rank-1 down to max(60, best_rank-20)
    # Each attempt: generate random candidate, verify
    for target_rank in range(best_rank - 1, max(55, best_rank - 25), -1):
        if time.time() > deadline:
            break
        # Only try a few random candidates per rank
        for attempt in range(3):
            if time.time() > deadline:
                break
            cand = _build_random_candidate(n, m, p, target_rank, rng)
            if cand is not None:
                best = cand
                best_rank = target_rank
                break  # Found one at this rank, try lower
    
    if best is None:
        # Fallback: always return something valid
        best = _build_kronecker_solution(n, m, p)
        best_rank = n*m*p
    
    return {"U": best["U"], "V": best["V"], "W": best["W"]}
