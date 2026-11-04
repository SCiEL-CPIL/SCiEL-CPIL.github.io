import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    start_time = time.time()
    deadline = 145  # seconds, leave margin for safety

    # 1. Build the Target Tensor T (2x3 * 3x3)
    # T[a, b, c] = 1 if (a,b,c) corresponds to a valid multiplication term
    # Indices: a in [0, nm), b in [0, mp), c in [0, pn)
    # a = i*m + j (i in 0..n-1, j in 0..m-1)
    # b = j*p + k (j in 0..m-1, k in 0..p-1)
    # c = k*n + i (k in 0..p-1, i in 0..n-1)
    
    T_target = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1.0

    # 2. Deterministic Construction for Rank 17
    # The rank of M_{3,3,3} is 19 (best known for general 3x3). 
    # However, for M_{2,3,3}, we can potentially do better.
    # Let's try to construct a rank 17 decomposition directly.
    #
    # Strategy: 
    # We know that M_{1,3,3} has rank 9.
    # M_{2,3,3} is the direct sum of two M_{1,3,3} tensors if we consider the rows independently.
    # Rank(M_{2,3,3}) <= 18.
    # Can we do 17?
    # 
    # Let's use a known decomposition for M_{2,2,2} (Strassen, rank 7) and extend it.
    # Or, simpler: use the fact that the tensor is sparse.
    #
    # Let's attempt to find a rank 17 decomposition by solving a system of linear equations
    # for a specific pattern of non-zeros.
    #
    # We will define U, V, W with many zeros and solve for the non-zero entries.
    # To ensure exactness, we use Fraction.
    
    # If we fail to find a better one quickly, we return schoolbook (18).
    # But let's try to hardcode a known rank 17 solution if possible.
    # 
    # Actually, a very strong heuristic for these specific small dimensions is to use
    # the "Landsberg" or "Strassen-like" algorithms.
    #
    # Let's try a random search with exact verification for rank 17, but limited to 
    # integer entries in {-1, 0, 1}. This space is small enough to be searched 
    # efficiently if we use pruning or just a few thousand attempts.
    
    def check_exact(U, V, W, T_target, tol=1e-9):
        # U, V, W are lists of Fractions or floats
        r = len(U[0]) if U else 0
        if r == 0: return False
        T_approx = np.zeros((nm, mp, pn))
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    val = 0.0
                    for r_idx in range(r):
                        val += float(U[a][r_idx]) * float(V[b][r_idx]) * float(W[c][r_idx])
                    T_approx[a, b, c] = val
        return np.allclose(T_approx, T_target, atol=tol)

    def to_lists(mat):
        return [[float(x) for x in row] for row in mat]

    # Schoolbook fallback (Rank 18)
    rank_schoolbook = n * m * p
    U_schoolbook = [[0.0]*rank_schoolbook for _ in range(nm)]
    V_schoolbook = [[0.0]*rank_schoolbook for _ in range(mp)]
    W_schoolbook = [[0.0]*rank_schoolbook for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U_schoolbook[a][col] = 1.0
                V_schoolbook[b][col] = 1.0
                W_schoolbook[c][col] = 1.0
                col += 1
                
    best_rank = rank_schoolbook
    best_U = U_schoolbook
    best_V = V_schoolbook
    best_W = W_schoolbook

    # Try to find rank 17 using a structured random search
    # We restrict entries to {-1, 0, 1} to keep it fast and likely to find exact solutions
    # if they exist with small integers.
    
    import random
    random.seed(42)
    
    def try_rank_int_search(r, max_iters=2000):
        if time.time() - start_time > deadline - 5:
            return None
            
        for _ in range(max_iters):
            if time.time() - start_time > deadline - 5:
                break
                
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(nm)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(mp)]
            W = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(pn)]
            
            if check_exact(U, V, W, T_target):
                return (U, V, W)
                
        return None

    # Try rank 17
    if time.time() - start_time < deadline - 10:
        decomp = try_rank_int_search(17, max_iters=500)
        if decomp is not None:
            best_rank = 17
            best_U, best_V, best_W = decomp

    # If rank 17 failed, try rank 16 with fewer iterations (less likely but possible)
    if best_rank > 16 and time.time() - start_time < deadline - 5:
        decomp = try_rank_int_search(16, max_iters=200)
        if decomp is not None:
            best_rank = 16
            best_U, best_V, best_W = decomp

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
