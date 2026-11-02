"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random
from collections import defaultdict

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def _verify_fast(U, V, W, n, m, p):
    """Faster verification using numpy."""
    r = len(U[0])
    U_arr = np.array(U, dtype=np.float64).reshape(n, m, r)
    V_arr = np.array(V, dtype=np.float64).reshape(m, p, r)
    W_arr = np.array(W, dtype=np.float64).reshape(p, n, r)
    
    Ue = U_arr.reshape(n, m, 1, r)
    Ve = V_arr.reshape(1, m, p, r)
    We = W_arr.T.reshape(n, 1, p, r)
    
    T = np.sum(Ue * Ve * We, axis=3)
    return np.allclose(T, np.ones((n, m, p)), atol=1e-9)

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Strassen's algorithm for 2x2 matrix multiplication, rank 7.
    Convention: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2: U has 4 rows (i in 0,1; j in 0,1), V has 4 rows (j in 0,1; k in 0,1), W has 4 rows (k in 0,1; i in 0,1)
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a+d)(e+h)
    # a=A[0,0] -> U[0,0], d=A[1,1] -> U[3,0]
    # e=B[0,0] -> V[0,0], h=B[1,1] -> V[3,0]
    # Contributes to C[0,0] (i=0,k=0) -> W[0,0] with +1
    # Contributes to C[1,1] (i=1,k=1) -> W[3,0] with +1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (c+d)e
    # c=A[1,0] -> U[2,1], d=A[1,1] -> U[3,1]
    # e=B[0,0] -> V[0,1]
    # Contributes to C[1,0] (i=1,k=0) -> W[2,1] with +1
    # Contributes to C[1,1] (i=1,k=1) -> W[3,1] with -1
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a(f-h)
    # a=A[0,0] -> U[0,2]
    # f=B[0,1] -> V[1,2], h=B[1,1] -> V[3,2]
    # Contributes to C[0,1] (i=0,k=1) -> W[1,2] with +1
    # Contributes to C[1,1] (i=1,k=1) -> W[3,2] with +1
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = d(g-e)
    # d=A[1,1] -> U[3,3]
    # g=B[1,0] -> V[2,3], e=B[0,0] -> V[0,3]
    # Contributes to C[0,0] (i=0,k=0) -> W[0,3] with +1
    # Contributes to C[1,0] (i=1,k=0) -> W[2,3] with +1
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a+b)h
    # a=A[0,0] -> U[0,4], b=A[0,1] -> U[1,4]
    # h=B[1,1] -> V[3,4]
    # Contributes to C[0,0] (i=0,k=0) -> W[0,4] with -1
    # Contributes to C[0,1] (i=0,k=1) -> W[1,4] with +1
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (c-d)(f+h)
    # c=A[1,0] -> U[2,5], d=A[1,1] -> U[3,5]
    # f=B[0,1] -> V[1,5], h=B[1,1] -> V[3,5]
    # Contributes to C[1,1] (i=1,k=1) -> W[3,5] with +1
    U[2, 5] = 1; U[3, 5] = -1
    V[1, 5] = 1; V[3, 5] = 1
    W[3, 5] = 1
    
    # M7 = (b-d)(e+g)
    # b=A[0,1] -> U[1,6], d=A[1,1] -> U[3,6]
    # e=B[0,0] -> V[0,6], g=B[1,0] -> V[2,6]
    # Contributes to C[0,0] (i=0,k=0) -> W[0,6] with +1
    U[1, 6] = 1; U[3, 6] = -1
    V[0, 6] = 1; V[2, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def _trivial_1x2x2():
    """Trivial 1x2x2 decomposition, rank 4."""
    # A is 1x2, B is 2x2, C is 1x2
    # C[0,k] = sum_j A[0,j]*B[j,k]
    # U has 1*2=2 rows, V has 2*2=4 rows, W has 2*1=2 rows
    U = np.zeros((2, 4))
    V = np.zeros((4, 4))
    W = np.zeros((2, 4))
    c = 0
    for j in range(2):
        for k in range(2):
            U[j, c] = 1
            V[j*2+k, c] = 1
            W[k, c] = 1
            c += 1
    return U, V, W

def _combine_blocks(n, m, p, blocks):
    """
    Combine block decompositions.
    blocks: list of (n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off)
    """
    total_rank = sum(b[3].shape[1] for b in blocks)
    U = np.zeros((n*m, total_rank))
    V = np.zeros((m*p, total_rank))
    W = np.zeros((p*n, total_rank))
    
    col_offset = 0
    for (n_b, m_b, p_b, U_b, V_b, W_b, i_off, j_off, k_off) in blocks:
        r_b = U_b.shape[1]
        for i_b in range(n_b):
            for j_b in range(m_b):
                U[(i_off+i_b)*m + (j_off+j_b), col_offset:col_offset+r_b] = U_b[i_b*m_b+j_b, :]
        for j_b in range(m_b):
            for k_b in range(p_b):
                V[(j_off+j_b)*p + (k_off+k_b), col_offset:col_offset+r_b] = V_b[j_b*p_b+k_b, :]
        for k_b in range(p_b):
            for i_b in range(n_b):
                W[(k_off+k_b)*n + (i_off+i_b), col_offset:col_offset+r_b] = W_b[k_b*n_b+i_b, :]
        col_offset += r_b
    
    return U, V, W, total_rank

def _hybrid_strassen_3x4x4():
    """
    Split 3x4x4 into 4 blocks:
    - (0,0,0) 2x2x2: Strassen rank 7
    - (0,2,0) 2x2x2: Strassen rank 7
    - (2,0,0) 1x2x2: trivial rank 4
    - (2,2,0) 1x2x2: trivial rank 4
    Total: 22
    """
    S_U, S_V, S_W = _strassen_2x2x2()
    T_U, T_V, T_W = _trivial_1x2x2()
    
    blocks = [
        (2, 2, 2, S_U, S_V, S_W, 0, 0, 0),
        (2, 2, 2, S_U, S_V, S_W, 0, 2, 0),
        (1, 2, 2, T_U, T_V, T_W, 2, 0, 0),
        (1, 2, 2, T_U, T_V, T_W, 2, 2, 0),
    ]
    
    U, V, W, rank = _combine_blocks(3, 4, 4, blocks)
    return U, V, W, rank

def _try_random_uv_solve_w(n, m, p, target_rank, rng, max_attempts=1000):
    """
    Try to find a decomposition by randomly choosing U and V with small integer entries,
    and then solving for W using linear systems for each (k, i) pair.
    """
    for _ in range(max_attempts):
        U = np.random.randint(-2, 3, size=(n, m, target_rank))
        V = np.random.randint(-2, 3, size=(m, p, target_rank))
        
        W_solution = np.zeros((p, n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i, j, c] * V[j, k, c]
                
                b = np.ones(m)
                
                try:
                    x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                two_x = 2 * x
                rounded_two_x = np.round(two_x)
                if not np.allclose(two_x, rounded_two_x, atol=1e-6):
                    valid = False
                    break
                
                W_solution[k, i, :] = x
        
        if not valid:
            continue
            
        if _verify_fast(U.tolist(), V.tolist(), W_solution.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W_solution.tolist(), target_rank
            
    return None

def _try_gaussian_random(n, m, p, target_rank, rng, max_attempts=200):
    """Try with Gaussian random then round to integers."""
    for attempt in range(max_attempts):
        U = np.round(np.random.randn(n, m, target_rank)).astype(int)
        V = np.round(np.random.randn(m, p, target_rank)).astype(int)
        
        W_solution = np.zeros((p, n, target_rank))
        valid = True
        
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i, j, c] * V[j, k, c]
                
                b = np.ones(m)
                
                try:
                    x, residuals, rank_A, s = np.linalg.lstsq(A, b, rcond=None)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                two_x = 2 * x
                rounded_two_x = np.round(two_x)
                if not np.allclose(two_x, rounded_two_x, atol=1e-6):
                    valid = False
                    break
                
                W_solution[k, i, :] = x
        
        if not valid:
            continue
            
        if _verify_fast(U.tolist(), V.tolist(), W_solution.tolist(), n, m, p):
            return U.tolist(), V.tolist(), W_solution.tolist(), target_rank
            
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140
    
    rng = random.Random(42)
    np.random.seed(42)
    
    # Baseline: schoolbook (rank 48)
    U, V, W, rank = _schoolbook(n, m, p)
    best = (U, V, W, rank)
    best_rank = rank
    
    # Try hybrid Strassen approach (rank 22)
    try:
        U_h, V_h, W_h, rank_h = _hybrid_strassen_3x4x4()
        if _verify_fast(U_h.tolist(), V_h.tolist(), W_h.tolist(), n, m, p):
            if rank_h < best_rank:
                best = (U_h.tolist(), V_h.tolist(), W_h.tolist(), rank_h)
                best_rank = rank_h
    except Exception:
        pass
    
    # Theoretical lower bound
    min_possible = max(n * p, m * p, n * m)  # max(12, 16, 12) = 16
    
    # Try random search from lower bound up to current best
    for target_rank in range(min_possible, best_rank):
        if time.time() - start_time > deadline - 10:
            break
        
        max_attempts = 300 if target_rank < 25 else 100
        
        result = _try_random_uv_solve_w(n, m, p, target_rank, rng, max_attempts)
        if result is not None:
            U_new, V_new, W_new, r_new = result
            if r_new < best_rank:
                best = (U_new, V_new, W_new, r_new)
                best_rank = r_new
                break
    
    # Try Gaussian random for remaining ranks
    if time.time() - start_time < deadline - 10:
        for target_rank in range(min_possible, best_rank):
            if time.time() - start_time > deadline - 5:
                break
            result = _try_gaussian_random(n, m, p, target_rank, rng, max_attempts=50)
            if result is not None:
                U_new, V_new, W_new, r_new = result
                if r_new < best_rank:
                    best = (U_new, V_new, W_new, r_new)
                    best_rank = r_new
                    break
    
    U_best, V_best, W_best, _ = best
    
    if time.time() - start_time > 140:
        U, V, W, rank = _schoolbook(n, m, p)
        U_best, V_best, W_best = U, V, W
    
    return {"U": U_best, "V": V_best, "W": W_best}
# EVOLVE-BLOCK-END
