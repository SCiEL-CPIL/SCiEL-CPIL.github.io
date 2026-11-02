import numpy as np
import time
from itertools import product, combinations

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin

    def build_schoolbook(n, m, p):
        U = np.zeros((n * m, n * m * p), dtype=float)
        V = np.zeros((m * p, n * m * p), dtype=float)
        W = np.zeros((p * n, n * m * p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    col = i * m * p + j * p + k
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
        return U, V, W

    def verify(U_mat, V_mat, W_mat, n, m, p, tol=1e-8):
        """Verify that U,V,W exactly decompose the multiplication tensor."""
        U_mat = np.asarray(U_mat, dtype=float)
        V_mat = np.asarray(V_mat, dtype=float)
        W_mat = np.asarray(W_mat, dtype=float)
        r = U_mat.shape[1]
        if U_mat.shape[0] != n * m or V_mat.shape[0] != m * p or W_mat.shape[0] != p * n:
            return False
        if U_mat.shape[1] != r or V_mat.shape[1] != r or W_mat.shape[1] != r:
            return False
        
        # Check required entries are 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val - 1) > tol:
                        return False
        
        # Check all other entries are 0
        for a in range(n * m):
            for b in range(m * p):
                for c in range(p * n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    kk = c // n
                    ii = c % n
                    if i == ii and j == j2 and k == kk:
                        continue
                    val = np.dot(U_mat[a, :], np.multiply(V_mat[b, :], W_mat[c, :]))
                    if abs(val) > tol:
                        return False
        return True

    # Build schoolbook for the full 3x3x7 tensor
    U_school, V_school, W_school = build_schoolbook(n, m, p)
    rank_schoolbook = n * m * p  # 63
    
    # Verify schoolbook
    if not verify(U_school, V_school, W_school, n, m, p):
        raise ValueError("Schoolbook verification failed")

    best_U, best_V, best_W = U_school.copy(), V_school.copy(), W_school.copy()
    best_rank = rank_schoolbook

    # Strategy: Decompose into 3x3x3 (k=0,1,2) and 3x3x4 (k=3,4,5,6)
    # Try to find better decompositions for each part
    
    def find_decomposition(n, m, p, target_rank, max_time=10, seed=None):
        """
        Try to find a rank-target_rank decomposition for n x m * p multiplication.
        Returns (U, V, W, rank) or None.
        """
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        np.random.seed(seed)
        
        start = time.time()
        best_rank = n * m * p
        best_U, best_V, best_W = None, None, None
        
        max_stuck = 50
        
        while time.time() - start < max_time:
            # Restart with schoolbook
            U_cur, V_cur, W_cur = build_schoolbook(n, m, p)
            cur_rank = n * m * p
            
            stuck_counter = 0
            
            while cur_rank > target_rank and time.time() - start < max_time:
                if stuck_counter > max_stuck:
                    break
                
                # Pick a random column
                col_to_try = np.random.randint(0, cur_rank)
                
                # Create mask
                mask = np.ones(cur_rank, dtype=bool)
                mask[col_to_try] = False
                
                U_cand = U_cur[:, mask]
                V_cand = V_cur[:, mask]
                W_cand = W_cur[:, mask]
                
                if verify(U_cand, V_cand, W_cand, n, m, p):
                    U_cur = U_cand
                    V_cur = V_cand
                    W_cur = W_cand
                    cur_rank -= 1
                    stuck_counter = 0
                    if cur_rank < best_rank:
                        best_rank = cur_rank
                        best_U = U_cur.copy()
                        best_V = V_cur.copy()
                        best_W = W_cur.copy()
                else:
                    stuck_counter += 1
            
            if cur_rank <= target_rank:
                break
        
        if best_U is not None:
            return best_U, best_V, best_W, best_rank
        return None

    # Try to find 3x3x3 decomposition (target rank 23)
    t_start = time.time()
    if time.time() - start_time < deadline:
        result_333 = find_decomposition(3, 3, 3, 23, max_time=min(30, deadline - (time.time() - start_time)), seed=42)
        if result_333:
            U3, V3, W3, rank_333 = result_333
            print(f"Found 3x3x3 decomposition with rank {rank_333}")
        else:
            U3, V3, W3 = build_schoolbook(3, 3, 3)
            rank_333 = 27
            print("Using schoolbook 3x3x3 rank 27")
    else:
        U3, V3, W3 = build_schoolbook(3, 3, 3)
        rank_333 = 27

    # Try to find 3x3x4 decomposition (target rank 30)
    t_start = time.time()
    remaining = deadline - (time.time() - start_time)
    if remaining > 10:
        result_334 = find_decomposition(3, 3, 4, 30, max_time=min(40, remaining - 5), seed=123)
        if result_334:
            U4, V4, W4, rank_334 = result_334
            print(f"Found 3x3x4 decomposition with rank {rank_334}")
        else:
            U4, V4, W4 = build_schoolbook(3, 3, 4)
            rank_334 = 36
            print("Using schoolbook 3x3x4 rank 36")
    else:
        U4, V4, W4 = build_schoolbook(3, 3, 4)
        rank_334 = 36

    # Combine the decompositions
    total_rank = rank_333 + rank_334
    
    U_full = np.zeros((3 * 3, total_rank), dtype=float)
    V_full = np.zeros((3 * 7, total_rank), dtype=float)
    W_full = np.zeros((7 * 3, total_rank), dtype=float)
    
    # Copy 3x3x3 part (k=0,1,2)
    U_full[:, :rank_333] = U3
    V_full[0:9, :rank_333] = V3
    W_full[0:9, :rank_333] = W3
    
    # Copy 3x3x4 part (k=3,4,5,6)
    U_full[:, rank_333:] = U4
    
    for j in range(3):
        for l in range(4):
            k_local = l
            k_full = 3 + l
            row_v4 = j * 4 + l
            row_v_full = j * 7 + k_full
            V_full[row_v_full, rank_333:] = V4[row_v4, :]
    
    for l in range(4):
        k_local = l
        k_full = 3 + l
        for i in range(3):
            row_w4 = l * 3 + i
            row_w_full = k_full * 3 + i
            W_full[row_w_full, rank_333:] = W4[row_w4, :]

    # Verify the full decomposition
    if verify(U_full, V_full, W_full, 3, 3, 7):
        best_U, best_V, best_W = U_full, V_full, W_full
        best_rank = total_rank
    else:
        best_U, best_V, best_W = U_school, V_school, W_school
        best_rank = rank_schoolbook

    # Convert to half-integers if possible
    def to_half_integer(mat):
        mat2 = mat * 2
        rounded = np.round(mat2)
        if np.allclose(mat2, rounded, atol=1e-9):
            return (rounded / 2).tolist()
        return mat.tolist()

    return {"U": to_half_integer(best_U), "V": to_half_integer(best_V), "W": to_half_integer(best_W)}
