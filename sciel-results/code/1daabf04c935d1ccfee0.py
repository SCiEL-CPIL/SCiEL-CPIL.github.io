"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

    n, m, p = DIMENSIONS
    # We exploit the structure: M_{2,3,9} = M_{2,3,3} \oplus M_{2,3,3} \oplus M_{2,3,3}
    # So we find a rank-minimal decomposition for M_{2,3,3} and replicate it 3 times.
    
    # M_{2,3,3} dimensions: n_sub=2, m_sub=3, p_sub=3
    n_sub, m_sub, p_sub = 2, 3, 3
    
    def schoolbook_sub():
        rank = n_sub * m_sub * p_sub
        U = [[0] * rank for _ in range(n_sub * m_sub)]
        V = [[0] * rank for _ in range(m_sub * p_sub)]
        W = [[0] * rank for _ in range(p_sub * n_sub)]
        column = 0
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[i * m_sub + j][column] = 1
                    V[j * p_sub + k][column] = 1
                    W[k * n_sub + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}

    def verify_sub(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    u = U_arr[i * m_sub + j, :]
                    v = V_arr[j * p_sub + k, :]
                    w = W_arr[k * n_sub + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    def snap_to_half_int(arr):
        return [[int(round(x * 2)) / 2.0 for x in row] for row in arr]

    def try_random_search_sub(rank, attempts=200):
        if time.time() > deadline - 10:
            return None
        
        for _ in range(attempts):
            if time.time() > deadline - 10:
                break
            
            U_mat = np.random.randn(n_sub * m_sub, rank)
            V_mat = np.random.randn(m_sub * p_sub, rank)
            W_mat = np.random.randn(p_sub * n_sub, rank)
            
            for it in range(100):
                if time.time() > deadline - 10:
                    return None
                
                # Solve for U
                U_new = np.zeros((n_sub * m_sub, rank))
                for i in range(n_sub):
                    for j in range(m_sub):
                        A = np.zeros(rank)
                        for k in range(p_sub):
                            A += V_mat[j * p_sub + k, :] * W_mat[k * n_sub + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            U_new[i * m_sub + j, :] = A / norm_sq
                U_mat = U_new
                
                # Solve for V
                V_new = np.zeros((m_sub * p_sub, rank))
                for j in range(m_sub):
                    for k in range(p_sub):
                        A = np.zeros(rank)
                        for i in range(n_sub):
                            A += U_mat[i * m_sub + j, :] * W_mat[k * n_sub + i, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            V_new[j * p_sub + k, :] = A / norm_sq
                V_mat = V_new
                
                # Solve for W
                W_new = np.zeros((p_sub * n_sub, rank))
                for k in range(p_sub):
                    for i in range(n_sub):
                        A = np.zeros(rank)
                        for j in range(m_sub):
                            A += U_mat[i * m_sub + j, :] * V_mat[j * p_sub + k, :]
                        norm_sq = np.sum(A * A)
                        if norm_sq > 1e-9:
                            W_new[k * n_sub + i, :] = A / norm_sq
                W_mat = W_new
                
                loss = 0.0
                for i in range(n_sub):
                    for j in range(m_sub):
                        for k in range(p_sub):
                            u = U_mat[i * m_sub + j, :]
                            v = V_mat[j * p_sub + k, :]
                            w = W_mat[k * n_sub + i, :]
                            val = np.dot(u * v, w)
                            loss += (val - 1.0) ** 2
                
                if loss < 1e-6:
                    U_snap = snap_to_half_int(U_mat.tolist())
                    V_snap = snap_to_half_int(V_mat.tolist())
                    W_snap = snap_to_half_int(W_mat.tolist())
                    
                    if verify_sub(U_snap, V_snap, W_snap):
                        return {"U": U_snap, "V": V_snap, "W": W_snap}
        
        return None

    # Search for minimal rank for M_{2,3,3}
    # Schoolbook is 18. Try 17, 16, ...
    best_sub = None
    for rank in range(17, 19):  # Try 17 and 18
        if time.time() > deadline - 15:
            break
        res = try_random_search_sub(rank, attempts=50)
        if res is not None:
            best_sub = res
            break
    
    if best_sub is None:
        # Fallback to schoolbook for sub-problem
        best_sub = schoolbook_sub()
    
    # Construct full M_{2,3,9} from 3 copies of M_{2,3,3}
    rank_sub = len(best_sub['U'][0])
    rank_full = rank_sub * 3
    
    U_full = [[0] * rank_full for _ in range(n * m)]
    V_full = [[0] * rank_full for _ in range(m * p)]
    W_full = [[0] * rank_full for _ in range(p * n)]
    
    # The 9 columns of B are split into 3 blocks of 3.
    # Block b (0, 1, 2) corresponds to columns 3*b, 3*b+1, 3*b+2.
    # U is the same for all blocks (since A is the same).
    # V is different for each block.
    # W is different for each block.
    
    for b in range(3):
        offset = b * rank_sub
        for i in range(n_sub):
            for j in range(m_sub):
                for r in range(rank_sub):
                    # In the direct sum, the U matrices are the same for each block,
                    # but mapped to different columns in the full rank space.
                    U_full[i * m + j][offset + r] = best_sub['U'][i * m_sub + j][r]
                    
                    for k_sub in range(p_sub):
                        k_global = b * p_sub + k_sub
                        V_full[j * p + k_global][offset + r] = best_sub['V'][j * p_sub + k_sub][r]
                        W_full[k_global * n + i][offset + r] = best_sub['W'][k_sub * n_sub + i][r]

    # Verify full decomposition
    def verify_full(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = U_arr[i * m + j, :]
                    v = V_arr[j * p + k, :]
                    w = W_arr[k * n + i, :]
                    val = np.dot(u * v, w)
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify_full(U_full, V_full, W_full):
        # Fallback to schoolbook full
        rank = n * m * p
        U_fb = [[0] * rank for _ in range(n * m)]
        V_fb = [[0] * rank for _ in range(m * p)]
        W_fb = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j][column] = 1
                    V_fb[j * p + k][column] = 1
                    W_fb[k * n + i][column] = 1
                    column += 1
        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U_full, "V": V_full, "W": W_full}
# EVOLVE-BLOCK-END
