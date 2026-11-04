import numpy as np
import time
from itertools import product

def construct():
    n, m, p = 3, 3, 6
    rank_school = n * m * p  # 54
    
    # Schoolbook decomposition as fallback
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    def verify(U, V, W):
        n, m, p = 3, 3, 6
        rank = U.shape[1]
        if U.shape != (n*m, rank) or V.shape != (m*p, rank) or W.shape != (p*n, rank):
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True

    if not verify(U_school, V_school, W_school):
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}

    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    best_rank = rank_school
    best_U, best_V, best_W = U_school, V_school, W_school
    
    # Strategy: Formulate as batched matrix-vector product
    # A is 3x3, B is 3x6 (six column vectors)
    # For each column b_k of B, compute A @ b_k
    # Use rank-3 decomposition of 3x3 matrix-vector multiplication:
    # A @ v = 1/2 * [(A + v^T * 1) @ 1 - (A - v^T * 1) @ 1]
    # This gives rank 3 per column, so total rank = 6 * 3 = 18
    
    # Let's construct the rank-18 decomposition
    rank_candidate = 18
    U_cand = np.zeros((n*m, rank_candidate))
    V_cand = np.zeros((m*p, rank_candidate))
    W_cand = np.zeros((p*n, rank_candidate))
    
    # For each of the 6 columns k, and for each of the 3 rank terms:
    # The identity: (A @ v)_i = 1/2 * sum_j (A_ij + v_j) - 1/2 * sum_j (A_ij - v_j)
    # = 1/2 * [sum_j A_ij + sum_j v_j - sum_j A_ij + sum_j v_j]
    # Wait, that's not right. Let me reconsider.
    
    # Actually, the standard rank-3 decomposition for 3x3 matrix-vector:
    # (A @ v)_i = 1/2 * [ (sum_j A_ij) + (sum_j v_j) - (sum_j (A_ij - v_j)) ]
    # No, let's use the known decomposition:
    # A @ v = 1/2 * [ (A + v*1^T) @ 1 - (A - v*1^T) @ 1 ]
    # where 1 is the all-ones vector of size 3
    # (A + v*1^T) @ 1 = A@1 + v*(1^T @ 1) = A@1 + 3*v
    # (A - v*1^T) @ 1 = A@1 - 3*v
    # So A@v = 1/2 * [(A@1 + 3v) - (A@1 - 3v)] = 3v, which is wrong.
    
    # Let me use the correct decomposition. For 3x3 matrix-vector:
    # (A @ v)_i = sum_j A_ij v_j
    # We can write this using 3 rank-1 terms if we use:
    # (A @ v)_i = 1/2 * sum_j [(A_ij + v_j)(1) - (A_ij - v_j)(1)]
    # That doesn't work either.
    
    # The correct approach: use the fact that for 3x3, we can decompose as:
    # (A @ v)_i = 1/2 * [sum_j (A_ij + v_j) - sum_j (A_ij - v_j)]
    # = 1/2 * [sum_j A_ij + sum_j v_j - sum_j A_ij + sum_j v_j]
    # = sum_j v_j, which is still wrong.
    
    # Let me think differently. The tensor for 3x3 matrix-vector has rank 3.
    # One decomposition:
    # (A @ v)_i = sum_j A_ij v_j
    # = 1/2 * [ (sum_j A_ij) * (sum_j v_j) - sum_j (A_ij - 1/3)(v_j - 1/3) ]
    # No, that's not a rank decomposition.
    
    # Actually, for the 3x3 matrix-vector multiplication tensor, the rank is 3.
    # A valid decomposition uses:
    # Term 1: U has 1 at (i,0) for all i, V has A_ij at (j,0), W has v_j at (j,0)
    # That's not rank 1.
    
    # Let me just try to find a rank-18 decomposition by construction.
    # For each column k (0 to 5), we need to compute A @ b_k.
    # Using schoolbook for each column: 3*3 = 9 terms, total 54.
    # To get rank 18, we need 3 terms per column.
    
    # One approach: use the identity for 3x3 matrix-vector with 3 terms.
    # (A @ v)_i = 1/2 * [ (A@1)_i + (1^T v) - ((A - 1/3 * 1*1^T) @ (v - 1/3 * 1)) ]
    # This is getting complicated. Let's try a different approach.
    
    # Try random search for rank 18 with better initialization
    candidate_ranks = [18, 20, 24, 27, 30, 36, 40, 45, 50, 53]
    
    for rank in candidate_ranks:
        if time.time() > deadline:
            break
        if rank >= best_rank:
            continue
            
        # Try multiple random initializations
        max_attempts = 20
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline:
                break
                
            np.random.seed(attempt * 7919 + rank * 31)
            
            # Initialize with sparse random patterns
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            # Each rank term picks random indices
            for r in range(rank):
                ui = np.random.randint(0, n*m)
                vj = np.random.randint(0, m*p)
                wk = np.random.randint(0, p*n)
                U[ui, r] = np.random.choice([-1.0, 1.0])
                V[vj, r] = np.random.choice([-1.0, 1.0])
                W[wk, r] = np.random.choice([-1.0, 1.0])
                
            if verify(U, V, W):
                best_rank = rank
                best_U, best_V, best_W = U, V, W
                found = True
                break
                
        if found:
            continue

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
