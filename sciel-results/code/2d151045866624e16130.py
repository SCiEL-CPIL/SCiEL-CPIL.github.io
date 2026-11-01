"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    
    # Build Strassen-based rank-35 decomposition
    rank35 = 7 * num_blocks
    U35 = [[0] * rank35 for _ in range(n * m)]
    V35 = [[0] * rank35 for _ in range(m * p)]
    W35 = [[0] * rank35 for _ in range(p * n)]
    
    for b in range(num_blocks):
        base = 7 * b
        col = base
        U35[0][col] = 1; U35[3][col] = 1
        V35[2*b][col] = 1; V35[10+2*b+1][col] = 1
        W35[4*b][col] = 1; W35[4*b+3][col] = 1
        
        col = base + 1
        U35[2][col] = 1; U35[3][col] = 1
        V35[2*b][col] = 1
        W35[4*b+1][col] = 1; W35[4*b+3][col] = -1
        
        col = base + 2
        U35[0][col] = 1
        V35[2*b+1][col] = 1; V35[10+2*b+1][col] = -1
        W35[4*b+2][col] = 1; W35[4*b+3][col] = 1
        
        col = base + 3
        U35[3][col] = 1
        V35[10+2*b][col] = 1; V35[2*b][col] = -1
        W35[4*b][col] = 1; W35[4*b+1][col] = 1
        
        col = base + 4
        U35[0][col] = 1; U35[1][col] = 1
        V35[10+2*b+1][col] = 1
        W35[4*b][col] = -1; W35[4*b+2][col] = 1
        
        col = base + 5
        U35[2][col] = 1; U35[0][col] = -1
        V35[2*b][col] = 1; V35[2*b+1][col] = 1
        W35[4*b+3][col] = 1
        
        col = base + 6
        U35[1][col] = 1; U35[3][col] = -1
        V35[10+2*b][col] = 1; V35[10+2*b+1][col] = 1
        W35[4*b][col] = 1
    
    # Check for linear dependencies among the 35 rank-1 terms
    # Build the 40x35 matrix where each column is a rank-1 term vector
    T_mat = np.zeros((n * m * p, rank35))
    for r in range(rank35):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i * m * p + j * p + k
                    T_mat[idx, r] = (U35[i*m+j][r] * V35[j*p+k][r] * W35[k*n+i][r])
    
    rank_T = np.linalg.matrix_rank(T_mat)
    
    if rank_T < rank35 and time.time() < deadline:
        # Find null space and reduce rank
        U_svd, s, Vt = np.linalg.svd(T_mat, full_matrices=True)
        null_dim = rank35 - rank_T
        
        if null_dim > 0:
            null_vecs = Vt[rank_T:, :]
            
            U_work = [row[:] for row in U35]
            V_work = [row[:] for row in V35]
            W_work = [row[:] for row in W35]
            eliminated = set()
            
            for nv_idx in range(null_dim):
                if time.time() > deadline:
                    break
                nv = null_vecs[nv_idx]
                
                # Find pivot term (largest coefficient, not eliminated)
                pivot = -1
                max_abs = 0
                for t in range(rank35):
                    if t not in eliminated and abs(nv[t]) > max_abs:
                        max_abs = abs(nv[t])
                        pivot = t
                
                if pivot < 0 or max_abs < 1e-10:
                    continue
                
                # Express term_pivot in terms of others:
                # term_pivot = -sum_{t != pivot} (nv[t]/nv[pivot]) * term_t
                for t in range(rank35):
                    if t == pivot or t in eliminated:
                        continue
                    coeff = -nv[t] / nv[pivot]
                    if abs(coeff) < 1e-10:
                        continue
                    for i in range(n*m):
                        U_work[i][pivot] += coeff * U35[i][t]
                    for j in range(m*p):
                        V_work[j][pivot] += coeff * V35[j][t]
                    for k in range(p*n):
                        W_work[k][pivot] += coeff * W35[k][t]
                
                eliminated.add(pivot)
            
            keep = [t for t in range(rank35) if t not in eliminated]
            U_final = [[U_work[i][t] for t in keep] for i in range(n*m)]
            V_final = [[V_work[j][t] for t in keep] for j in range(m*p)]
            W_final = [[W_work[k][t] for t in keep] for k in range(p*n)]
            
            return {"U": U_final, "V": V_final, "W": W_final}
    
    return {"U": U35, "V": V35, "W": W35}
# EVOLVE-BLOCK-END
