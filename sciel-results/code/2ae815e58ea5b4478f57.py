import time
import numpy as np

DIMENSIONS = [2, 4, 8]
_START_TIME = time.time()
_DEADLINE = 120.0  # Safety margin before 150s kill

def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=8
    
    # Strassen-based decomposition: 4 k-groups × 14 terms each = 56 rank
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 rows
    V = [[0] * rank for _ in range(m * p)]  # 32 rows
    W = [[0] * rank for _ in range(p * n)]  # 16 rows
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, col_idx):
        for (i, j), val in u_coeffs.items():
            if val != 0:
                U[i*m + j][col_idx] += val
        for (j, k), val in v_coeffs.items():
            if val != 0:
                V[j*p + k][col_idx] += val
        for (k, i), val in w_coeffs.items():
            if val != 0:
                W[k*n + i][col_idx] += val

    col = 0
    
    for k0 in [0, 2, 4, 6]:
        # Part 1: A0 (cols 0,1) * B0 (rows 0,1, cols k0,k0+1)
        # Strassen 7 terms for 2x2 * 2x2
        add_term({(0,0):1,(1,1):1}, {(0,k0):1,(1,k0+1):1}, {(k0,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,0):1,(1,1):1}, {(0,k0):1}, {(k0,1):1,(k0+1,1):-1}, col); col+=1
        add_term({(0,0):1}, {(0,k0+1):1,(1,k0+1):-1}, {(k0+1,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,1):1}, {(1,k0):1,(0,k0):-1}, {(k0,0):1,(k0,1):1}, col); col+=1
        add_term({(0,0):1,(0,1):1}, {(1,k0+1):1}, {(k0,0):-1,(k0+1,0):1}, col); col+=1
        add_term({(1,0):1,(0,0):-1}, {(0,k0):1,(0,k0+1):1}, {(k0+1,1):1}, col); col+=1
        add_term({(0,1):1,(1,1):-1}, {(1,k0):1,(1,k0+1):1}, {(k0,0):1}, col); col+=1
        
        # Part 2: A1 (cols 2,3) * B1 (rows 2,3, cols k0,k0+1)
        # Strassen 7 terms for 2x2 * 2x2
        add_term({(0,2):1,(1,3):1}, {(2,k0):1,(3,k0+1):1}, {(k0,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,2):1,(1,3):1}, {(2,k0):1}, {(k0,1):1,(k0+1,1):-1}, col); col+=1
        add_term({(0,2):1}, {(2,k0+1):1,(3,k0+1):-1}, {(k0+1,0):1,(k0+1,1):1}, col); col+=1
        add_term({(1,3):1}, {(3,k0):1,(2,k0):-1}, {(k0,0):1,(k0,1):1}, col); col+=1
        add_term({(0,2):1,(0,3):1}, {(3,k0+1):1}, {(k0,0):-1,(k0+1,0):1}, col); col+=1
        add_term({(1,2):1,(0,2):-1}, {(2,k0):1,(2,k0+1):1}, {(k0+1,1):1}, col); col+=1
        add_term({(0,3):1,(1,3):-1}, {(3,k0):1,(3,k0+1):1}, {(k0,0):1}, col); col+=1
    
    # Attempt rank reduction via column elimination
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    T_target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                T_target[u, v, w] = 1.0
    
    # Try to eliminate columns by expressing them as linear combinations
    # of other columns while preserving the tensor equation
    # T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # For each column r, check if it can be expressed as sum of other columns
    # with appropriate scaling of U, V, W
    current_rank = rank
    U_list = [U_np[:, r].copy() for r in range(rank)]
    V_list = [V_np[:, r].copy() for r in range(rank)]
    W_list = [W_np[:, r].copy() for r in range(rank)]
    active = [True] * rank
    
    # Greedy elimination: try to remove each column
    for r in range(rank):
        if _time_left() < 30:
            break
        if not active[r]:
            continue
        # Try to express column r as linear combination of other active columns
        # We need: U_r ⊗ V_r ⊗ W_r = sum_s c_s * U_s' ⊗ V_s' ⊗ W_s'
        # This is complex. Instead, try simple case: if two columns have
        # proportional U vectors, we might combine them.
        # For now, skip complex elimination and just validate
        pass
    
    # Reconstruct from active columns
    U_final = np.column_stack([U_list[r] for r in range(rank) if active[r]])
    V_final = np.column_stack([V_list[r] for r in range(rank) if active[r]])
    W_final = np.column_stack([W_list[r] for r in range(rank) if active[r]])
    
    T_recon = np.einsum('ur,vr,wr->uvw', U_final, V_final, W_final)
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i*m+j
                    v = j*p+k
                    w = k*n+i
                    U[u][col] = 1
                    V[v][col] = 1
                    W[w][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Convert back to lists
    U_out = U_final.tolist()
    V_out = V_final.tolist()
    W_out = W_final.tolist()
    
    return {
        "U": U_out,
        "V": V_out,
        "W": W_out
    }

if __name__ == "__main__":
    result = construct()
    print("Rank:", len(result['U'][0]))
