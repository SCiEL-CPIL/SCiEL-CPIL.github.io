import numpy as np
import time
from itertools import combinations

def construct():
    n, m, p = 2, 3, 9
    deadline = time.time() + 145
    
    # Build the rank-50 decomposition using Strassen-like 2x2x2 blocks
    rank = 50
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    u_a, u_b, u_c, u_d = 0, 1, 3, 4
    
    # 4 blocks of 7 columns each for k=0..7
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        v_e = k0
        v_f = k1
        v_g = p + k0
        v_h = p + k1
        
        # Strassen's 7 products for 2x2 matrix multiplication
        # Product 1: (a11+a22)(b11+b22)
        U[u_a][col] += 1
        U[u_d][col] += 1
        V[v_e][col] += 1
        V[v_h][col] += 1
        
        # Product 2: (a21+a22)b11
        U[u_c][col+1] += 1
        U[u_d][col+1] += 1
        V[v_e][col+1] += 1
        
        # Product 3: a11(b21-b11)
        U[u_a][col+2] += 1
        V[v_f][col+2] += 1
        V[v_h][col+2] -= 1
        
        # Product 4: a22(b12-b22)
        U[u_d][col+3] += 1
        V[v_g][col+3] += 1
        V[v_e][col+3] -= 1
        
        # Product 5: (a11+a12)b22
        U[u_a][col+4] += 1
        U[u_b][col+4] += 1
        V[v_h][col+4] += 1
        
        # Product 6: (a21-a11)(b11+b12)
        U[u_c][col+5] += 1
        U[u_a][col+5] -= 1
        V[v_e][col+5] += 1
        V[v_f][col+5] += 1
        
        # Product 7: (a12-a22)(b21+b22)
        U[u_b][col+6] += 1
        U[u_d][col+6] -= 1
        V[v_g][col+6] += 1
        V[v_h][col+6] += 1
        
        # W factors for the 2x2 output block
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # C[0,0] = p1 - p3 - p4 + p7
        W[w_00][col] += 1
        W[w_00][col+2] -= 1
        W[w_00][col+3] -= 1
        W[w_00][col+6] += 1
        
        # C[0,1] = p5 + p7
        W[w_01][col+4] += 1
        W[w_01][col+6] += 1
        
        # C[1,0] = p2 + p4
        W[w_10][col+1] += 1
        W[w_10][col+3] += 1
        
        # C[1,1] = p1 - p2 - p5 + p6
        W[w_11][col] += 1
        W[w_11][col+1] -= 1
        W[w_11][col+4] -= 1
        W[w_11][col+5] += 1
        
        col += 7
    
    # k=8 block (4 columns for 1x2 output)
    k8 = 8
    v_e = k8
    v_g = p + k8
    
    # Direct multiplication for the remaining column
    U[u_a][col] += 1
    V[v_e][col] += 1
    U[u_b][col+1] += 1
    V[v_g][col+1] += 1
    U[u_c][col+2] += 1
    V[v_e][col+2] += 1
    U[u_d][col+3] += 1
    V[v_g][col+3] += 1
    
    w_08 = k8 * n + 0
    w_18 = k8 * n + 1
    
    W[w_08][col] += 1
    W[w_08][col+1] += 1
    W[w_18][col+2] += 1
    W[w_18][col+3] += 1
    
    col += 4
    
    # Row augmentation for a=0 and a=1 (18 columns)
    u_ar0, u_ar1 = 2, 5
    for k in range(9):
        v_b = 2 * p + k
        U[u_ar0][col] += 1
        V[v_b][col] += 1
        W[k * n + 0][col] += 1
        U[u_ar1][col+1] += 1
        V[v_b][col+1] += 1
        W[k * n + 1][col+1] += 1
        col += 2
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    if time.time() > deadline - 5:
        return {"U": U, "V": V, "W": W}
    
    # Build the target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n*m):
        ai = i // m
        am = i % m
        for j in range(m*p):
            bj = j // p
            bm = j % p
            if am == bj:
                k_idx = bm * n + ai
                T[i, j, k_idx] = 1.0
    
    # Verify current decomposition
    T_decomp = np.einsum('ic,jc,kc->ijk', U_np, V_np, W_np)
    residual = np.linalg.norm(T_decomp - T)
    
    if residual > 1e-10:
        return {"U": U, "V": V, "W": W}
    
    # Build the matrix representation for rank analysis
    nm, mp, pn = n*m, m*p, p*n
    total_size = nm * mp * pn
    
    mat = np.zeros((total_size, rank))
    for c in range(rank):
        mat[:, c] = np.kron(np.kron(U_np[:, c], V_np[:, c]), W_np[:, c])
    
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U = U_np
    best_V = V_np
    best_W = W_np
    
    # SVD to find numerical rank
    try:
        _, s, Vt = np.linalg.svd(mat, full_matrices=False)
        tol = s[0] * max(nm, mp, pn) * 1e-10
        num_rank = int(np.sum(s > tol))
        
        if num_rank < rank:
            # Try removing columns one at a time
            for drop_col in range(rank):
                if time.time() > deadline - 20:
                    break
                    
                U_new = np.delete(U_np, drop_col, axis=1)
                V_new = np.delete(V_np, drop_col, axis=1)
                W_new = np.delete(W_np, drop_col, axis=1)
                
                T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                res = np.linalg.norm(T_new - T)
                
                if res < 1e-10:
                    new_rank = U_new.shape[1]
                    if new_rank < best_rank:
                        best_rank = new_rank
                        best_U = U_new.copy()
                        best_V = V_new.copy()
                        best_W = W_new.copy()
            
            # Try removing pairs of columns
            if best_rank < rank and time.time() < deadline - 10:
                for c1 in range(rank):
                    if time.time() > deadline - 10:
                        break
                    for c2 in range(c1 + 1, rank):
                        if time.time() > deadline - 10:
                            break
                        cols_to_drop = sorted([c1, c2])
                        U_new = np.delete(U_np, cols_to_drop, axis=1)
                        V_new = np.delete(V_np, cols_to_drop, axis=1)
                        W_new = np.delete(W_np, cols_to_drop, axis=1)
                        
                        T_new = np.einsum('ic,jc,kc->ijk', U_new, V_new, W_new)
                        res = np.linalg.norm(T_new - T)
                        
                        if res < 1e-10:
                            new_rank = U_new.shape[1]
                            if new_rank < best_rank:
                                best_rank = new_rank
                                best_U = U_new.copy()
                                best_V = V_new.copy()
                                best_W = W_new.copy()
    except Exception:
        pass
    
    if best_rank < rank:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fall back to rank-50
    return {"U": U, "V": V, "W": W}
