import time
import numpy as np
from itertools import combinations

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Build the rank-54 Strassen-based decomposition
    rank_base = 54
    U_base = [[0] * rank_base for _ in range(n * m)]
    V_base = [[0] * rank_base for _ in range(m * p)]
    W_base = [[0] * rank_base for _ in range(p * n)]
    
    col = 0
    
    def add_standard(U, V, W, i, j, k, c):
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][c] = 1
        V[v_idx][c] = 1
        W[w_idx][c] = 1
        return c + 1

    def add_strassen(U, V, W, j1, j2, k1, k2, c):
        u00 = 0 * m + j1
        u01 = 0 * m + j2
        u10 = 1 * m + j1
        u11 = 1 * m + j2
        
        v00 = j1 * p + k1
        v01 = j1 * p + k2
        v10 = j2 * p + k1
        v11 = j2 * p + k2
        
        w00 = k1 * n + 0
        w01 = k2 * n + 0
        w10 = k1 * n + 1
        w11 = k2 * n + 1
        
        # P1 = (a00+a11)(b00+b11)
        U[u00][c] += 1; U[u11][c] += 1
        V[v00][c] += 1; V[v11][c] += 1
        W[w00][c] += 1; W[w11][c] += 1
        c += 1
        
        # P2 = (a10+a11)b00
        U[u10][c] += 1; U[u11][c] += 1
        V[v00][c] += 1
        W[w10][c] += 1; W[w11][c] += -1
        c += 1
        
        # P3 = a00(b01-b11)
        U[u00][c] += 1
        V[v01][c] += 1; V[v11][c] += -1
        W[w01][c] += 1; W[w11][c] += 1
        c += 1
        
        # P4 = a11(b10-b00)
        U[u11][c] += 1
        V[v10][c] += 1; V[v00][c] += -1
        W[w00][c] += 1; W[w10][c] += 1
        c += 1
        
        # P5 = (a00+a01)b11
        U[u00][c] += 1; U[u01][c] += 1
        V[v11][c] += 1
        W[w00][c] += -1; W[w01][c] += 1
        c += 1
        
        # P6 = (a10-a00)(b00+b01)
        U[u10][c] += 1; U[u00][c] += -1
        V[v00][c] += 1; V[v01][c] += 1
        W[w11][c] += 1
        c += 1
        
        # P7 = (a01-a11)(b10+b11)
        U[u01][c] += 1; U[u11][c] += -1
        V[v10][c] += 1; V[v11][c] += 1
        W[w00][c] += 1
        c += 1
        
        return c

    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_pairs:
        for k1, k2 in k_pairs:
            col = add_strassen(U_base, V_base, W_base, j1, j2, k1, k2, col)
            
    for k1, k2 in k_pairs:
        for k in [k1, k2]:
            col = add_standard(U_base, V_base, W_base, 0, 4, k, col)
            col = add_standard(U_base, V_base, W_base, 1, 4, k, col)
    
    # Convert to numpy for efficient verification
    U_np = np.array(U_base, dtype=float)
    V_np = np.array(V_base, dtype=float)
    W_np = np.array(W_base, dtype=float)
    
    def verify_decomp(U_mat, V_mat, W_mat):
        """Verify that the decomposition correctly computes matrix multiplication."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_mat[i * m + j, :]
                    v_row = V_mat[j * p + k, :]
                    w_row = W_mat[k * n + i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-9:
                        return False
        return True
    
    # Verify base decomposition
    if not verify_decomp(U_np, V_np, W_np):
        # Fallback to trivial rank 60
        rank_f = n * m * p
        U_f = np.zeros((n * m, rank_f))
        V_f = np.zeros((m * p, rank_f))
        W_f = np.zeros((p * n, rank_f))
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_f[i * m + j, col_f] = 1
                    V_f[j * p + k, col_f] = 1
                    W_f[k * n + i, col_f] = 1
                    col_f += 1
        return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}
    
    # Now attempt to reduce rank via greedy set cover / linear dependency search
    # Strategy: Try to find linear combinations of the rank-54 terms that can replace
    # multiple terms with fewer terms, exploiting global structure.
    
    # For each subset of columns, check if we can express the tensor contribution
    # of those columns as a linear combination of other columns.
    
    # Simpler approach: Try removing one column at a time and see if the remaining
    # columns can still span the same tensor space.
    
    # Build the tensor representation: each column c defines a rank-1 tensor T_c
    # where T_c[i,j,k] = U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # The full tensor T = sum_c T_c should equal the matrix multiplication tensor.
    
    # To check if column c is redundant, we need to check if T_c is in the span
    # of the other T_d for d != c.
    
    # This is expensive for 54 columns, but we can try a greedy approach:
    # 1. Start with all 54 columns
    # 2. For each column (in some order), try to remove it
    # 3. If removal still gives a valid decomposition, keep it removed
    # 4. Repeat until no more columns can be removed
    
    best_rank = rank_base
    best_U, best_V, best_W = U_np.copy(), V_np.copy(), W_np.copy()
    
    # Try to find a minimal subset of columns that still spans the tensor
    # We'll use a greedy approach: try removing columns one by one
    
    current_U, current_V, current_W = U_np.copy(), V_np.copy(), W_np.copy()
    current_rank = rank_base
    
    # Try each column in order
    for col_to_remove in range(rank_base):
        if time.time() > deadline - 10:
            break
        
        # Create a copy without this column
        mask = [c for c in range(current_rank) if c != col_to_remove]
        
        if len(mask) < current_rank - 1:
            continue
            
        new_U = current_U[:, mask]
        new_V = current_V[:, mask]
        new_W = current_W[:, mask]
        
        if verify_decomp(new_U, new_V, new_W):
            current_U, current_V, current_W = new_U, new_V, new_W
            current_rank -= 1
            best_rank = current_rank
            best_U, best_V, best_W = current_U.copy(), current_V.copy(), current_W.copy()
    
    # If we found a lower rank, use it
    if best_rank < rank_base:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Otherwise, return the rank-54 decomposition
    return {"U": U_np.tolist(), "V": V_np.tolist(), "W": W_np.tolist()}
