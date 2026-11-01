import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=5
    start_time = time.time()
    deadline = start_time + 110
    
    # Target rank 16
    rank = 16
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val=1):
        nonlocal col
        U[u_idx, col] += val
        V[v_idx, col] += val
        W[w_idx, col] += val
        col += 1

    # Helper to add a Strassen 2x2x2 block
    # i_off: offset for i (0 or 1)
    # j_off: offset for j (0 or 1)
    # k_off: offset for k (0, 1, 2, 3)
    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # A indices: (i_off, j_off), (i_off, j_off+1), (i_off+1, j_off), (i_off+1, j_off+1)
        # B indices: (j_off, k_off), (j_off, k_off+1), (j_off+1, k_off), (j_off+1, k_off+1)
        # C indices: (i_off, k_off), (i_off, k_off+1), (i_off+1, k_off), (i_off+1, k_off+1)
        
        a_idx = (i_off, j_off)
        b_idx = (i_off, j_off + 1)
        c_idx = (i_off + 1, j_off)
        d_idx = (i_off + 1, j_off + 1)
        
        e_idx = (j_off, k_off)
        f_idx = (j_off, k_off + 1)
        g_idx = (j_off + 1, k_off)
        h_idx = (j_off + 1, k_off + 1)
        
        c00_idx = (i_off, k_off)
        c01_idx = (i_off, k_off + 1)
        c10_idx = (i_off + 1, k_off)
        c11_idx = (i_off + 1, k_off + 1)
        
        def u_idx(i, j): return i * m + j
        def v_idx(j, k): return j * p + k
        def w_idx(k, i): return k * n + i
        
        # M1 = (a+d)(e+h)
        U[u_idx(*a_idx), col] = 1
        U[u_idx(*d_idx), col] = 1
        V[v_idx(*e_idx), col] = 1
        V[v_idx(*h_idx), col] = 1
        W[w_idx(*c00_idx), col] = 1
        W[w_idx(*c11_idx), col] = 1
        col += 1
        
        # M2 = (c+d)e
        U[u_idx(*c_idx), col] = 1
        U[u_idx(*d_idx), col] = 1
        V[v_idx(*e_idx), col] = 1
        W[w_idx(*c10_idx), col] = 1
        W[w_idx(*c11_idx), col] = -1
        col += 1
        
        # M3 = a(f-g)
        U[u_idx(*a_idx), col] = 1
        V[v_idx(*f_idx), col] = 1
        V[v_idx(*g_idx), col] = -1
        W[w_idx(*c01_idx), col] = 1
        W[w_idx(*c11_idx), col] = 1
        col += 1
        
        # M4 = d(g-e)
        U[u_idx(*d_idx), col] = 1
        V[v_idx(*g_idx), col] = 1
        V[v_idx(*e_idx), col] = -1
        W[w_idx(*c00_idx), col] = 1
        W[w_idx(*c10_idx), col] = 1
        col += 1
        
        # M5 = (a+b)h
        U[u_idx(*a_idx), col] = 1
        U[u_idx(*b_idx), col] = 1
        V[v_idx(*h_idx), col] = 1
        W[w_idx(*c00_idx), col] = -1
        W[w_idx(*c01_idx), col] = 1
        col += 1
        
        # M6 = (c-a)(e+h)
        U[u_idx(*c_idx), col] = 1
        U[u_idx(*a_idx), col] = -1
        V[v_idx(*e_idx), col] = 1
        V[v_idx(*h_idx), col] = 1
        W[w_idx(*c10_idx), col] = 1
        W[w_idx(*c11_idx), col] = 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[u_idx(*b_idx), col] = 1
        U[u_idx(*d_idx), col] = -1
        V[v_idx(*g_idx), col] = 1
        V[v_idx(*h_idx), col] = 1
        W[w_idx(*c00_idx), col] = 1
        W[w_idx(*c11_idx), col] = 1
        col += 1

    # Part 1: j=2, all k (Rank 1)
    # U[i, 2], V[2, k], W[k, i]
    for i in range(n):
        U[i * m + 2, col] = 1
    for k in range(p):
        V[2 * p + k, col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i, col] = 1
    col += 1
    
    # Part 2: j in {0,1}, k in {0,1} (Rank 7)
    add_strassen(0, 0, 0)
    
    # Part 3: j in {0,1}, k in {2,3} (Rank 7)
    add_strassen(0, 0, 2)
    
    # Part 4: j in {0,1}, k=4 (Rank 1)
    for i in range(n):
        for j in range(2):
            U[i * m + j, col] = 1
    for j in range(2):
        V[j * p + 4, col] = 1
    for i in range(n):
        W[4 * n + i, col] = 1
    col += 1
    
    assert col == rank, f"Column count mismatch: {col} != {rank}"
    
    # Verify
    def verify(U_mat, V_mat, W_mat):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    coeff = 0
                    for r in range(U_mat.shape[1]):
                        coeff += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(coeff - 1) > 1e-10:
                        return False
        return True

    if not verify(U, V, W):
        # Fallback to schoolbook
        U = np.zeros((n*m, n*m*p))
        V = np.zeros((m*p, n*m*p))
        W = np.zeros((p*n, n*m*p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    idx = i*m*p + j*p + k
                    U[i*m+j, idx] = 1
                    V[j*p+k, idx] = 1
                    W[k*n+i, idx] = 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    n, m, p = 2, 3, 5
    U = np.array(res['U'])
    V = np.array(res['V'])
    W = np.array(res['W'])
    rank = U.shape[1]
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                coeff = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(rank))
                if abs(coeff - 1) > 1e-10:
                    ok = False
                    print(f"FAIL at ({i},{j},{k}): {coeff}")
    print("Valid:", ok)
