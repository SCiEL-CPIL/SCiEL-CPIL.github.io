import time
import numpy as np

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Partitions
    # n=4 -> [2, 2]
    n_parts = [2, 2]
    n_offsets = [0, 2]
    
    # m=5 -> [2, 2, 1]
    m_parts = [2, 2, 1]
    m_offsets = [0, 2, 4]
    
    # p=6 -> [2, 2, 2]
    p_parts = [2, 2, 2]
    p_offsets = [0, 2, 4]
    
    # Pre-allocate global matrices
    # We don't know the exact rank yet, so we can use lists of columns
    # Or estimate max rank.
    # Max rank per block: 7 (2x2x2) or 4 (2x1x2).
    # Max blocks: 2*3*3 = 18.
    # Max total rank: 18 * 7 = 126.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_decomposition(n_off, m_off, p_off, n_size, m_size, p_size, U_loc, V_loc, W_loc):
        """
        Maps local decomposition to global and appends to lists.
        U_loc: (n_size * m_size, rank)
        V_loc: (m_size * p_size, rank)
        W_loc: (p_size * n_size, rank)
        """
        rank_loc = U_loc.shape[1]
        
        for r in range(rank_loc):
            u_col = [0] * (n * m)
            v_col = [0] * (m * p)
            w_col = [0] * (p * n)
            
            # Map U
            for idx_loc in range(n_size * m_size):
                val = U_loc[idx_loc, r]
                if val != 0:
                    i_loc = idx_loc // m_size
                    j_loc = idx_loc % m_size
                    i_glob = n_off + i_loc
                    j_glob = m_off + j_loc
                    idx_glob = i_glob * m + j_glob
                    u_col[idx_glob] = val
            
            # Map V
            for idx_loc in range(m_size * p_size):
                val = V_loc[idx_loc, r]
                if val != 0:
                    j_loc = idx_loc // p_size
                    k_loc = idx_loc % p_size
                    j_glob = m_off + j_loc
                    k_glob = p_off + k_loc
                    idx_glob = j_glob * p + k_glob
                    v_col[idx_glob] = val
            
            # Map W
            for idx_loc in range(p_size * n_size):
                val = W_loc[idx_loc, r]
                if val != 0:
                    k_loc = idx_loc // n_size
                    i_loc = idx_loc % n_size
                    k_glob = p_off + k_loc
                    i_glob = n_off + i_loc
                    idx_glob = k_glob * n + i_glob
                    w_col[idx_glob] = val
            
            U_cols.append(u_col)
            V_cols.append(v_col)
            W_cols.append(w_col)
    
    # Generate Strassen 2x2x2 decomposition
    def get_strassen_2x2x2():
        # Rank 7
        # U, V, W are 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1: (A11+A22)(B11+B22) -> C11 += 1, C22 += 1
        # U: A11 (0,0), A22 (1,1) -> idx 0, 3
        U[0, 0] = 1; U[3, 0] = 1
        # V: B11 (0,0), B22 (1,1) -> idx 0, 3
        V[0, 0] = 1; V[3, 0] = 1
        # W: C11 (k=0,i=0) idx 0, C22 (k=1,i=1) idx 3
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A21+A22)B11 -> C21 += 1, C22 += -1
        # U: A21 (1,0) idx 2, A22 (1,1) idx 3
        U[2, 1] = 1; U[3, 1] = 1
        # V: B11 (0,0) idx 0
        V[0, 1] = 1
        # W: C21 (k=0,i=1) idx 1, C22 (k=1,i=1) idx 3
        W[1, 1] = 1; W[3, 1] = -1
        
        # M3: A11(B21-B22) -> C12 += 1, C22 += 1
        # U: A11 (0,0) idx 0
        U[0, 2] = 1
        # V: B21 (1,0) idx 2, B22 (1,1) idx 3
        V[2, 2] = 1; V[3, 2] = -1
        # W: C12 (k=1,i=0) idx 2, C22 (k=1,i=1) idx 3
        W[2, 2] = 1; W[3, 2] = 1
        
        # M4: A22(B12-B11) -> C11 += 1, C21 += 1
        # U: A22 (1,1) idx 3
        U[3, 3] = 1
        # V: B12 (0,1) idx 1, B11 (0,0) idx 0
        V[1, 3] = 1; V[0, 3] = -1
        # W: C11 (k=0,i=0) idx 0, C21 (k=0,i=1) idx 1
        W[0, 3] = 1; W[1, 3] = 1
        
        # M5: (A11+A12)B22 -> C11 += -1, C12 += 1
        # U: A11 (0,0) idx 0, A12 (0,1) idx 1
        U[0, 4] = 1; U[1, 4] = 1
        # V: B22 (1,1) idx 3
        V[3, 4] = 1
        # W: C11 (k=0,i=0) idx 0, C12 (k=1,i=0) idx 2
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6: (A21-A11)(B11+B12) -> C22 += 1
        # U: A21 (1,0) idx 2, A11 (0,0) idx 0
        U[2, 5] = 1; U[0, 5] = -1
        # V: B11 (0,0) idx 0, B12 (0,1) idx 1
        V[0, 5] = 1; V[1, 5] = 1
        # W: C22 (k=1,i=1) idx 3
        W[3, 5] = 1
        
        # M7: (A12-A22)(B21+B22) -> C11 += 1
        # U: A12 (0,1) idx 1, A22 (1,1) idx 3
        U[1, 6] = 1; U[3, 6] = -1
        # V: B21 (1,0) idx 2, B22 (1,1) idx 3
        V[2, 6] = 1; V[3, 6] = 1
        # W: C11 (k=0,i=0) idx 0
        W[0, 6] = 1
        
        return U, V, W

    # Generate Outer 2x1x2 decomposition
    def get_outer_2x1x2():
        # Rank 4
        # U: 2x1 -> 2 rows. V: 1x2 -> 2 rows. W: 2x2 -> 4 rows.
        U = np.zeros((2, 4))
        V = np.zeros((2, 4))
        W = np.zeros((4, 4))
        
        # C00 = A00 B00
        U[0, 0] = 1
        V[0, 0] = 1
        W[0, 0] = 1 # k=0, i=0 -> 0*2+0=0
        
        # C01 = A00 B01
        U[0, 1] = 1
        V[1, 1] = 1 # k=1
        W[2, 1] = 1 # k=1, i=0 -> 1*2+0=2
        
        # C10 = A10 B00
        U[1, 2] = 1
        V[0, 2] = 1
        W[1, 2] = 1 # k=0, i=1 -> 0*2+1=1
        
        # C11 = A10 B01
        U[1, 3] = 1
        V[1, 3] = 1
        W[3, 3] = 1 # k=1, i=1 -> 1*2+1=3
        
        return U, V, W

    # Iterate over blocks
    for i_idx, n_off in enumerate(n_offsets):
        n_size = n_parts[i_idx]
        for j_idx, m_off in enumerate(m_offsets):
            m_size = m_parts[j_idx]
            for k_idx, p_off in enumerate(p_offsets):
                p_size = p_parts[k_idx]
                
                if n_size == 2 and m_size == 2 and p_size == 2:
                    U_loc, V_loc, W_loc = get_strassen_2x2x2()
                elif n_size == 2 and m_size == 1 and p_size == 2:
                    U_loc, V_loc, W_loc = get_outer_2x1x2()
                else:
                    # Fallback to schoolbook for any other size (should not happen with current partition)
                    rank_fb = n_size * m_size * p_size
                    U_loc = np.eye(n_size * m_size, rank_fb)
                    V_loc = np.eye(m_size * p_size, rank_fb)
                    W_loc = np.zeros((p_size * n_size, rank_fb))
                    col = 0
                    for i in range(n_size):
                        for j in range(m_size):
                            for k in range(p_size):
                                U_loc[i*m_size+j, col] = 1
                                V_loc[j*p_size+k, col] = 1
                                W_loc[k*n_size+i, col] = 1
                                col += 1

                add_decomposition(n_off, m_off, p_off, n_size, m_size, p_size, U_loc, V_loc, W_loc)
    
    # Convert lists to lists of lists (as required by interface)
    U_final = U_cols
    V_final = V_cols
    W_final = W_cols
    
    return {"U": U_final, "V": V_final, "W": W_final}

if __name__ == "__main__":
    # Quick self-test
    res = construct()
    rank = len(res['U'][0])
    print(f"Rank: {rank}")
    # Verify
    n, m, p = DIMENSIONS
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_calc = np.zeros((n, p))
    for r in range(rank):
        u_vec = np.array(res['U'][:, r]).reshape(n, m)
        v_vec = np.array(res['V'][:, r]).reshape(m, p)
        w_vec = np.array(res['W'][:, r]).reshape(p, n)
        
        # L_r = sum(u_vec * A)
        # R_r = sum(v_vec * B)
        L_r = np.sum(u_vec * A)
        R_r = np.sum(v_vec * B)
        
        # C += w_vec * L_r * R_r
        # Note: w_vec is indexed by k*n+i.
        # C[i,k] += w_vec[k*n+i] * L_r * R_r
        # w_vec is flattened p*n.
        # C is n*p.
        # We need to add w_vec * L_r * R_r to C appropriately.
        # w_vec[k*n+i] corresponds to C[i,k].
        # So we reshape w_vec to (p, n) and transpose?
        # w_vec is length p*n. Index k*n+i.
        # C[i,k] is at row i, col k.
        # So we want C += (w_vec * L_r * R_r).T ?
        # w_vec[k*n+i] -> matrix W_mat[k,i].
        # C[i,k] += W_mat[k,i] * L_r * R_r.
        # C += (L_r * R_r) * W_mat.T
        W_mat = w_vec.reshape(p, n)
        C_calc += (L_r * R_r) * W_mat.T
        
    error = np.max(np.abs(C_true - C_calc))
    print(f"Max Error: {error}")
